//===-- Executor.cpp ------------------------------------------------------===//
 
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Executor.h"

#include "../Expr/ArrayExprOptimizer.h"
#include "Context.h"
#include "CoreStats.h"
#include "ExecutorTimerInfo.h"
#include "ExternalDispatcher.h"
#include "ImpliedValue.h"
#include "Memory.h"
#include "MemoryManager.h"
#include "PTree.h"
#include "Searcher.h"
#include "SeedInfo.h"
#include "SpecialFunctionHandler.h"
#include "StatsTracker.h"
#include "TimingSolver.h"
#include "UserSearcher.h"

#include "klee/Common.h"
#include "klee/Config/Version.h"
#include "klee/ExecutionState.h"
#include "klee/Expr.h"
#include "klee/Internal/ADT/KTest.h"
#include "klee/Internal/ADT/RNG.h"
#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "klee/Internal/Support/ErrorHandling.h"
#include "klee/Internal/Support/FileHandling.h"
#include "klee/Internal/Support/FloatEvaluation.h"
#include "klee/Internal/Support/ModuleUtil.h"
#include "klee/Internal/System/MemoryUsage.h"
#include "klee/Internal/System/Time.h"
#include "klee/Interpreter.h"
#include "klee/OptionCategories.h"
#include "klee/SolverCmdLine.h"
#include "klee/SolverStats.h"
#include "klee/TimerStatIncrementer.h"
#include "klee/util/ArrayCache.h"
#include "klee/util/Assignment.h"
#include "klee/util/ExprPPrinter.h"
#include "klee/util/ExprSMTLIBPrinter.h"
#include "klee/util/ExprUtil.h"
#include "klee/util/GetElementPtrTypeIterator.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cxxabi.h>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <vector>

// #define Lewis_DEBUG_SPECU
// #define Lewis_DEBUG_CACHE
#define Lewis_O1
// #define Lewis_O2

using namespace llvm;
using namespace klee;
using namespace libconfig;

namespace klee {
cl::OptionCategory DebugCat("Debugging options",
                            "These are debugging options.");

cl::OptionCategory ExtCallsCat("External call policy options",
                               "These options impact external calls.");

cl::OptionCategory SeedingCat(
    "Seeding options",
    "These options are related to the use of seeds to start exploration.");

cl::OptionCategory
    TerminationCat("State and overall termination options",
                   "These options control termination of the overall KLEE "
                   "execution and of individual states.");

cl::OptionCategory TestGenCat("Test generation options",
                              "These options impact test generation.");
} // namespace klee

namespace {

/*** Test generation options ***/

cl::opt<bool> DumpStatesOnHalt(
    "dump-states-on-halt",
    cl::init(true),
    cl::desc("Dump test cases for all active states on exit (default=true)"),
    cl::cat(TestGenCat));

cl::opt<bool> OnlyOutputStatesCoveringNew(
    "only-output-states-covering-new",
    cl::init(false),
    cl::desc("Only output test cases covering new code (default=false)"),
    cl::cat(TestGenCat));

cl::opt<bool> EmitAllErrors(
    "emit-all-errors", cl::init(false),
    cl::desc("Generate tests cases for all errors "
             "(default=false, i.e. one per (error,instruction) pair)"),
    cl::cat(TestGenCat));


/* Constraint solving options */

cl::opt<unsigned> MaxSymArraySize(
    "max-sym-array-size",
    cl::desc(
        "If a symbolic array exceeds this size (in bytes), symbolic addresses "
        "into this array are concretized.  Set to 0 to disable (default=0)"),
    cl::init(0),
    cl::cat(SolvingCat));

cl::opt<bool>
    SimplifySymIndices("simplify-sym-indices",
                       cl::init(false),
                       cl::desc("Simplify symbolic accesses using equalities "
                                "from other constraints (default=false)"),
                       cl::cat(SolvingCat));

cl::opt<bool>
    EqualitySubstitution("equality-substitution", cl::init(true),
                         cl::desc("Simplify equality expressions before "
                                  "querying the solver (default=true)"),
                         cl::cat(SolvingCat));


/*** External call policy options ***/

enum class ExternalCallPolicy {
  None,     // No external calls allowed
  Concrete, // Only external calls with concrete arguments allowed
  All,      // All external calls allowed
};

cl::opt<ExternalCallPolicy> ExternalCalls(
    "external-calls",
    cl::desc("Specify the external call policy"),
    cl::values(
        clEnumValN(
            ExternalCallPolicy::None, "none",
            "No external function calls are allowed.  Note that KLEE always "
            "allows some external calls with concrete arguments to go through "
            "(in particular printf and puts), regardless of this option."),
        clEnumValN(ExternalCallPolicy::Concrete, "concrete",
                   "Only external function calls with concrete arguments are "
                   "allowed (default)"),
        clEnumValN(ExternalCallPolicy::All, "all",
                   "All external function calls are allowed.  This concretizes "
                   "any symbolic arguments in calls to external functions.")
            KLEE_LLVM_CL_VAL_END),
    cl::init(ExternalCallPolicy::Concrete),
    cl::cat(ExtCallsCat));

cl::opt<bool> SuppressExternalWarnings(
    "suppress-external-warnings",
    cl::init(false),
    cl::desc("Supress warnings about calling external functions."),
    cl::cat(ExtCallsCat));

cl::opt<bool> AllExternalWarnings(
    "all-external-warnings",
    cl::init(false),
    cl::desc("Issue a warning everytime an external call is made, "
             "as opposed to once per function (default=false)"),
    cl::cat(ExtCallsCat));


/*** Seeding options ***/

cl::opt<bool> AlwaysOutputSeeds(
    "always-output-seeds",
    cl::init(true),
    cl::desc(
        "Dump test cases even if they are driven by seeds only (default=true)"),
    cl::cat(SeedingCat));

cl::opt<bool> OnlyReplaySeeds(
    "only-replay-seeds",
    cl::init(false),
    cl::desc("Discard states that do not have a seed (default=false)."),
    cl::cat(SeedingCat));

cl::opt<bool> OnlySeed("only-seed",
                       cl::init(false),
                       cl::desc("Stop execution after seeding is done without "
                                "doing regular search (default=false)."),
                       cl::cat(SeedingCat));

cl::opt<bool>
    AllowSeedExtension("allow-seed-extension",
                       cl::init(false),
                       cl::desc("Allow extra (unbound) values to become "
                                "symbolic during seeding (default=false)."),
                       cl::cat(SeedingCat));

cl::opt<bool> ZeroSeedExtension(
    "zero-seed-extension",
    cl::init(false),
    cl::desc(
        "Use zero-filled objects if matching seed not found (default=false)"),
    cl::cat(SeedingCat));

cl::opt<bool> AllowSeedTruncation(
    "allow-seed-truncation",
    cl::init(false),
    cl::desc("Allow smaller buffers than in seeds (default=false)."),
    cl::cat(SeedingCat));

cl::opt<bool> NamedSeedMatching(
    "named-seed-matching",
    cl::init(false),
    cl::desc("Use names to match symbolic objects to inputs (default=false)."),
    cl::cat(SeedingCat));

cl::opt<std::string>
SeedTime("seed-time",
    cl::desc("Amount of time to dedicate to seeds, before normal "
      "search (default=0s (off))"),
    cl::cat(SeedingCat));


/******************************* Speculative Execution Modeling *****************/
cl::opt<bool> SpeculativeModeling(
    "model-specu",
    cl::init(true),
    cl::desc("Speculative Execution Modeling in KLEE (default=false)"));

cl::opt<unsigned> MaxRBC(
    "max-rbc",
    cl::desc("Max number of instructions in Reorder Buffer. Set to 0 to disable (default=200)"),
    cl::init(200),
    cl::cat(TerminationCat));

cl::opt<unsigned> MaxSpeculativeDepth(
    "max-specu-depth",
    cl::desc("Max number of speculative execution depth. Set to 0 to disable (default=2)"),
    cl::init(2),
    cl::cat(TerminationCat));
/******************************* End Speculative Execution Modeling *****************/


/******************************* Cache Modeling *********************************/
cl::opt<bool> CacheModeling(
    "model-cache",
    cl::init(true),
    cl::desc("Cache Modeling for speculative execution (default=false)"));

cl::opt<std::string> CfgFile(
    "cache-cfg",
    cl::desc("Specify a file path to load cache-related configuration"),
    cl::init("cache.cfg"));

cl::opt<std::string> MemMapFile(
    "mem-map",
    cl::desc("Sepcify a file path to map memory from binary to LLVM"),
    cl::init("memmap.txt"));
    // cl::init("memmap.cfg"));

cl::opt<std::string> KleeMemAccessFile(
    "klee-mem-access",
    cl::desc("Sepcify a file path to read load/store casued by klee instrumentation"),
    cl::init("kleememaccess.txt"));
/******************************* End Cache Modeling *********************************/



/*** Termination criteria options ***/

cl::list<Executor::TerminateReason> ExitOnErrorType(
    "exit-on-error-type",
    cl::desc(
        "Stop execution after reaching a specified condition (default=false)"),
    cl::values(
        clEnumValN(Executor::Abort, "Abort", "The program crashed"),
        clEnumValN(Executor::Assert, "Assert", "An assertion was hit"),
        clEnumValN(Executor::BadVectorAccess, "BadVectorAccess",
                   "Vector accessed out of bounds"),
        clEnumValN(Executor::Exec, "Exec",
                   "Trying to execute an unexpected instruction"),
        clEnumValN(Executor::External, "External",
                   "External objects referenced"),
        clEnumValN(Executor::Free, "Free", "Freeing invalid memory"),
        clEnumValN(Executor::Model, "Model", "Memory model limit hit"),
        clEnumValN(Executor::Overflow, "Overflow", "An overflow occurred"),
        clEnumValN(Executor::Ptr, "Ptr", "Pointer error"),
        clEnumValN(Executor::ReadOnly, "ReadOnly", "Write to read-only memory"),
        clEnumValN(Executor::ReportError, "ReportError",
                   "klee_report_error called"),
        clEnumValN(Executor::User, "User", "Wrong klee_* functions invocation"),
        clEnumValN(Executor::Unhandled, "Unhandled",
                   "Unhandled instruction hit") KLEE_LLVM_CL_VAL_END),
    cl::ZeroOrMore,
    cl::cat(TerminationCat));

cl::opt<unsigned long long> MaxInstructions(
    "max-instructions",
    cl::desc("Stop execution after this many instructions.  Set to 0 to disable (default=0)"),
    cl::init(0),
    cl::cat(TerminationCat));

cl::opt<unsigned>
    MaxForks("max-forks",
             cl::desc("Only fork this many times.  Set to -1 to disable (default=-1)"),
             // cl::init(~0u),
             cl::init(20000u),
             cl::cat(TerminationCat));

cl::opt<unsigned> MaxDepth(
    "max-depth",
    cl::desc("Only allow this many symbolic branches.  Set to 0 to disable (default=0)"),
    cl::init(0),
    cl::cat(TerminationCat));

cl::opt<unsigned> MaxMemory("max-memory",
                            cl::desc("Refuse to fork when above this amount of "
                                     "memory (in MB) (default=2000)"),
                            cl::init(2000),
                            cl::cat(TerminationCat));

cl::opt<bool> MaxMemoryInhibit(
    "max-memory-inhibit",
    cl::desc(
        "Inhibit forking at memory cap (vs. random terminate) (default=true)"),
    cl::init(true),
    cl::cat(TerminationCat));

cl::opt<unsigned> RuntimeMaxStackFrames(
    "max-stack-frames",
    cl::desc("Terminate a state after this many stack frames.  Set to 0 to "
             "disable (default=8192)"),
    cl::init(8192),
    cl::cat(TerminationCat));

cl::opt<std::string> MaxInstructionTime(
    "max-instruction-time",
    cl::desc("Allow a single instruction to take only this much time.  Enables "
             "--use-forked-solver.  Set to 0s to disable (default=0s)"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticForkPct(
    "max-static-fork-pct", cl::init(1.),
    cl::desc("Maximum percentage spent by an instruction forking out of the "
             "forking of all instructions (default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticSolvePct(
    "max-static-solve-pct", cl::init(1.),
    cl::desc("Maximum percentage of solving time that can be spent by a single "
             "instruction over total solving time for all instructions "
             "(default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticCPForkPct(
    "max-static-cpfork-pct", cl::init(1.),
    cl::desc("Maximum percentage spent by an instruction of a call path "
             "forking out of the forking of all instructions in the call path "
             "(default=1.0 (always))"),
    cl::cat(TerminationCat));

cl::opt<double> MaxStaticCPSolvePct(
    "max-static-cpsolve-pct", cl::init(1.),
    cl::desc("Maximum percentage of solving time that can be spent by a single "
             "instruction of a call path over total solving time for all "
             "instructions (default=1.0 (always))"),
    cl::cat(TerminationCat));


/*** Debugging options ***/

/// The different query logging solvers that can switched on/off
enum PrintDebugInstructionsType {
  STDERR_ALL, ///
  STDERR_SRC,
  STDERR_COMPACT,
  FILE_ALL,    ///
  FILE_SRC,    ///
  FILE_COMPACT ///
};

llvm::cl::bits<PrintDebugInstructionsType> DebugPrintInstructions(
    "debug-print-instructions",
    llvm::cl::desc("Log instructions during execution."),
    llvm::cl::values(
        clEnumValN(STDERR_ALL, "all:stderr",
                   "Log all instructions to stderr "
                   "in format [src, inst_id, "
                   "llvm_inst]"),
        clEnumValN(STDERR_SRC, "src:stderr",
                   "Log all instructions to stderr in format [src, inst_id]"),
        clEnumValN(STDERR_COMPACT, "compact:stderr",
                   "Log all instructions to stderr in format [inst_id]"),
        clEnumValN(FILE_ALL, "all:file",
                   "Log all instructions to file "
                   "instructions.txt in format [src, "
                   "inst_id, llvm_inst]"),
        clEnumValN(FILE_SRC, "src:file",
                   "Log all instructions to file "
                   "instructions.txt in format [src, "
                   "inst_id]"),
        clEnumValN(FILE_COMPACT, "compact:file",
                   "Log all instructions to file instructions.txt in format "
                   "[inst_id]") KLEE_LLVM_CL_VAL_END),
    llvm::cl::CommaSeparated,
    cl::cat(DebugCat));

#ifdef HAVE_ZLIB_H
cl::opt<bool> DebugCompressInstructions(
    "debug-compress-instructions", cl::init(false),
    cl::desc(
        "Compress the logged instructions in gzip format (default=false)."),
    cl::cat(DebugCat));
#endif

cl::opt<bool> DebugCheckForImpliedValues(
    "debug-check-for-implied-values", cl::init(false),
    cl::desc("Debug the implied value optimization"),
    cl::cat(DebugCat));

} // namespace

namespace klee {
  RNG theRNG;
}

const char *Executor::TerminateReasonNames[] = {
  [ Abort ] = "abort",
  [ Assert ] = "assert",
  [ BadVectorAccess ] = "bad_vector_access",
  [ Exec ] = "exec",
  [ External ] = "external",
  [ Free ] = "free",
  [ Model ] = "model",
  [ Overflow ] = "overflow",
  [ Ptr ] = "ptr",
  [ ReadOnly ] = "readonly",
  [ ReportError ] = "reporterror",
  [ User ] = "user",
  [ Unhandled ] = "xxx",
};

Executor::Executor(LLVMContext &ctx, const InterpreterOptions &opts,
    InterpreterHandler *ih)
  : Interpreter(opts), interpreterHandler(ih), searcher(0),
  externalDispatcher(new ExternalDispatcher(ctx)), statsTracker(0),
  pathWriter(0), symPathWriter(0), specialFunctionHandler(0),
  processTree(0), replayKTest(0), replayPath(0), usingSeeds(0),
  atMemoryLimit(false), inhibitForking(false), haltExecution(false),
  ivcEnabled(false), debugLogBuffer(debugBufferString) 
{

#ifdef Lewis_O1
  fprintf(stderr, "[+] O1\n");
#endif

#ifdef Lewis_O2
  fprintf(stderr, "[+] O2\n");
#endif

  /********************** Cache Modeling **************************/

  // Load the SpecuSym config file first, sjguo
  if(CacheModeling) {
    loadConfigFile(CfgFile);
    loadMemMapFile(MemMapFile);
    loadKleeMemAccessFile(KleeMemAccessFile);
  }

  /******************** End Cache Modeling ***********************/

  const time::Span maxCoreSolverTime(MaxCoreSolverTime);
  maxInstructionTime = time::Span(MaxInstructionTime);
  coreSolverTimeout = maxCoreSolverTime && maxInstructionTime ?
    std::min(maxCoreSolverTime, maxInstructionTime)
    : std::max(maxCoreSolverTime, maxInstructionTime);

  if (coreSolverTimeout) UseForkedCoreSolver = true;
  Solver *coreSolver = klee::createCoreSolver(CoreSolverToUse);
  if (!coreSolver) {
    klee_error("Failed to create core solver\n");
  }

  Solver *solver = constructSolverChain(
      coreSolver,
      interpreterHandler->getOutputFilename(ALL_QUERIES_SMT2_FILE_NAME),
      interpreterHandler->getOutputFilename(SOLVER_QUERIES_SMT2_FILE_NAME),
      interpreterHandler->getOutputFilename(ALL_QUERIES_KQUERY_FILE_NAME),
      interpreterHandler->getOutputFilename(SOLVER_QUERIES_KQUERY_FILE_NAME));

  this->solver = new TimingSolver(solver, EqualitySubstitution);
  memory = new MemoryManager(&arrayCache);

  initializeSearchOptions();

  if (OnlyOutputStatesCoveringNew && !StatsTracker::useIStats())
    klee_error("To use --only-output-states-covering-new, you need to enable --output-istats.");

  if (DebugPrintInstructions.isSet(FILE_ALL) ||
      DebugPrintInstructions.isSet(FILE_COMPACT) ||
      DebugPrintInstructions.isSet(FILE_SRC)) {
    std::string debug_file_name =
      interpreterHandler->getOutputFilename("instructions.txt");
    std::string error;
#ifdef HAVE_ZLIB_H
    if (!DebugCompressInstructions) {
#endif
      debugInstFile = klee_open_output_file(debug_file_name, error);
#ifdef HAVE_ZLIB_H
    } else {
      debug_file_name.append(".gz");
      debugInstFile = klee_open_compressed_output_file(debug_file_name, error);
    }
#endif
    if (!debugInstFile) {
      klee_error("Could not open file %s : %s", debug_file_name.c_str(),
          error.c_str());
    }
  }
}


llvm::Module *
Executor::setModule(std::vector<std::unique_ptr<llvm::Module>> &modules,
                    const ModuleOptions &opts) {
  assert(!kmodule && !modules.empty() &&
         "can only register one module"); // XXX gross

  kmodule = std::unique_ptr<KModule>(new KModule());

  // Preparing the final module happens in multiple stages

  // Link with KLEE intrinsics library before running any optimizations
  SmallString<128> LibPath(opts.LibraryDir);
  llvm::sys::path::append(LibPath, "libkleeRuntimeIntrinsic.bca");
  std::string error;
  if (!klee::loadFile(LibPath.str(), modules[0]->getContext(), modules,
                      error)) {
    klee_error("Could not load KLEE intrinsic file %s", LibPath.c_str());
  }

  // 1.) Link the modules together
  while (kmodule->link(modules, opts.EntryPoint)) {
    // 2.) Apply different instrumentation
    kmodule->instrument(opts);
  }

  // 3.) Optimise and prepare for KLEE

  // Create a list of functions that should be preserved if used
  std::vector<const char *> preservedFunctions;
  specialFunctionHandler = new SpecialFunctionHandler(*this);
  specialFunctionHandler->prepare(preservedFunctions);

  preservedFunctions.push_back(opts.EntryPoint.c_str());

  // Preserve the free-standing library calls
  preservedFunctions.push_back("memset");
  preservedFunctions.push_back("memcpy");
  preservedFunctions.push_back("memcmp");
  preservedFunctions.push_back("memmove");
/********************** Speculative Execution Modeling ***************/
  // kmodule->optimiseAndPrepare(opts, preservedFunctions);
/****************** End Speculative Execution Modeling ***************/
  kmodule->checkModule();

  // 4.) Manifest the module
  kmodule->manifest(interpreterHandler, StatsTracker::useStatistics());

  specialFunctionHandler->bind();

  if (StatsTracker::useStatistics() || userSearcherRequiresMD2U()) {
    statsTracker = 
      new StatsTracker(*this,
                       interpreterHandler->getOutputFilename("assembly.ll"),
                       userSearcherRequiresMD2U());
  }

  // Initialize the context.
  DataLayout *TD = kmodule->targetData.get();
  Context::initialize(TD->isLittleEndian(),
                      (Expr::Width)TD->getPointerSizeInBits());

  return kmodule->module.get();
}

Executor::~Executor() {
  delete memory;
  delete externalDispatcher;
  delete processTree;
  delete specialFunctionHandler;
  delete statsTracker;
  delete solver;
  while(!timers.empty()) {
    delete timers.back();
    timers.pop_back();
  }
}

/***/

void Executor::initializeGlobalObject(ExecutionState &state, ObjectState *os,
                                      const Constant *c, 
                                      unsigned offset) {
  const auto targetData = kmodule->targetData.get();
  if (const ConstantVector *cp = dyn_cast<ConstantVector>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cp->getType()->getElementType());
    for (unsigned i=0, e=cp->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cp->getOperand(i), 
			     offset + i*elementSize);
  } else if (isa<ConstantAggregateZero>(c)) {
    unsigned i, size = targetData->getTypeStoreSize(c->getType());
    for (i=0; i<size; i++)
      os->write8(offset+i, (uint8_t) 0);
  } else if (const ConstantArray *ca = dyn_cast<ConstantArray>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(ca->getType()->getElementType());
    for (unsigned i=0, e=ca->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, ca->getOperand(i), 
			     offset + i*elementSize);
  } else if (const ConstantStruct *cs = dyn_cast<ConstantStruct>(c)) {
    const StructLayout *sl =
      targetData->getStructLayout(cast<StructType>(cs->getType()));
    for (unsigned i=0, e=cs->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cs->getOperand(i), 
			     offset + sl->getElementOffset(i));
  } else if (const ConstantDataSequential *cds =
               dyn_cast<ConstantDataSequential>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cds->getElementType());
    for (unsigned i=0, e=cds->getNumElements(); i != e; ++i)
      initializeGlobalObject(state, os, cds->getElementAsConstant(i),
                             offset + i*elementSize);
  } else if (!isa<UndefValue>(c) && !isa<MetadataAsValue>(c)) {
    unsigned StoreBits = targetData->getTypeStoreSizeInBits(c->getType());
    ref<ConstantExpr> C = evalConstant(c);

    // Extend the constant if necessary;
    assert(StoreBits >= C->getWidth() && "Invalid store size!");
    if (StoreBits > C->getWidth())
      C = C->ZExt(StoreBits);

    os->write(offset, C);
  }
}

MemoryObject * Executor::addExternalObject(ExecutionState &state, 
                                           void *addr, unsigned size, 
                                           bool isReadOnly) {
  auto mo = memory->allocateFixed(reinterpret_cast<std::uint64_t>(addr),
                                  size, nullptr);
  ObjectState *os = bindObjectInState(state, mo, false);
  for(unsigned i = 0; i < size; i++)
    os->write8(i, ((uint8_t*)addr)[i]);
  if(isReadOnly)
    os->setReadOnly(true);  
  return mo;
}


extern void *__dso_handle __attribute__ ((__weak__));

void Executor::initializeGlobals(ExecutionState &state) {
  Module *m = kmodule->module.get();

  if (m->getModuleInlineAsm() != "")
    klee_warning("executable has module level assembly (ignoring)");
  // represent function globals using the address of the actual llvm function
  // object. given that we use malloc to allocate memory in states this also
  // ensures that we won't conflict. we don't need to allocate a memory object
  // since reading/writing via a function pointer is unsupported anyway.
  for (Module::iterator i = m->begin(), ie = m->end(); i != ie; ++i) {
    Function *f = &*i;
    ref<ConstantExpr> addr(0);

    // If the symbol has external weak linkage then it is implicitly
    // not defined in this module; if it isn't resolvable then it
    // should be null.
    if (f->hasExternalWeakLinkage() && 
        !externalDispatcher->resolveSymbol(f->getName())) {
      addr = Expr::createPointer(0);
    } else {
      addr = Expr::createPointer(reinterpret_cast<std::uint64_t>(f));
      legalFunctions.insert(reinterpret_cast<std::uint64_t>(f));
    }
    
    globalAddresses.insert(std::make_pair(f, addr));
  }

#ifndef WINDOWS
  int *errno_addr = getErrnoLocation(state);
  MemoryObject *errnoObj =
      addExternalObject(state, (void *)errno_addr, sizeof *errno_addr, false);
  // Copy values from and to program space explicitly
  errnoObj->isUserSpecified = true;
#endif

  // Disabled, we don't want to promote use of live externals.
#ifdef HAVE_CTYPE_EXTERNALS
#ifndef WINDOWS
#ifndef DARWIN
  /* from /usr/include/ctype.h:
       These point into arrays of 384, so they can be indexed by any `unsigned
       char' value [0,255]; by EOF (-1); or by any `signed char' value
       [-128,-1).  ISO C requires that the ctype functions work for `unsigned */
  const uint16_t **addr = __ctype_b_loc();
  addExternalObject(state, const_cast<uint16_t*>(*addr-128),
                    384 * sizeof **addr, true);
  addExternalObject(state, addr, sizeof(*addr), true);
    
  const int32_t **lower_addr = __ctype_tolower_loc();
  addExternalObject(state, const_cast<int32_t*>(*lower_addr-128),
                    384 * sizeof **lower_addr, true);
  addExternalObject(state, lower_addr, sizeof(*lower_addr), true);
  
  const int32_t **upper_addr = __ctype_toupper_loc();
  addExternalObject(state, const_cast<int32_t*>(*upper_addr-128),
                    384 * sizeof **upper_addr, true);
  addExternalObject(state, upper_addr, sizeof(*upper_addr), true);
#endif
#endif
#endif

  // allocate and initialize globals, done in two passes since we may
  // need address of a global in order to initialize some other one.

  // allocate memory objects for all globals
  for (Module::const_global_iterator i = m->global_begin(),
         e = m->global_end();
       i != e; ++i) {
    const GlobalVariable *v = &*i;
    size_t globalObjectAlignment = getAllocationAlignment(v);
    if (i->isDeclaration()) {
      // FIXME: We have no general way of handling unknown external
      // symbols. If we really cared about making external stuff work
      // better we could support user definition, or use the EXE style
      // hack where we check the object file information.

      Type *ty = i->getType()->getElementType();
      uint64_t size = 0;
      if (ty->isSized()) {
        size = kmodule->targetData->getTypeStoreSize(ty);
      } else {
        klee_warning("Type for %.*s is not sized", (int)i->getName().size(),
			i->getName().data());
      }

      // XXX - DWD - hardcode some things until we decide how to fix.
#ifndef WINDOWS
      if (i->getName() == "_ZTVN10__cxxabiv117__class_type_infoE") {
        size = 0x2C;
      } else if (i->getName() == "_ZTVN10__cxxabiv120__si_class_type_infoE") {
        size = 0x2C;
      } else if (i->getName() == "_ZTVN10__cxxabiv121__vmi_class_type_infoE") {
        size = 0x2C;
      }
#endif

      if (size == 0) {
        klee_warning("Unable to find size for global variable: %.*s (use will result in out of bounds access)",
			(int)i->getName().size(), i->getName().data());
      }

      /********************** Cache Modeling **************************/

      fprintf(stderr, "[+] Initializing Global %s\n", v->getName().str().c_str());

      MemoryObject* mo = NULL;

      if (CacheModeling && memoryMapFromFile.count(v->getName().str())) {
        mo = memory->allocate(size, /*isLocal=*/false, /*isGlobal=*/true, 
            /*allocSite=*/v, /*alignment=*/globalObjectAlignment, 
            memoryMapFromFile[v->getName().str()]);
      } else
        mo = memory->allocate(size, /*isLocal=*/false, /*isGlobal=*/true, 
            /*allocSite=*/v, /*alignment=*/globalObjectAlignment);

      /********************* End Cache Modeling ***********************/

      ObjectState *os = bindObjectInState(state, mo, false);
      globalObjects.insert(std::make_pair(v, mo));
      globalAddresses.insert(std::make_pair(v, mo->getBaseExpr()));

      // Program already running = object already initialized.  Read
      // concrete value and write it to our copy.
      if (size) {
        void *addr;
        if (i->getName() == "__dso_handle") {
          addr = &__dso_handle; // wtf ?
        } else {
          addr = externalDispatcher->resolveSymbol(i->getName());
        }
        if (!addr)
          klee_error("unable to load symbol(%s) while initializing globals.", 
                     i->getName().data());

        for (unsigned offset=0; offset<mo->size; offset++){
        	os->write8(offset, ((unsigned char*)addr)[offset]);
        }
      }
    } else {
      Type *ty = i->getType()->getElementType();
      uint64_t size = kmodule->targetData->getTypeStoreSize(ty);

      /********************** Cache Modeling **************************/

      fprintf(stderr, "[+] Initializing Global %s\n", v->getName().str().c_str());
      MemoryObject* mo = NULL;

      if (CacheModeling && memoryMapFromFile.count(v->getName().str())) {
        mo = memory->allocate(size, /*isLocal=*/false, /*isGlobal=*/true, 
            /*allocSite=*/v, /*alignment=*/globalObjectAlignment, 
            memoryMapFromFile[v->getName().str()]);
      } else {
        mo = memory->allocate(size, /*isLocal=*/false, /*isGlobal=*/true, 
            /*allocSite=*/v, /*alignment=*/globalObjectAlignment);
      }

      /********************* End Cache Modeling ***********************/
      
      if (!mo)
        llvm::report_fatal_error("out of memory");
      ObjectState *os = bindObjectInState(state, mo, false);
      globalObjects.insert(std::make_pair(v, mo));
      globalAddresses.insert(std::make_pair(v, mo->getBaseExpr()));

      if (!i->hasInitializer())
          os->initializeToRandom();
    }
  }
  
  // link aliases to their definitions (if bound)
  for (auto i = m->alias_begin(), ie = m->alias_end(); i != ie; ++i) {
    // Map the alias to its aliasee's address. This works because we have
    // addresses for everything, even undefined functions.

    // Alias may refer to other alias, not necessarily known at this point.
    // Thus, resolve to real alias directly.
    const GlobalAlias *alias = &*i;
    while (const auto *ga = dyn_cast<GlobalAlias>(alias->getAliasee())) {
      assert(ga != alias && "alias pointing to itself");
      alias = ga;
    }

    globalAddresses.insert(std::make_pair(&*i, evalConstant(alias->getAliasee())));
  }

  // once all objects are allocated, do the actual initialization
  for (Module::const_global_iterator i = m->global_begin(),
         e = m->global_end();
       i != e; ++i) {
    if (i->hasInitializer()) {
      const GlobalVariable *v = &*i;
      MemoryObject *mo = globalObjects.find(v)->second;
      const ObjectState *os = state.addressSpace.findObject(mo);
      assert(os);
      ObjectState *wos = state.addressSpace.getWriteable(mo, os);
      
      initializeGlobalObject(state, wos, i->getInitializer(), 0);
      // if(i->isConstant()) os->setReadOnly(true);
    }
  }
}

void Executor::branch(ExecutionState &state, 
                      const std::vector< ref<Expr> > &conditions,
                      std::vector<ExecutionState*> &result) {
  TimerStatIncrementer timer(stats::forkTime);
  unsigned N = conditions.size();
  assert(N);

  if (MaxForks!=~0u && stats::forks >= MaxForks) {
    unsigned next = theRNG.getInt32() % N;
    for (unsigned i=0; i<N; ++i) {
      if (i == next) {
        result.push_back(&state);
      } else {
        result.push_back(NULL);
      }
    }
  } else {
    stats::forks += N-1;

    // XXX do proper balance or keep random?
    result.push_back(&state);
    for (unsigned i=1; i<N; ++i) {
      ExecutionState *es = result[theRNG.getInt32() % i];
      ExecutionState *ns = es->branch();
      addedStates.push_back(ns);
      result.push_back(ns);
      es->ptreeNode->data = 0;
      std::pair<PTree::Node*,PTree::Node*> res = 
        processTree->split(es->ptreeNode, ns, es);
      ns->ptreeNode = res.first;
      es->ptreeNode = res.second;
    }
  }

  // If necessary redistribute seeds to match conditions, killing
  // states if necessary due to OnlyReplaySeeds (inefficient but
  // simple).
  
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    std::vector<SeedInfo> seeds = it->second;
    seedMap.erase(it);

    // Assume each seed only satisfies one condition (necessarily true
    // when conditions are mutually exclusive and their conjunction is
    // a tautology).
    for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
           siie = seeds.end(); siit != siie; ++siit) {
      unsigned i;
      for (i=0; i<N; ++i) {
        ref<ConstantExpr> res;
        bool success = 
          solver->getValue(state, siit->assignment.evaluate(conditions[i]), 
                           res);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue())
          break;
      }
      
      // If we didn't find a satisfying condition randomly pick one
      // (the seed will be patched).
      if (i==N)
        i = theRNG.getInt32() % N;

      // Extra check in case we're replaying seeds with a max-fork
      if (result[i])
        seedMap[result[i]].push_back(*siit);
    }

    if (OnlyReplaySeeds) {
      for (unsigned i=0; i<N; ++i) {
        if (result[i] && !seedMap.count(result[i])) {
          terminateState(*result[i]);
          result[i] = NULL;
        }
      } 
    }
  }

  for (unsigned i=0; i<N; ++i)
    if (result[i])
      addConstraint(*result[i], conditions[i]);
}


Executor::StatePair 
Executor::fork(ExecutionState &current, ref<Expr> condition, bool isInternal) {
  Solver::Validity res;
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&current);
  bool isSeeding = it != seedMap.end();

  if (!isSeeding && !isa<ConstantExpr>(condition) && 
      (MaxStaticForkPct!=1. || MaxStaticSolvePct != 1. ||
       MaxStaticCPForkPct!=1. || MaxStaticCPSolvePct != 1.) &&
      statsTracker->elapsed() > time::seconds(60)) {
    StatisticManager &sm = *theStatisticManager;
    CallPathNode *cpn = current.stack.back().callPathNode;
    if ((MaxStaticForkPct<1. &&
         sm.getIndexedValue(stats::forks, sm.getIndex()) > 
         stats::forks*MaxStaticForkPct) ||
        (MaxStaticCPForkPct<1. &&
         cpn && (cpn->statistics.getValue(stats::forks) > 
                 stats::forks*MaxStaticCPForkPct)) ||
        (MaxStaticSolvePct<1 &&
         sm.getIndexedValue(stats::solverTime, sm.getIndex()) > 
         stats::solverTime*MaxStaticSolvePct) ||
        (MaxStaticCPForkPct<1. &&
         cpn && (cpn->statistics.getValue(stats::solverTime) > 
                 stats::solverTime*MaxStaticCPSolvePct))) {
      ref<ConstantExpr> value; 
      bool success = solver->getValue(current, condition, value);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      addConstraint(current, EqExpr::create(value, condition));
      condition = value;
    }
  }

  time::Span timeout = coreSolverTimeout;
  if (isSeeding)
    timeout *= static_cast<unsigned>(it->second.size());
  solver->setTimeout(timeout);
  bool success = solver->evaluate(current, condition, res);
  solver->setTimeout(time::Span());
  if (!success) {
    current.pc = current.prevPC;
    terminateStateEarly(current, "Query timed out (fork).");
    return StatePair(0, 0);
  }

  if (!isSeeding) {
    if (replayPath && !isInternal) {
      assert(replayPosition<replayPath->size() &&
             "ran out of branches in replay path mode");
      bool branch = (*replayPath)[replayPosition++];
      
      if (res==Solver::True) {
        assert(branch && "hit invalid branch in replay path mode");
      } else if (res==Solver::False) {
        assert(!branch && "hit invalid branch in replay path mode");
      } else {
        // add constraints
        if(branch) {
          res = Solver::True;
          addConstraint(current, condition);
        } else  {
          res = Solver::False;
          addConstraint(current, Expr::createIsZero(condition));
        }
      }
    } else if (res==Solver::Unknown) {
      assert(!replayKTest && "in replay mode, only one branch can be true.");
      
      if ((MaxMemoryInhibit && atMemoryLimit) || 
          current.forkDisabled ||
          inhibitForking || 
          (MaxForks!=~0u && stats::forks >= MaxForks)) {

	if (MaxMemoryInhibit && atMemoryLimit)
	  klee_warning_once(0, "skipping fork (memory cap exceeded)");
	else if (current.forkDisabled)
	  klee_warning_once(0, "skipping fork (fork disabled on current path)");
	else if (inhibitForking)
	  klee_warning_once(0, "skipping fork (fork disabled globally)");
	else 
	  klee_warning_once(0, "skipping fork (max-forks reached)");

        TimerStatIncrementer timer(stats::forkTime);
        if (theRNG.getBool()) {
          addConstraint(current, condition);
          res = Solver::True;        
        } else {
          addConstraint(current, Expr::createIsZero(condition));
          res = Solver::False;
        }
      }
    }
  }

  // Fix branch in only-replay-seed mode, if we don't have both true
  // and false seeds.
  if (isSeeding && 
      (current.forkDisabled || OnlyReplaySeeds) && 
      res == Solver::Unknown) {
    bool trueSeed=false, falseSeed=false;
    // Is seed extension still ok here?
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      ref<ConstantExpr> res;
      bool success = 
        solver->getValue(current, siit->assignment.evaluate(condition), res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res->isTrue()) {
        trueSeed = true;
      } else {
        falseSeed = true;
      }
      if (trueSeed && falseSeed)
        break;
    }
    if (!(trueSeed && falseSeed)) {
      assert(trueSeed || falseSeed);
      
      res = trueSeed ? Solver::True : Solver::False;
      addConstraint(current, trueSeed ? condition : Expr::createIsZero(condition));
    }
  }


  // XXX - even if the constraint is provable one way or the other we
  // can probably benefit by adding this constraint and allowing it to
  // reduce the other constraints. For example, if we do a binary
  // search on a particular value, and then see a comparison against
  // the value it has been fixed at, we should take this as a nice
  // hint to just use the single constraint instead of all the binary
  // search ones. If that makes sense.
  if (res==Solver::True) {
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "1";
      }
    }

    return StatePair(&current, 0);
  } else if (res==Solver::False) {
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "0";
      }
    }

    return StatePair(0, &current);
  } else {
    TimerStatIncrementer timer(stats::forkTime);
    ExecutionState *falseState, *trueState = &current;

    ++stats::forks;

    falseState = trueState->branch();
    addedStates.push_back(falseState);

    if (it != seedMap.end()) {
      std::vector<SeedInfo> seeds = it->second;
      it->second.clear();
      std::vector<SeedInfo> &trueSeeds = seedMap[trueState];
      std::vector<SeedInfo> &falseSeeds = seedMap[falseState];
      for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
             siie = seeds.end(); siit != siie; ++siit) {
        ref<ConstantExpr> res;
        bool success = 
          solver->getValue(current, siit->assignment.evaluate(condition), res);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue()) {
          trueSeeds.push_back(*siit);
        } else {
          falseSeeds.push_back(*siit);
        }
      }
      
      bool swapInfo = false;
      if (trueSeeds.empty()) {
        if (&current == trueState) swapInfo = true;
        seedMap.erase(trueState);
      }
      if (falseSeeds.empty()) {
        if (&current == falseState) swapInfo = true;
        seedMap.erase(falseState);
      }
      if (swapInfo) {
        std::swap(trueState->coveredNew, falseState->coveredNew);
        std::swap(trueState->coveredLines, falseState->coveredLines);
      }
    }

    current.ptreeNode->data = 0;
    std::pair<PTree::Node*, PTree::Node*> res =
      processTree->split(current.ptreeNode, falseState, trueState);
    falseState->ptreeNode = res.first;
    trueState->ptreeNode = res.second;

    if (pathWriter) {
      // Need to update the pathOS.id field of falseState, otherwise the same id
      // is used for both falseState and trueState.
      falseState->pathOS = pathWriter->open(current.pathOS);
      if (!isInternal) {
        trueState->pathOS << "1";
        falseState->pathOS << "0";
      }
    }
    if (symPathWriter) {
      falseState->symPathOS = symPathWriter->open(current.symPathOS);
      if (!isInternal) {
        trueState->symPathOS << "1";
        falseState->symPathOS << "0";
      }
    }

    addConstraint(*trueState, condition);
    addConstraint(*falseState, Expr::createIsZero(condition));

    // Kinda gross, do we even really still want this option?
    if (MaxDepth && MaxDepth<=trueState->depth) {
      terminateStateEarly(*trueState, "max-depth exceeded.");
      terminateStateEarly(*falseState, "max-depth exceeded.");
      return StatePair(0, 0);
    }

    return StatePair(trueState, falseState);
  }
}

void Executor::addConstraint(ExecutionState &state, ref<Expr> condition) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(condition)) {
    if (!CE->isTrue())
      llvm::report_fatal_error("attempt to add invalid constraint");
    return;
  }

  // Check to see if this constraint violates seeds.
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    bool warn = false;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      bool res;
      bool success = 
        solver->mustBeFalse(state, siit->assignment.evaluate(condition), res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        siit->patchSeed(state, condition, solver);
        warn = true;
      }
    }
    if (warn)
      klee_warning("seeds patched for violating constraint"); 
  }

  state.addConstraint(condition);
  if (ivcEnabled)
    doImpliedValueConcretization(state, condition, 
                                 ConstantExpr::alloc(1, Expr::Bool));
}

const Cell& Executor::eval(KInstruction *ki, unsigned index, 
                           ExecutionState &state) const {
  assert(index < ki->inst->getNumOperands());
  int vnumber = ki->operands[index];

  assert(vnumber != -1 &&
         "Invalid operand to eval(), not a value or constant!");

  // Determine if this is a constant or not.
  if (vnumber < 0) {
    unsigned index = -vnumber - 2;
    return kmodule->constantTable[index];
  } else {
    unsigned index = vnumber;
    StackFrame &sf = state.stack.back();
    return sf.locals[index];
  }
}

void Executor::bindLocal(KInstruction *target, ExecutionState &state, 
                         ref<Expr> value) {
  getDestCell(state, target).value = value;
}

void Executor::bindArgument(KFunction *kf, unsigned index, 
                            ExecutionState &state, ref<Expr> value) {
  getArgumentCell(state, kf, index).value = value;
}

ref<Expr> Executor::toUnique(const ExecutionState &state, 
                             ref<Expr> &e) {
  ref<Expr> result = e;

  if (!isa<ConstantExpr>(e)) {
    ref<ConstantExpr> value;
    bool isTrue = false;
    e = optimizer.optimizeExpr(e, true);
    solver->setTimeout(coreSolverTimeout);
    if (solver->getValue(state, e, value)) {
      ref<Expr> cond = EqExpr::create(e, value);
      cond = optimizer.optimizeExpr(cond, false);
      if (solver->mustBeTrue(state, cond, isTrue) && isTrue)
        result = value;
    }
    solver->setTimeout(time::Span());
  }
  
  return result;
}


/* Concretize the given expression, and return a possible constant value. 
   'reason' is just a documentation string stating the reason for concretization. */
ref<klee::ConstantExpr> 
Executor::toConstant(ExecutionState &state, 
                     ref<Expr> e,
                     const char *reason) {
  e = state.constraints.simplifyExpr(e);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(e))
    return CE;

  ref<ConstantExpr> value;
  bool success = solver->getValue(state, e, value);
  assert(success && "FIXME: Unhandled solver failure");
  (void) success;

  std::string str;
  llvm::raw_string_ostream os(str);
  os << "silently concretizing (reason: " << reason << ") expression " << e
     << " to value " << value << " (" << (*(state.pc)).info->file << ":"
     << (*(state.pc)).info->line << ")";

  if (AllExternalWarnings)
    klee_warning("%s", os.str().c_str());
  else
    klee_warning_once(reason, "%s", os.str().c_str());

  addConstraint(state, EqExpr::create(e, value));
    
  return value;
}

void Executor::executeGetValue(ExecutionState &state,
                               ref<Expr> e,
                               KInstruction *target) {
  e = state.constraints.simplifyExpr(e);
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it==seedMap.end() || isa<ConstantExpr>(e)) {
    ref<ConstantExpr> value;
    e = optimizer.optimizeExpr(e, true);
    bool success = solver->getValue(state, e, value);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    bindLocal(target, state, value);
  } else {
    std::set< ref<Expr> > values;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      ref<Expr> cond = siit->assignment.evaluate(e);
      cond = optimizer.optimizeExpr(cond, true);
      ref<ConstantExpr> value;
      bool success = solver->getValue(state, cond, value);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      values.insert(value);
    }
    
    std::vector< ref<Expr> > conditions;
    for (std::set< ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit)
      conditions.push_back(EqExpr::create(e, *vit));

    std::vector<ExecutionState*> branches;
    branch(state, conditions, branches);
    
    std::vector<ExecutionState*>::iterator bit = branches.begin();
    for (std::set< ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit) {
      ExecutionState *es = *bit;
      if (es)
        bindLocal(target, *es, *vit);
      ++bit;
    }
  }
}

void Executor::printDebugInstructions(ExecutionState &state) {
  // check do not print
  if (DebugPrintInstructions.getBits() == 0)
	  return;

  llvm::raw_ostream *stream = 0;
  if (DebugPrintInstructions.isSet(STDERR_ALL) ||
      DebugPrintInstructions.isSet(STDERR_SRC) ||
      DebugPrintInstructions.isSet(STDERR_COMPACT))
    stream = &llvm::errs();
  else
    stream = &debugLogBuffer;

  if (!DebugPrintInstructions.isSet(STDERR_COMPACT) &&
      !DebugPrintInstructions.isSet(FILE_COMPACT)) {
    (*stream) << "     " << state.pc->getSourceLocation() << ":";
  }

  (*stream) << state.pc->info->assemblyLine;

  if (DebugPrintInstructions.isSet(STDERR_ALL) ||
      DebugPrintInstructions.isSet(FILE_ALL))
    (*stream) << ":" << *(state.pc->inst);
  (*stream) << "\n";

  if (DebugPrintInstructions.isSet(FILE_ALL) ||
      DebugPrintInstructions.isSet(FILE_COMPACT) ||
      DebugPrintInstructions.isSet(FILE_SRC)) {
    debugLogBuffer.flush();
    (*debugInstFile) << debugLogBuffer.str();
    debugBufferString = "";
  }
}

void Executor::stepInstruction(ExecutionState &state) {
  printDebugInstructions(state);
  if (statsTracker)
    statsTracker->stepInstruction(state);

  ++stats::instructions;
  ++state.steppedInstructions;
  state.prevPC = state.pc;
  ++state.pc;

  if (stats::instructions == MaxInstructions)
    haltExecution = true;
}

static inline const llvm::fltSemantics *fpWidthToSemantics(unsigned width) {
  switch (width) {
#if LLVM_VERSION_CODE >= LLVM_VERSION(4, 0)
  case Expr::Int32:
    return &llvm::APFloat::IEEEsingle();
  case Expr::Int64:
    return &llvm::APFloat::IEEEdouble();
  case Expr::Fl80:
    return &llvm::APFloat::x87DoubleExtended();
#else
  case Expr::Int32:
    return &llvm::APFloat::IEEEsingle;
  case Expr::Int64:
    return &llvm::APFloat::IEEEdouble;
  case Expr::Fl80:
    return &llvm::APFloat::x87DoubleExtended;
#endif
  default:
    return 0;
  }
}

void Executor::executeCall(ExecutionState &state, 
                           KInstruction *ki,
                           Function *f,
                           std::vector< ref<Expr> > &arguments) {
  Instruction *i = ki->inst;
  if (i && isa<DbgInfoIntrinsic>(i))
    return;
  if (f && f->isDeclaration()) {
    switch(f->getIntrinsicID()) {
    case Intrinsic::not_intrinsic:
      // state may be destroyed by this call, cannot touch
      callExternalFunction(state, ki, f, arguments);
      break;
    case Intrinsic::fabs: {
      ref<ConstantExpr> arg =
          toConstant(state, eval(ki, 0, state).value, "floating point");
      if (!fpWidthToSemantics(arg->getWidth()))
        return terminateStateOnExecError(
            state, "Unsupported intrinsic llvm.fabs call");

      llvm::APFloat Res(*fpWidthToSemantics(arg->getWidth()),
                        arg->getAPValue());
      Res = llvm::abs(Res);

      bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
      break;
    }
    // va_arg is handled by caller and intrinsic lowering, see comment for
    // ExecutionState::varargs
    case Intrinsic::vastart:  {
      StackFrame &sf = state.stack.back();

      // varargs can be zero if no varargs were provided
      if (!sf.varargs)
        return;

      // FIXME: This is really specific to the architecture, not the pointer
      // size. This happens to work for x86-32 and x86-64, however.
      Expr::Width WordSize = Context::get().getPointerWidth();
      if (WordSize == Expr::Int32) {
        executeMemoryOperation(state, true, arguments[0], 
                               sf.varargs->getBaseExpr(), 0);
      } else {
        assert(WordSize == Expr::Int64 && "Unknown word size!");

        // x86-64 has quite complicated calling convention. However,
        // instead of implementing it, we can do a simple hack: just
        // make a function believe that all varargs are on stack.
        executeMemoryOperation(state, true, arguments[0],
                               ConstantExpr::create(48, 32), 0); // gp_offset
        executeMemoryOperation(state, true,
                               AddExpr::create(arguments[0], 
                                               ConstantExpr::create(4, 64)),
                               ConstantExpr::create(304, 32), 0); // fp_offset
        executeMemoryOperation(state, true,
                               AddExpr::create(arguments[0], 
                                               ConstantExpr::create(8, 64)),
                               sf.varargs->getBaseExpr(), 0); // overflow_arg_area
        executeMemoryOperation(state, true,
                               AddExpr::create(arguments[0], 
                                               ConstantExpr::create(16, 64)),
                               ConstantExpr::create(0, 64), 0); // reg_save_area
      }
      break;
    }
    case Intrinsic::vaend:
      // va_end is a noop for the interpreter.
      //
      // FIXME: We should validate that the target didn't do something bad
      // with va_end, however (like call it twice).
      break;
        
    case Intrinsic::vacopy:
      // va_copy should have been lowered.
      //
      // FIXME: It would be nice to check for errors in the usage of this as
      // well.
    default:
      klee_error("unknown intrinsic: %s", f->getName().data());
    }

    if (InvokeInst *ii = dyn_cast<InvokeInst>(i))
      transferToBasicBlock(ii->getNormalDest(), i->getParent(), state);
  } else {
    // Check if maximum stack size was reached.
    // We currently only count the number of stack frames
    if (RuntimeMaxStackFrames && state.stack.size() > RuntimeMaxStackFrames) {
      terminateStateEarly(state, "Maximum stack size reached.");
      klee_warning("Maximum stack size reached.");
      return;
    }

    // FIXME: I'm not really happy about this reliance on prevPC but it is ok, I
    // guess. This just done to avoid having to pass KInstIterator everywhere
    // instead of the actual instruction, since we can't make a KInstIterator
    // from just an instruction (unlike LLVM).
    KFunction *kf = kmodule->functionMap[f];

    state.pushFrame(state.prevPC, kf);
    state.pc = kf->instructions;

    if (statsTracker)
      statsTracker->framePushed(state, &state.stack[state.stack.size()-2]);

     // TODO: support "byval" parameter attribute
     // TODO: support zeroext, signext, sret attributes

    unsigned callingArgs = arguments.size();
    unsigned funcArgs = f->arg_size();
    if (!f->isVarArg()) {
      if (callingArgs > funcArgs) {
        klee_warning_once(f, "calling %s with extra arguments.", 
                          f->getName().data());
      } else if (callingArgs < funcArgs) {
        terminateStateOnError(state, "calling function with too few arguments",
                              User);
        return;
      }
    } else {
      Expr::Width WordSize = Context::get().getPointerWidth();

      if (callingArgs < funcArgs) {
        terminateStateOnError(state, "calling function with too few arguments",
                              User);
        return;
      }

      StackFrame &sf = state.stack.back();
      unsigned size = 0;
      bool requires16ByteAlignment = false;
      for (unsigned i = funcArgs; i < callingArgs; i++) {
        // FIXME: This is really specific to the architecture, not the pointer
        // size. This happens to work for x86-32 and x86-64, however.
        if (WordSize == Expr::Int32) {
          size += Expr::getMinBytesForWidth(arguments[i]->getWidth());
        } else {
          Expr::Width argWidth = arguments[i]->getWidth();
          // AMD64-ABI 3.5.7p5: Step 7. Align l->overflow_arg_area upwards to a
          // 16 byte boundary if alignment needed by type exceeds 8 byte
          // boundary.
          //
          // Alignment requirements for scalar types is the same as their size
          if (argWidth > Expr::Int64) {
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 9)
             size = llvm::alignTo(size, 16);
#else
             size = llvm::RoundUpToAlignment(size, 16);
#endif
             requires16ByteAlignment = true;
          }
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 9)
          size += llvm::alignTo(argWidth, WordSize) / 8;
#else
          size += llvm::RoundUpToAlignment(argWidth, WordSize) / 8;
#endif
        }
      }

      /********************** Cache Modeling **************************/

      MemoryObject* mo = NULL;
      if (CacheModeling) {
#ifdef Lewis_DEBUG_CACHE
        fprintf(stderr, "[+] Local alloc in vacopy in size %u\n", size);
#endif
        state.rsp -= size; // little endian
        mo = sf.varargs = memory->allocate(size, true, false, state.prevPC->inst,
            (requires16ByteAlignment ? 16 : 8), state.rsp);
      } else {
        mo = sf.varargs = memory->allocate(size, true, false, state.prevPC->inst,
            (requires16ByteAlignment ? 16 : 8));
      }

      /******************* End Cache Modeling *************************/

      if (!mo && size) {
        terminateStateOnExecError(state, "out of memory (varargs)");
        return;
      }

      if (mo) {
        if ((WordSize == Expr::Int64) && (mo->address & 15) &&
            requires16ByteAlignment) {
          // Both 64bit Linux/Glibc and 64bit MacOSX should align to 16 bytes.
          klee_warning_once(
              0, "While allocating varargs: malloc did not align to 16 bytes.");
        }

        ObjectState *os = bindObjectInState(state, mo, true);
        unsigned offset = 0;
        for (unsigned i = funcArgs; i < callingArgs; i++) {
          // FIXME: This is really specific to the architecture, not the pointer
          // size. This happens to work for x86-32 and x86-64, however.
          if (WordSize == Expr::Int32) {
            os->write(offset, arguments[i]);
            offset += Expr::getMinBytesForWidth(arguments[i]->getWidth());
          } else {
            assert(WordSize == Expr::Int64 && "Unknown word size!");

            Expr::Width argWidth = arguments[i]->getWidth();
            if (argWidth > Expr::Int64) {
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 9)
              offset = llvm::alignTo(offset, 16);
#else
              offset = llvm::RoundUpToAlignment(offset, 16);
#endif
            }
            os->write(offset, arguments[i]);
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 9)
            offset += llvm::alignTo(argWidth, WordSize) / 8;
#else
            offset += llvm::RoundUpToAlignment(argWidth, WordSize) / 8;
#endif
          }
        }
      }
    }

    unsigned numFormals = f->arg_size();
    for (unsigned i=0; i<numFormals; ++i) 
      bindArgument(kf, i, state, arguments[i]);
  }
}

void Executor::transferToBasicBlock(BasicBlock *dst, BasicBlock *src, 
                                    ExecutionState &state) {
  // Note that in general phi nodes can reuse phi values from the same
  // block but the incoming value is the eval() result *before* the
  // execution of any phi nodes. this is pathological and doesn't
  // really seem to occur, but just in case we run the PhiCleanerPass
  // which makes sure this cannot happen and so it is safe to just
  // eval things in order. The PhiCleanerPass also makes sure that all
  // incoming blocks have the same order for each PHINode so we only
  // have to compute the index once.
  //
  // With that done we simply set an index in the state so that PHI
  // instructions know which argument to eval, set the pc, and continue.
  
  // XXX this lookup has to go ?
  KFunction *kf = state.stack.back().kf;
  unsigned entry = kf->basicBlockEntry[dst];
  state.pc = &kf->instructions[entry];
  if (state.pc->inst->getOpcode() == Instruction::PHI) {
    PHINode *first = static_cast<PHINode*>(state.pc->inst);
    state.incomingBBIndex = first->getBasicBlockIndex(src);
  }
}

/// Compute the true target of a function call, resolving LLVM aliases
/// and bitcasts.
Function* Executor::getTargetFunction(Value *calledVal, ExecutionState &state) {
  SmallPtrSet<const GlobalValue*, 3> Visited;

  Constant *c = dyn_cast<Constant>(calledVal);
  if (!c)
    return 0;

  while (true) {
    if (GlobalValue *gv = dyn_cast<GlobalValue>(c)) {
      if (!Visited.insert(gv).second)
        return 0;

      if (Function *f = dyn_cast<Function>(gv))
        return f;
      else if (GlobalAlias *ga = dyn_cast<GlobalAlias>(gv))
        c = ga->getAliasee();
      else
        return 0;
    } else if (llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(c)) {
      if (ce->getOpcode()==Instruction::BitCast)
        c = ce->getOperand(0);
      else
        return 0;
    } else
      return 0;
  }
}

void Executor::executeInstruction(ExecutionState &state, KInstruction *ki) {
  Instruction *i = ki->inst;

  switch (i->getOpcode()) {
    // Control flow
  case Instruction::Ret: {
    /********************** Speculative Execution Modeling ***************/
    if (SpeculativeModeling && state.stateType == ExecutionState::SPECULATIVE) {
#ifdef Lewis_DEBUG_SPECU
      fprintf(stderr, "[+] return in speculative state, terminate\n");
#endif
      stopSpeculativeExecution(state);
      terminateStateEarly(state, "Speculative execution reaches threshold");
      return;
    }
    /****************** End Speculative Execution Modeling ***************/
    ReturnInst *ri = cast<ReturnInst>(i);
    KInstIterator kcaller = state.stack.back().caller;
    Instruction *caller = kcaller ? kcaller->inst : 0;
    bool isVoidReturn = (ri->getNumOperands() == 0);
    ref<Expr> result = ConstantExpr::alloc(0, Expr::Bool);
    
    if (!isVoidReturn) {
      result = eval(ki, 0, state).value;
    }
    
    if (state.stack.size() <= 1) {
      assert(!caller && "caller set on initial stack frame");
      terminateStateOnExit(state);
    } else {
      state.popFrame();

      if (statsTracker)
        statsTracker->framePopped(state);

      if (InvokeInst *ii = dyn_cast<InvokeInst>(caller)) {
        transferToBasicBlock(ii->getNormalDest(), caller->getParent(), state);
      } else {
        state.pc = kcaller;
        ++state.pc;
      }

      if (!isVoidReturn) {
        Type *t = caller->getType();
        if (t != Type::getVoidTy(i->getContext())) {
          // may need to do coercion due to bitcasts
          Expr::Width from = result->getWidth();
          Expr::Width to = getWidthForLLVMType(t);
            
          if (from != to) {
            CallSite cs = (isa<InvokeInst>(caller) ? CallSite(cast<InvokeInst>(caller)) : 
                           CallSite(cast<CallInst>(caller)));

            // XXX need to check other param attrs ?
#if LLVM_VERSION_CODE >= LLVM_VERSION(5, 0)
            bool isSExt = cs.hasRetAttr(llvm::Attribute::SExt);
#else
            bool isSExt = cs.paramHasAttr(0, llvm::Attribute::SExt);
#endif
            if (isSExt) {
              result = SExtExpr::create(result, to);
            } else {
              result = ZExtExpr::create(result, to);
            }
          }

          bindLocal(kcaller, state, result);
        }
      } else {
        // We check that the return value has no users instead of
        // checking the type, since C defaults to returning int for
        // undeclared functions.
        if (!caller->use_empty()) {
          terminateStateOnExecError(state, "return void when caller expected a result");
        }
      }
    }      
    /********************** Cache Modeling **************************/
    state.rsp = state.old_rsp.back();
    state.old_rsp.pop_back();
    /******************* End Cache Modeling *************************/
    break;
  }
  case Instruction::Br: {
  	BranchInst *bi = cast<BranchInst>(i);
  	if (bi->isUnconditional()) {

      /********************** Speculative Execution Modeling ***************/

      /*
      if (SpeculativeModeling && state.stateType == ExecutionState::SPECULATIVE) {
#ifdef Lewis_DEBUG_SPECU
        fprintf(stderr, "[+] unconditional branch in speculative state, terminate\n");
#endif
        stopSpeculativeExecution(state);
        terminateStateEarly(state, "nested speculative execution.");
        */
      /****************** End Speculative Execution Modeling ***************/

      //} else {
        transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), state);
      //}
  	} else {
      // FIXME: Find a way that we don't have this hidden dependency.
      assert(bi->getCondition() == bi->getOperand(0) && "Wrong operand index!");
      ref<Expr> cond = eval(ki, 0, state).value;

      cond = optimizer.optimizeExpr(cond, false);
      Executor::StatePair branches;

      /********************** Speculative Execution Modeling ***************/

      if (SpeculativeModeling) {

        /*
        ExecutionState* specu1 = NULL;
        ExecutionState* specu2 = NULL;
        fprintf(stderr, "[+] wanna fork, nested depth %u\n", state.nestedCnt);
        branches = fork(state, cond, false, specu1, specu2);

        if (branches.first) {
          fprintf(stderr, "[+] fork symbolically (%lu-%lu)\n", 
              state.id, branches.first->id);
          if (specu1) {
            fprintf(stderr, "[+] fork speculatively (%lu-%lu)\n", 
                branches.first->id, specu1->id);
            branches.first->continueFlag = false;
            transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
            specu1->continueFlag = true;
            transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *specu1);
          } else {
            transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
          }
        }

        if (branches.second) {
          fprintf(stderr, "[+] fork symbolically (%lu-%lu)\n", 
              state.id, branches.second->id);
          if (specu2) {
            fprintf(stderr, "[+] fork speculatively (%lu-%lu)\n",
                branches.second->id, specu2->id);
            branches.second->continueFlag = false;
            transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
            specu2->continueFlag = true;
            transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *specu2);
          } else {
            transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
          }
        }
        */

        if (state.stateType == ExecutionState::SPECULATIVE) {
          // XXX for now, we do not consider nested speculative execution
#ifdef Lewis_DEBUG_SPECU
            fprintf(stderr, "[+] nested speculative state, terminate\n");
#endif
            stopSpeculativeExecution(state);
            terminateStateEarly(state, "nested speculative execution.");
            return;
        } else {
          // Fork speculative states on interpretating a regular symbolic state
          assert(state.stateType == ExecutionState::SYMBOLIC);
          ExecutionState* specu1 = NULL;
          ExecutionState* specu2 = NULL;
          fprintf(stderr, "[+] wanna fork, nested depth %u\n", state.nestedCnt);
          branches = fork(state, cond, false, specu1, specu2);

          if (branches.first) {
            fprintf(stderr, "[+] fork symbolically (%lu-%lu)\n", 
                state.id, branches.first->id);
            if (specu1) {
              fprintf(stderr, "[+] fork speculatively (%lu-%lu)\n", 
                  branches.first->id, specu1->id);
              branches.first->continueFlag = false;
              transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
              specu1->continueFlag = true;
              transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *specu1);
            } else {
              transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
            }
          }
          
          if ( branches.second) {
            fprintf(stderr, "[+] fork symbolically (%lu-%lu)\n", 
                state.id, branches.second->id);
            if (specu2) {
              fprintf(stderr, "[+] fork speculatively (%lu-%lu)\n",
                  branches.second->id, specu2->id);
              branches.second->continueFlag = false;
              transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
              specu2->continueFlag = true;
              transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *specu2);
            } else {
              transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
            }
          }
        }
      } else {
        // The non-speculative execution
        branches = fork(state, cond, false);
        // NOTE: There is a hidden dependency here, markBranchVisited
        // requires that we still be in the context of the branch
        // instruction (it reuses its statistic id). Should be cleaned
        // up with convenient instruction specific data.
        if (statsTracker && state.stack.back().kf->trackCoverage)
          statsTracker->markBranchVisited(branches.first, branches.second);

        if (branches.first)
          transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
        if (branches.second)
          transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
  		}
  	}
  	break;
  }
  case Instruction::IndirectBr: {
    // implements indirect branch to a label within the current function
    const auto bi = cast<IndirectBrInst>(i);
    auto address = eval(ki, 0, state).value;
    address = toUnique(state, address);

    // concrete address
    if (const auto CE = dyn_cast<ConstantExpr>(address.get())) {
      const auto bb_address = (BasicBlock *) CE->getZExtValue(Context::get().getPointerWidth());
      transferToBasicBlock(bb_address, bi->getParent(), state);
      break;
    }

    // symbolic address
    const auto numDestinations = bi->getNumDestinations();
    std::vector<BasicBlock *> targets;
    targets.reserve(numDestinations);
    std::vector<ref<Expr>> expressions;
    expressions.reserve(numDestinations);

    ref<Expr> errorCase = ConstantExpr::alloc(1, Expr::Bool);
    SmallPtrSet<BasicBlock *, 5> destinations;
    // collect and check destinations from label list
    for (unsigned k = 0; k < numDestinations; ++k) {
      // filter duplicates
      const auto d = bi->getDestination(k);
      if (destinations.count(d)) continue;
      destinations.insert(d);

      // create address expression
      const auto PE = Expr::createPointer(reinterpret_cast<std::uint64_t>(d));
      ref<Expr> e = EqExpr::create(address, PE);

      // exclude address from errorCase
      errorCase = AndExpr::create(errorCase, Expr::createIsZero(e));

      // check feasibility
      bool result;
      bool success __attribute__ ((unused)) = solver->mayBeTrue(state, e, result);
      assert(success && "FIXME: Unhandled solver failure");
      if (result) {
        targets.push_back(d);
        expressions.push_back(e);
      }
    }
    // check errorCase feasibility
    bool result;
    bool success __attribute__ ((unused)) = solver->mayBeTrue(state, errorCase, result);
    assert(success && "FIXME: Unhandled solver failure");
    if (result) {
      expressions.push_back(errorCase);
    }

    // fork states
    std::vector<ExecutionState *> branches;
    branch(state, expressions, branches);

    // terminate error state
    if (result) {
      terminateStateOnExecError(*branches.back(), "indirectbr: illegal label address");
      branches.pop_back();
    }

    // branch states to resp. target blocks
    assert(targets.size() == branches.size());
    for (std::vector<ExecutionState *>::size_type k = 0; k < branches.size(); ++k) {
      if (branches[k]) {
        transferToBasicBlock(targets[k], bi->getParent(), *branches[k]);
      }
    }

    break;
  }
  case Instruction::Switch: {
    SwitchInst *si = cast<SwitchInst>(i);
    ref<Expr> cond = eval(ki, 0, state).value;
    BasicBlock *bb = si->getParent();

    cond = toUnique(state, cond);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(cond)) {
      // Somewhat gross to create these all the time, but fine till we
      // switch to an internal rep.
      llvm::IntegerType *Ty = cast<IntegerType>(si->getCondition()->getType());
      ConstantInt *ci = ConstantInt::get(Ty, CE->getZExtValue());
#if LLVM_VERSION_CODE >= LLVM_VERSION(5, 0)
      unsigned index = si->findCaseValue(ci)->getSuccessorIndex();
#else
      unsigned index = si->findCaseValue(ci).getSuccessorIndex();
#endif
      transferToBasicBlock(si->getSuccessor(index), si->getParent(), state);
    } else {
      // Handle possible different branch targets

      // We have the following assumptions:
      // - each case value is mutual exclusive to all other values including the
      //   default value
      // - order of case branches is based on the order of the expressions of
      //   the scase values, still default is handled last
      std::vector<BasicBlock *> bbOrder;
      std::map<BasicBlock *, ref<Expr> > branchTargets;

      std::map<ref<Expr>, BasicBlock *> expressionOrder;

      // Iterate through all non-default cases and order them by expressions
      for (auto i : si->cases()) {
        ref<Expr> value = evalConstant(i.getCaseValue());

        BasicBlock *caseSuccessor = i.getCaseSuccessor();
        expressionOrder.insert(std::make_pair(value, caseSuccessor));
      }

      // Track default branch values
      ref<Expr> defaultValue = ConstantExpr::alloc(1, Expr::Bool);

      // iterate through all non-default cases but in order of the expressions
      for (std::map<ref<Expr>, BasicBlock *>::iterator
               it = expressionOrder.begin(),
               itE = expressionOrder.end();
           it != itE; ++it) {
        ref<Expr> match = EqExpr::create(cond, it->first);

        // Make sure that the default value does not contain this target's value
        defaultValue = AndExpr::create(defaultValue, Expr::createIsZero(match));

        // Check if control flow could take this case
        bool result;
        match = optimizer.optimizeExpr(match, false);
        bool success = solver->mayBeTrue(state, match, result);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (result) {
          BasicBlock *caseSuccessor = it->second;

          // Handle the case that a basic block might be the target of multiple
          // switch cases.
          // Currently we generate an expression containing all switch-case
          // values for the same target basic block. We spare us forking too
          // many times but we generate more complex condition expressions
          // TODO Add option to allow to choose between those behaviors
          std::pair<std::map<BasicBlock *, ref<Expr> >::iterator, bool> res =
              branchTargets.insert(std::make_pair(
                  caseSuccessor, ConstantExpr::alloc(0, Expr::Bool)));

          res.first->second = OrExpr::create(match, res.first->second);

          // Only add basic blocks which have not been target of a branch yet
          if (res.second) {
            bbOrder.push_back(caseSuccessor);
          }
        }
      }

      // Check if control could take the default case
      defaultValue = optimizer.optimizeExpr(defaultValue, false);
      bool res;
      bool success = solver->mayBeTrue(state, defaultValue, res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        std::pair<std::map<BasicBlock *, ref<Expr> >::iterator, bool> ret =
            branchTargets.insert(
                std::make_pair(si->getDefaultDest(), defaultValue));
        if (ret.second) {
          bbOrder.push_back(si->getDefaultDest());
        }
      }

      // Fork the current state with each state having one of the possible
      // successors of this switch
      std::vector< ref<Expr> > conditions;
      for (std::vector<BasicBlock *>::iterator it = bbOrder.begin(),
                                               ie = bbOrder.end();
           it != ie; ++it) {
        conditions.push_back(branchTargets[*it]);
      }
      std::vector<ExecutionState*> branches;
      branch(state, conditions, branches);

      std::vector<ExecutionState*>::iterator bit = branches.begin();
      for (std::vector<BasicBlock *>::iterator it = bbOrder.begin(),
                                               ie = bbOrder.end();
           it != ie; ++it) {
        ExecutionState *es = *bit;
        if (es)
          transferToBasicBlock(*it, bb, *es);
        ++bit;
      }
    }
    break;
  }
  case Instruction::Unreachable:
    // Note that this is not necessarily an internal bug, llvm will
    // generate unreachable instructions in cases where it knows the
    // program will crash. So it is effectively a SEGV or internal
    // error.
    terminateStateOnExecError(state, "reached \"unreachable\" instruction");
    break;

  case Instruction::Invoke:
  case Instruction::Call: {
    /********************** Speculative Execution Modeling ***************/
    if (SpeculativeModeling && state.stateType == ExecutionState::SPECULATIVE) {
#ifdef Lewis_DEBUG_SPECU
      fprintf(stderr, "[+] Invoke/Call in speculative state, terminate\n");
#endif
      stopSpeculativeExecution(state);
      terminateStateEarly(state, "Speculative execution reaches threshold.");
      return;
    }
    /****************** End Speculative Execution Modeling ***************/
    // Ignore debug intrinsic calls
    if (isa<DbgInfoIntrinsic>(i))
      break;
    CallSite cs(i);

    unsigned numArgs = cs.arg_size();
    Value *fp = cs.getCalledValue();
    Function *f = getTargetFunction(fp, state);

    // Skip debug intrinsics, we can't evaluate their metadata arguments.
    if (isa<DbgInfoIntrinsic>(i))
      break;

    if (isa<InlineAsm>(fp)) {
      terminateStateOnExecError(state, "inline assembly is unsupported");
      break;
    }
    // evaluate arguments
    std::vector< ref<Expr> > arguments;
    arguments.reserve(numArgs);

    for (unsigned j=0; j<numArgs; ++j)
      arguments.push_back(eval(ki, j+1, state).value);

    if (f) {
      const FunctionType *fType = 
        dyn_cast<FunctionType>(cast<PointerType>(f->getType())->getElementType());
      const FunctionType *fpType =
        dyn_cast<FunctionType>(cast<PointerType>(fp->getType())->getElementType());

      // special case the call with a bitcast case
      if (fType != fpType) {
        assert(fType && fpType && "unable to get function type");

        // XXX check result coercion

        // XXX this really needs thought and validation
        unsigned i=0;
        for (std::vector< ref<Expr> >::iterator
               ai = arguments.begin(), ie = arguments.end();
             ai != ie; ++ai) {
          Expr::Width to, from = (*ai)->getWidth();
            
          if (i<fType->getNumParams()) {
            to = getWidthForLLVMType(fType->getParamType(i));

            if (from != to) {
              // XXX need to check other param attrs ?
#if LLVM_VERSION_CODE >= LLVM_VERSION(5, 0)
              bool isSExt = cs.paramHasAttr(i, llvm::Attribute::SExt);
#else
              bool isSExt = cs.paramHasAttr(i+1, llvm::Attribute::SExt);
#endif
              if (isSExt) {
                arguments[i] = SExtExpr::create(arguments[i], to);
              } else {
                arguments[i] = ZExtExpr::create(arguments[i], to);
              }
            }
          }
            
          i++;
        }
      }

      executeCall(state, ki, f, arguments);
      /********************** Cache Modeling **************************/
      if (f->getName().str() != "klee_make_symbolic" && 
          f->getName().str() != "klee_assume") {
        state.old_rsp.push_back(state.rsp);
        state.rsp -= 0x10;
      }
      /******************* End Cache Modeling *************************/
      
    } else {
      ref<Expr> v = eval(ki, 0, state).value;

      ExecutionState *free = &state;
      bool hasInvalid = false, first = true;

      /* XXX This is wasteful, no need to do a full evaluate since we
         have already got a value. But in the end the caches should
         handle it for us, albeit with some overhead. */
      do {
        v = optimizer.optimizeExpr(v, true);
        ref<ConstantExpr> value;
        bool success = solver->getValue(*free, v, value);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        StatePair res = fork(*free, EqExpr::create(v, value), true);
        if (res.first) {
          uint64_t addr = value->getZExtValue();
          if (legalFunctions.count(addr)) {
            f = (Function*) addr;

            // Don't give warning on unique resolution
            if (res.second || !first)
              klee_warning_once(reinterpret_cast<void*>(addr),
                                "resolved symbolic function pointer to: %s",
                                f->getName().data());

            executeCall(*res.first, ki, f, arguments);
          } else {
            if (!hasInvalid) {
              terminateStateOnExecError(state, "invalid function pointer");
              hasInvalid = true;
            }
          }
        }

        first = false;
        free = res.second;
      } while (free);
      /********************** Cache Modeling **************************/
      state.old_rsp.push_back(state.rsp);
      state.rsp -= 0x10;
      /******************* End Cache Modeling *************************/
    }
    break;
  }
  case Instruction::PHI: {
    ref<Expr> result = eval(ki, state.incomingBBIndex, state).value;
    bindLocal(ki, state, result);
    break;
  }

    // Special instructions
  case Instruction::Select: {
    // NOTE: It is not required that operands 1 and 2 be of scalar type.
    ref<Expr> cond = eval(ki, 0, state).value;
    ref<Expr> tExpr = eval(ki, 1, state).value;
    ref<Expr> fExpr = eval(ki, 2, state).value;
    ref<Expr> result = SelectExpr::create(cond, tExpr, fExpr);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::VAArg:
    terminateStateOnExecError(state, "unexpected VAArg instruction");
    break;

    // Arithmetic / logical

  case Instruction::Add: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    bindLocal(ki, state, AddExpr::create(left, right));
    break;
  }

  case Instruction::Sub: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    bindLocal(ki, state, SubExpr::create(left, right));
    break;
  }
 
  case Instruction::Mul: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    bindLocal(ki, state, MulExpr::create(left, right));
    break;
  }

  case Instruction::UDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = UDivExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::SDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = SDivExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::URem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = URemExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::SRem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = SRemExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::And: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AndExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Or: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = OrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Xor: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = XorExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Shl: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = ShlExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::LShr: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = LShrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::AShr: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AShrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

    // Compare

  case Instruction::ICmp: {
    CmpInst *ci = cast<CmpInst>(i);
    ICmpInst *ii = cast<ICmpInst>(ci);

    switch(ii->getPredicate()) {
    case ICmpInst::ICMP_EQ: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = EqExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_NE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = NeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_UGT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UgtExpr::create(left, right);
      bindLocal(ki, state,result);
      break;
    }

    case ICmpInst::ICMP_UGE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UgeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UltExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UleExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SgtExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SgeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SltExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SleExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    default:
      terminateStateOnExecError(state, "invalid ICmp predicate");
    }
    break;
  }
 
    // Memory instructions...
  case Instruction::Alloca: {
    AllocaInst *ai = cast<AllocaInst>(i);
    unsigned elementSize = 
      kmodule->targetData->getTypeStoreSize(ai->getAllocatedType());
    ref<Expr> size = Expr::createPointer(elementSize);
    if (ai->isArrayAllocation()) {
      ref<Expr> count = eval(ki, 0, state).value;
      count = Expr::createZExtToPointerWidth(count);
      size = MulExpr::create(size, count);
    }
    executeAlloc(state, size, true, ki);
    break;
  }

  case Instruction::Load: {
    ref<Expr> base = eval(ki, 0, state).value;
    
    if (SpeculativeModeling) {
      std::vector<unsigned>::iterator it = std::find(kleeMemAccessLine.begin(),
          kleeMemAccessLine.end(), ki->info->assemblyLine);

      // load caused by klee
      if (it != kleeMemAccessLine.end()) {
        executeMemoryOperation(state, false, base, 0, ki);
#ifdef Lewis_DEBUG_CACHE
        fprintf(stderr, "[+] load caused by klee instrumentation, skip cache analysis\n");
#endif
      } else {
        executeMemoryOperation(state, false, base, 0, ki, true);
        
        // on-the-fly analysis entry
        /*
        if (state.stateType == ExecutionState::SYMBOLIC && CacheModeling && !state.RegObj)
          analyzeMemCache(state, solver);
          */
      }
    } else {
      executeMemoryOperation(state, false, base, 0, ki);
    }
    break;
  }
  case Instruction::Store: {
    ref<Expr> base = eval(ki, 1, state).value;
    ref<Expr> value = eval(ki, 0, state).value;

    if (SpeculativeModeling) {
      std::vector<unsigned>::iterator it = std::find(kleeMemAccessLine.begin(),
          kleeMemAccessLine.end(), ki->info->assemblyLine);

      // store caused by klee
      if (it != Executor::kleeMemAccessLine.end()) {
        executeMemoryOperation(state, true, base, value, 0);
#ifdef Lewis_DEBUG_CACHE
        fprintf(stderr, "[+] store caused by klee instrumentation, skip cache analysis\n");
#endif
      } else {
        executeMemoryOperation(state, true, base, value, 0, true);

        // on-the-fly analysis entry
        /*
        if (state.stateType == ExecutionState::SYMBOLIC && CacheModeling && !state.RegObj)
          analyzeMemCache(state, solver);
          */
      }
    } else {
      executeMemoryOperation(state, true, base, value, 0);
    }
    break;
  }

  case Instruction::GetElementPtr: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);
    ref<Expr> base = eval(ki, 0, state).value;

    for (std::vector< std::pair<unsigned, uint64_t> >::iterator 
           it = kgepi->indices.begin(), ie = kgepi->indices.end(); 
         it != ie; ++it) {
      uint64_t elementSize = it->second;
      ref<Expr> index = eval(ki, it->first, state).value;
      base = AddExpr::create(base,
                             MulExpr::create(Expr::createSExtToPointerWidth(index),
                                             Expr::createPointer(elementSize)));
    }
    if (kgepi->offset)
      base = AddExpr::create(base,
                             Expr::createPointer(kgepi->offset));
    bindLocal(ki, state, base);
    break;
  }

    // Conversion
  case Instruction::Trunc: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = ExtractExpr::create(eval(ki, 0, state).value,
                                           0,
                                           getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ZExt: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = ZExtExpr::create(eval(ki, 0, state).value,
                                        getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::SExt: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = SExtExpr::create(eval(ki, 0, state).value,
                                        getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::IntToPtr: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width pType = getWidthForLLVMType(ci->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    bindLocal(ki, state, ZExtExpr::create(arg, pType));
    break;
  }
  case Instruction::PtrToInt: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width iType = getWidthForLLVMType(ci->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    bindLocal(ki, state, ZExtExpr::create(arg, iType));
    break;
  }

  case Instruction::BitCast: {
    ref<Expr> result = eval(ki, 0, state).value;
    bindLocal(ki, state, result);
    break;
  }

    // Floating point instructions

  case Instruction::FAdd: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FAdd operation");

    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.add(APFloat(*fpWidthToSemantics(right->getWidth()),right->getAPValue()), APFloat::rmNearestTiesToEven);
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FSub: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FSub operation");
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.subtract(APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()), APFloat::rmNearestTiesToEven);
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FMul: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FMul operation");

    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.multiply(APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()), APFloat::rmNearestTiesToEven);
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FDiv: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FDiv operation");

    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.divide(APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()), APFloat::rmNearestTiesToEven);
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FRem: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FRem operation");
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.mod(
        APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()));
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FPTrunc: {
    FPTruncInst *fi = cast<FPTruncInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > arg->getWidth())
      return terminateStateOnExecError(state, "Unsupported FPTrunc operation");

    llvm::APFloat Res(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
    bool losesInfo = false;
    Res.convert(*fpWidthToSemantics(resultType),
                llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
    bindLocal(ki, state, ConstantExpr::alloc(Res));
    break;
  }

  case Instruction::FPExt: {
    FPExtInst *fi = cast<FPExtInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || arg->getWidth() > resultType)
      return terminateStateOnExecError(state, "Unsupported FPExt operation");
    llvm::APFloat Res(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
    bool losesInfo = false;
    Res.convert(*fpWidthToSemantics(resultType),
                llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
    bindLocal(ki, state, ConstantExpr::alloc(Res));
    break;
  }

  case Instruction::FPToUI: {
    FPToUIInst *fi = cast<FPToUIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64)
      return terminateStateOnExecError(state, "Unsupported FPToUI operation");

    llvm::APFloat Arg(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
    uint64_t value = 0;
    bool isExact = true;
#if LLVM_VERSION_CODE >= LLVM_VERSION(5, 0)
    auto valueRef = makeMutableArrayRef(value);
#else
    uint64_t *valueRef = &value;
#endif
    Arg.convertToInteger(valueRef, resultType, false,
                         llvm::APFloat::rmTowardZero, &isExact);
    bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    break;
  }

  case Instruction::FPToSI: {
    FPToSIInst *fi = cast<FPToSIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64)
      return terminateStateOnExecError(state, "Unsupported FPToSI operation");
    llvm::APFloat Arg(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());

    uint64_t value = 0;
    bool isExact = true;
#if LLVM_VERSION_CODE >= LLVM_VERSION(5, 0)
    auto valueRef = makeMutableArrayRef(value);
#else
    uint64_t *valueRef = &value;
#endif
    Arg.convertToInteger(valueRef, resultType, true,
                         llvm::APFloat::rmTowardZero, &isExact);
    bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    break;
  }

  case Instruction::UIToFP: {
    UIToFPInst *fi = cast<UIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics)
      return terminateStateOnExecError(state, "Unsupported UIToFP operation");
    llvm::APFloat f(*semantics, 0);
    f.convertFromAPInt(arg->getAPValue(), false,
                       llvm::APFloat::rmNearestTiesToEven);

    bindLocal(ki, state, ConstantExpr::alloc(f));
    break;
  }

  case Instruction::SIToFP: {
    SIToFPInst *fi = cast<SIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics)
      return terminateStateOnExecError(state, "Unsupported SIToFP operation");
    llvm::APFloat f(*semantics, 0);
    f.convertFromAPInt(arg->getAPValue(), true,
                       llvm::APFloat::rmNearestTiesToEven);

    bindLocal(ki, state, ConstantExpr::alloc(f));
    break;
  }

  case Instruction::FCmp: {
    FCmpInst *fi = cast<FCmpInst>(i);
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FCmp operation");

    APFloat LHS(*fpWidthToSemantics(left->getWidth()),left->getAPValue());
    APFloat RHS(*fpWidthToSemantics(right->getWidth()),right->getAPValue());
    APFloat::cmpResult CmpRes = LHS.compare(RHS);

    bool Result = false;
    switch( fi->getPredicate() ) {
      // Predicates which only care about whether or not the operands are NaNs.
    case FCmpInst::FCMP_ORD:
      Result = (CmpRes != APFloat::cmpUnordered);
      break;

    case FCmpInst::FCMP_UNO:
      Result = (CmpRes == APFloat::cmpUnordered);
      break;

      // Ordered comparisons return false if either operand is NaN.  Unordered
      // comparisons return true if either operand is NaN.
    case FCmpInst::FCMP_UEQ:
      Result = (CmpRes == APFloat::cmpUnordered || CmpRes == APFloat::cmpEqual);
      break;
    case FCmpInst::FCMP_OEQ:
      Result = (CmpRes != APFloat::cmpUnordered && CmpRes == APFloat::cmpEqual);
      break;

    case FCmpInst::FCMP_UGT:
      Result = (CmpRes == APFloat::cmpUnordered || CmpRes == APFloat::cmpGreaterThan);
      break;
    case FCmpInst::FCMP_OGT:
      Result = (CmpRes != APFloat::cmpUnordered && CmpRes == APFloat::cmpGreaterThan);
      break;

    case FCmpInst::FCMP_UGE:
      Result = (CmpRes == APFloat::cmpUnordered || (CmpRes == APFloat::cmpGreaterThan || CmpRes == APFloat::cmpEqual));
      break;
    case FCmpInst::FCMP_OGE:
      Result = (CmpRes != APFloat::cmpUnordered && (CmpRes == APFloat::cmpGreaterThan || CmpRes == APFloat::cmpEqual));
      break;

    case FCmpInst::FCMP_ULT:
      Result = (CmpRes == APFloat::cmpUnordered || CmpRes == APFloat::cmpLessThan);
      break;
    case FCmpInst::FCMP_OLT:
      Result = (CmpRes != APFloat::cmpUnordered && CmpRes == APFloat::cmpLessThan);
      break;

    case FCmpInst::FCMP_ULE:
      Result = (CmpRes == APFloat::cmpUnordered || (CmpRes == APFloat::cmpLessThan || CmpRes == APFloat::cmpEqual));
      break;
    case FCmpInst::FCMP_OLE:
      Result = (CmpRes != APFloat::cmpUnordered && (CmpRes == APFloat::cmpLessThan || CmpRes == APFloat::cmpEqual));
      break;

    case FCmpInst::FCMP_UNE:
      Result = (CmpRes == APFloat::cmpUnordered || CmpRes != APFloat::cmpEqual);
      break;
    case FCmpInst::FCMP_ONE:
      Result = (CmpRes != APFloat::cmpUnordered && CmpRes != APFloat::cmpEqual);
      break;

    default:
      assert(0 && "Invalid FCMP predicate!");
      break;
    case FCmpInst::FCMP_FALSE:
      Result = false;
      break;
    case FCmpInst::FCMP_TRUE:
      Result = true;
      break;
    }

    bindLocal(ki, state, ConstantExpr::alloc(Result, Expr::Bool));
    break;
  }
  case Instruction::InsertValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    ref<Expr> agg = eval(ki, 0, state).value;
    ref<Expr> val = eval(ki, 1, state).value;

    ref<Expr> l = NULL, r = NULL;
    unsigned lOffset = kgepi->offset*8, rOffset = kgepi->offset*8 + val->getWidth();

    if (lOffset > 0)
      l = ExtractExpr::create(agg, 0, lOffset);
    if (rOffset < agg->getWidth())
      r = ExtractExpr::create(agg, rOffset, agg->getWidth() - rOffset);

    ref<Expr> result;
    if (!l.isNull() && !r.isNull())
      result = ConcatExpr::create(r, ConcatExpr::create(val, l));
    else if (!l.isNull())
      result = ConcatExpr::create(val, l);
    else if (!r.isNull())
      result = ConcatExpr::create(r, val);
    else
      result = val;

    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ExtractValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    ref<Expr> agg = eval(ki, 0, state).value;

    ref<Expr> result = ExtractExpr::create(agg, kgepi->offset*8, getWidthForLLVMType(i->getType()));

    bindLocal(ki, state, result);
    break;
  }
  case Instruction::Fence: {
    // Ignore for now
    break;
  }
  case Instruction::InsertElement: {
    InsertElementInst *iei = cast<InsertElementInst>(i);
    ref<Expr> vec = eval(ki, 0, state).value;
    ref<Expr> newElt = eval(ki, 1, state).value;
    ref<Expr> idx = eval(ki, 2, state).value;

    ConstantExpr *cIdx = dyn_cast<ConstantExpr>(idx);
    if (cIdx == NULL) {
      terminateStateOnError(
          state, "InsertElement, support for symbolic index not implemented",
          Unhandled);
      return;
    }
    uint64_t iIdx = cIdx->getZExtValue();
    const llvm::VectorType *vt = iei->getType();
    unsigned EltBits = getWidthForLLVMType(vt->getElementType());

    if (iIdx >= vt->getNumElements()) {
      // Out of bounds write
      terminateStateOnError(state, "Out of bounds write when inserting element",
                            BadVectorAccess);
      return;
    }

    const unsigned elementCount = vt->getNumElements();
    llvm::SmallVector<ref<Expr>, 8> elems;
    elems.reserve(elementCount);
    for (unsigned i = elementCount; i != 0; --i) {
      auto of = i - 1;
      unsigned bitOffset = EltBits * of;
      elems.push_back(
          of == iIdx ? newElt : ExtractExpr::create(vec, bitOffset, EltBits));
    }

    assert(Context::get().isLittleEndian() && "FIXME:Broken for big endian");
    ref<Expr> Result = ConcatExpr::createN(elementCount, elems.data());
    bindLocal(ki, state, Result);
    break;
  }
  case Instruction::ExtractElement: {
    ExtractElementInst *eei = cast<ExtractElementInst>(i);
    ref<Expr> vec = eval(ki, 0, state).value;
    ref<Expr> idx = eval(ki, 1, state).value;

    ConstantExpr *cIdx = dyn_cast<ConstantExpr>(idx);
    if (cIdx == NULL) {
      terminateStateOnError(
          state, "ExtractElement, support for symbolic index not implemented",
          Unhandled);
      return;
    }
    uint64_t iIdx = cIdx->getZExtValue();
    const llvm::VectorType *vt = eei->getVectorOperandType();
    unsigned EltBits = getWidthForLLVMType(vt->getElementType());

    if (iIdx >= vt->getNumElements()) {
      // Out of bounds read
      terminateStateOnError(state, "Out of bounds read when extracting element",
                            BadVectorAccess);
      return;
    }

    unsigned bitOffset = EltBits * iIdx;
    ref<Expr> Result = ExtractExpr::create(vec, bitOffset, EltBits);
    bindLocal(ki, state, Result);
    break;
  }
  case Instruction::ShuffleVector:
    // Should never happen due to Scalarizer pass removing ShuffleVector
    // instructions.
    terminateStateOnExecError(state, "Unexpected ShuffleVector instruction");
    break;
  case Instruction::AtomicRMW:
    terminateStateOnExecError(state, "Unexpected Atomic instruction, should be "
                                     "lowered by LowerAtomicInstructionPass");
    break;
  case Instruction::AtomicCmpXchg:
    terminateStateOnExecError(state,
                              "Unexpected AtomicCmpXchg instruction, should be "
                              "lowered by LowerAtomicInstructionPass");
    break;
  // Other instructions...
  // Unhandled
  default:
    terminateStateOnExecError(state, "illegal instruction");
    break;
  }
}

void Executor::updateStates(ExecutionState *current) {
  if (searcher) {
    searcher->update(current, addedStates, removedStates);
  }
  
  states.insert(addedStates.begin(), addedStates.end());
  addedStates.clear();

  for (std::vector<ExecutionState *>::iterator it = removedStates.begin(),
                                               ie = removedStates.end();
       it != ie; ++it) {
    ExecutionState *es = *it;
    std::set<ExecutionState*>::iterator it2 = states.find(es);
    assert(it2!=states.end());
    states.erase(it2);
    std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(es);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    processTree->remove(es->ptreeNode);
    delete es;
  }
  removedStates.clear();

  if (searcher) {
    searcher->update(nullptr, continuedStates, pausedStates);
    pausedStates.clear();
    continuedStates.clear();
  }
}

template <typename TypeIt>
void Executor::computeOffsets(KGEPInstruction *kgepi, TypeIt ib, TypeIt ie) {
  ref<ConstantExpr> constantOffset =
    ConstantExpr::alloc(0, Context::get().getPointerWidth());
  uint64_t index = 1;
  for (TypeIt ii = ib; ii != ie; ++ii) {
    if (StructType *st = dyn_cast<StructType>(*ii)) {
      const StructLayout *sl = kmodule->targetData->getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(ii.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned) ci->getZExtValue());
      constantOffset = constantOffset->Add(ConstantExpr::alloc(addend,
                                                               Context::get().getPointerWidth()));
    } else if (const auto set = dyn_cast<SequentialType>(*ii)) {
      uint64_t elementSize = 
        kmodule->targetData->getTypeStoreSize(set->getElementType());
      Value *operand = ii.getOperand();
      if (Constant *c = dyn_cast<Constant>(operand)) {
        ref<ConstantExpr> index = 
          evalConstant(c)->SExt(Context::get().getPointerWidth());
        ref<ConstantExpr> addend = 
          index->Mul(ConstantExpr::alloc(elementSize,
                                         Context::get().getPointerWidth()));
        constantOffset = constantOffset->Add(addend);
      } else {
        kgepi->indices.push_back(std::make_pair(index, elementSize));
      }
#if LLVM_VERSION_CODE >= LLVM_VERSION(4, 0)
    } else if (const auto ptr = dyn_cast<PointerType>(*ii)) {
      auto elementSize =
        kmodule->targetData->getTypeStoreSize(ptr->getElementType());
      auto operand = ii.getOperand();
      if (auto c = dyn_cast<Constant>(operand)) {
        auto index = evalConstant(c)->SExt(Context::get().getPointerWidth());
        auto addend = index->Mul(ConstantExpr::alloc(elementSize,
                                         Context::get().getPointerWidth()));
        constantOffset = constantOffset->Add(addend);
      } else {
        kgepi->indices.push_back(std::make_pair(index, elementSize));
      }
#endif
    } else
      assert("invalid type" && 0);
    index++;
  }
  kgepi->offset = constantOffset->getZExtValue();
}

void Executor::bindInstructionConstants(KInstruction *KI) {
  KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(KI);

  if (GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(KI->inst)) {
    computeOffsets(kgepi, gep_type_begin(gepi), gep_type_end(gepi));
  } else if (InsertValueInst *ivi = dyn_cast<InsertValueInst>(KI->inst)) {
    computeOffsets(kgepi, iv_type_begin(ivi), iv_type_end(ivi));
    assert(kgepi->indices.empty() && "InsertValue constant offset expected");
  } else if (ExtractValueInst *evi = dyn_cast<ExtractValueInst>(KI->inst)) {
    computeOffsets(kgepi, ev_type_begin(evi), ev_type_end(evi));
    assert(kgepi->indices.empty() && "ExtractValue constant offset expected");
  }
}

void Executor::bindModuleConstants() {
  for (auto &kfp : kmodule->functions) {
    KFunction *kf = kfp.get();
    for (unsigned i=0; i<kf->numInstructions; ++i)
      bindInstructionConstants(kf->instructions[i]);
  }

  kmodule->constantTable =
      std::unique_ptr<Cell[]>(new Cell[kmodule->constants.size()]);
  for (unsigned i=0; i<kmodule->constants.size(); ++i) {
    Cell &c = kmodule->constantTable[i];
    c.value = evalConstant(kmodule->constants[i]);
  }
}

void Executor::checkMemoryUsage() {
  if (!MaxMemory)
    return;
  if ((stats::instructions & 0xFFFF) == 0) {
    // We need to avoid calling GetTotalMallocUsage() often because it
    // is O(elts on freelist). This is really bad since we start
    // to pummel the freelist once we hit the memory cap.
    unsigned mbs = (util::GetTotalMallocUsage() >> 20) +
                   (memory->getUsedDeterministicSize() >> 20);

    if (mbs > MaxMemory) {
      if (mbs > MaxMemory + 100) {
        // just guess at how many to kill
        unsigned numStates = states.size();
        unsigned toKill = std::max(1U, numStates - numStates * MaxMemory / mbs);
        klee_warning("killing %d states (over memory cap)", toKill);
        std::vector<ExecutionState *> arr(states.begin(), states.end());
        for (unsigned i = 0, N = arr.size(); N && i < toKill; ++i, --N) {
          unsigned idx = rand() % N;
          // Make two pulls to try and not hit a state that
          // covered new code.
          if (arr[idx]->coveredNew)
            idx = rand() % N;

          std::swap(arr[idx], arr[N - 1]);
          terminateStateEarly(*arr[N - 1], "Memory limit exceeded.");
        }
      }
      atMemoryLimit = true;
    } else {
      atMemoryLimit = false;
    }
  }
}

void Executor::doDumpStates() {
  if (!DumpStatesOnHalt || states.empty())
    return;

  klee_message("halting execution, dumping remaining states");
  for (const auto &state : states)
    terminateStateEarly(*state, "Execution halting.");
  updateStates(nullptr);
}

void Executor::run(ExecutionState &initialState) {
  bindModuleConstants();

  // Delay init till now so that ticks don't accrue during
  // optimization and such.
  initTimers();

  states.insert(&initialState);

  if (usingSeeds) {
    std::vector<SeedInfo> &v = seedMap[&initialState];
    
    for (std::vector<KTest*>::const_iterator it = usingSeeds->begin(), 
           ie = usingSeeds->end(); it != ie; ++it)
      v.push_back(SeedInfo(*it));

    int lastNumSeeds = usingSeeds->size()+10;
    time::Point lastTime, startTime = lastTime = time::getWallTime();
    ExecutionState *lastState = 0;
    while (!seedMap.empty()) {
      if (haltExecution) {
        doDumpStates();
        return;
      }

      std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it = 
        seedMap.upper_bound(lastState);
      if (it == seedMap.end())
        it = seedMap.begin();
      lastState = it->first;
      unsigned numSeeds = it->second.size();
      ExecutionState &state = *lastState;
      KInstruction *ki = state.pc;
      stepInstruction(state);

      executeInstruction(state, ki);
      processTimers(&state, maxInstructionTime * numSeeds);
      updateStates(&state);

      if ((stats::instructions % 1000) == 0) {
        int numSeeds = 0, numStates = 0;
        for (std::map<ExecutionState*, std::vector<SeedInfo> >::iterator
               it = seedMap.begin(), ie = seedMap.end();
             it != ie; ++it) {
          numSeeds += it->second.size();
          numStates++;
        }
        const auto time = time::getWallTime();
        const time::Span seedTime(SeedTime);
        if (seedTime && time > startTime + seedTime) {
          klee_warning("seed time expired, %d seeds remain over %d states",
                       numSeeds, numStates);
          break;
        } else if (numSeeds<=lastNumSeeds-10 ||
                   time - lastTime >= time::seconds(10)) {
          lastTime = time;
          lastNumSeeds = numSeeds;          
          klee_message("%d seeds remaining over: %d states", 
                       numSeeds, numStates);
        }
      }
    }

    klee_message("seeding done (%d states remain)", (int) states.size());

    // XXX total hack, just because I like non uniform better but want
    // seed results to be equally weighted.
    for (std::set<ExecutionState*>::iterator
           it = states.begin(), ie = states.end();
         it != ie; ++it) {
      (*it)->weight = 1.;
    }

    if (OnlySeed) {
      doDumpStates();
      return;
    }
  }

  searcher = constructUserSearcher(*this);

  std::vector<ExecutionState *> newStates(states.begin(), states.end());
  searcher->update(0, newStates, std::vector<ExecutionState *>());

  while (!states.empty() && !haltExecution) {
    ExecutionState &state = searcher->selectState();
    KInstruction *ki = state.pc;
    
/********************** Speculative Execution Modeling ***************/

    // Instruction* i = ki->inst;
#ifdef Lewis_DEBUG_SPECU
    fprintf(stderr, "\n[+] Executing line %u/%u ", ki->info->assemblyLine, ki->info->line);
    if (state.stateType == ExecutionState::SPECULATIVE)
      fprintf(stderr, "in speculative state %lu rsp: 0x%lx\n", state.id, state.rsp);
    else if (state.stateType == ExecutionState::SYMBOLIC)
      fprintf(stderr, "in symbolic state %lu rsp: 0x%lx\n", state.id, state.rsp);
    else {
      assert(state.stateType == ExecutionState::UNKNOWN);
      fprintf(stderr, "in unknown state\n");
    }
    ki->inst->print(errs());
    fprintf(stderr, "\n");
#endif

    if(!state.continueFlag) {
      // assert(state.stateType == ExecutionState::SYMBOLIC);
      assert(state.nestedCnt < MaxSpeculativeDepth);
#ifdef Lewis_DEBUG_SPECU
      fprintf(stderr, "[+] Wait for speculative state %lu, skip\n", state.childId);
#endif
      continue;
    }

    if(state.stateType == ExecutionState::SPECULATIVE &&
        ++state.rbc >= MaxRBC) {
#ifdef Lewis_DEBUG_SPECU
      fprintf(stderr, "[+] speculative exec threshold reached (RBC), terminate\n");
#endif
      stopSpeculativeExecution(state);
      terminateStateEarly(state, "speculative exec threshold reached.");
      updateStates(&state);
      continue;
    }

/**************** End Speculative Execution Modeling ***************/

    stepInstruction(state);

    executeInstruction(state, ki);
    processTimers(&state, maxInstructionTime);

    checkMemoryUsage();

    updateStates(&state);
  }

  delete searcher;
  searcher = 0;

  doDumpStates();
}

std::string Executor::getAddressInfo(ExecutionState &state, 
                                     ref<Expr> address) const{
  std::string Str;
  llvm::raw_string_ostream info(Str);
  info << "\taddress: " << address << "\n";
  uint64_t example;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(address)) {
    example = CE->getZExtValue();
  } else {
    ref<ConstantExpr> value;
    bool success = solver->getValue(state, address, value);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    example = value->getZExtValue();
    info << "\texample: " << example << "\n";
    std::pair< ref<Expr>, ref<Expr> > res = solver->getRange(state, address);
    info << "\trange: [" << res.first << ", " << res.second <<"]\n";
  }
  
  MemoryObject hack((unsigned) example);    
  MemoryMap::iterator lower = state.addressSpace.objects.upper_bound(&hack);
  info << "\tnext: ";
  if (lower==state.addressSpace.objects.end()) {
    info << "none\n";
  } else {
    const MemoryObject *mo = lower->first;
    std::string alloc_info;
    mo->getAllocInfo(alloc_info);
    info << "object at " << mo->address
         << " of size " << mo->size << "\n"
         << "\t\t" << alloc_info << "\n";
  }
  if (lower!=state.addressSpace.objects.begin()) {
    --lower;
    info << "\tprev: ";
    if (lower==state.addressSpace.objects.end()) {
      info << "none\n";
    } else {
      const MemoryObject *mo = lower->first;
      std::string alloc_info;
      mo->getAllocInfo(alloc_info);
      info << "object at " << mo->address 
           << " of size " << mo->size << "\n"
           << "\t\t" << alloc_info << "\n";
    }
  }

  return info.str();
}

void Executor::pauseState(ExecutionState &state){
  auto it = std::find(continuedStates.begin(), continuedStates.end(), &state);
  // If the state was to be continued, but now gets paused again
  if (it != continuedStates.end()){
    // ...just don't continue it
    std::swap(*it, continuedStates.back());
    continuedStates.pop_back();
  } else {
    pausedStates.push_back(&state);
  }
}

void Executor::continueState(ExecutionState &state){
  auto it = std::find(pausedStates.begin(), pausedStates.end(), &state);
  // If the state was to be paused, but now gets continued again
  if (it != pausedStates.end()){
    // ...don't pause it
    std::swap(*it, pausedStates.back());
    pausedStates.pop_back();
  } else {
    continuedStates.push_back(&state);
  }
}


void Executor::terminateState(ExecutionState &state) {
  if (replayKTest && replayPosition!=replayKTest->numObjects) {
    klee_warning_once(replayKTest,
                      "replay did not consume all objects in test input.");
  }

/********************** Speculative Execution Modeling ***************/
  // offline-analysis entry
  if (state.stateType == ExecutionState::SYMBOLIC) {
    fprintf(stderr, "[+] Terminate symbolic state %lu\n", state.id);
    analyzeMemCache(state, solver);
  }

  if (state.stateType == ExecutionState::SPECULATIVE) {
    fprintf(stderr, "[+] Terminate speculative state %lu\n", state.id);
    // analyzeMemCache(state, solver);
  }
/***************** End Speculative Execution Modeling ***************/

  interpreterHandler->incPathsExplored();

  std::vector<ExecutionState *>::iterator it =
      std::find(addedStates.begin(), addedStates.end(), &state);
  if (it==addedStates.end()) {
    state.pc = state.prevPC;

    removedStates.push_back(&state);
  } else {
    // never reached searcher, just delete immediately
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(&state);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    addedStates.erase(it);
    processTree->remove(state.ptreeNode);
    delete &state;
  }
}

void Executor::terminateStateEarly(ExecutionState &state, 
                                   const Twine &message) {
  if (!OnlyOutputStatesCoveringNew || state.coveredNew ||
      (AlwaysOutputSeeds && seedMap.count(&state)))
    interpreterHandler->processTestCase(state, (message + "\n").str().c_str(),
                                        "early");
  terminateState(state);
}

void Executor::terminateStateOnExit(ExecutionState &state) {
  if (!OnlyOutputStatesCoveringNew || state.coveredNew || 
      (AlwaysOutputSeeds && seedMap.count(&state)))
    interpreterHandler->processTestCase(state, 0, 0);
  terminateState(state);
}

const InstructionInfo & Executor::getLastNonKleeInternalInstruction(const ExecutionState &state,
    Instruction ** lastInstruction) {
  // unroll the stack of the applications state and find
  // the last instruction which is not inside a KLEE internal function
  ExecutionState::stack_ty::const_reverse_iterator it = state.stack.rbegin(),
      itE = state.stack.rend();

  // don't check beyond the outermost function (i.e. main())
  itE--;

  const InstructionInfo * ii = 0;
  if (kmodule->internalFunctions.count(it->kf->function) == 0){
    ii =  state.prevPC->info;
    *lastInstruction = state.prevPC->inst;
    //  Cannot return yet because even though
    //  it->function is not an internal function it might of
    //  been called from an internal function.
  }

  // Wind up the stack and check if we are in a KLEE internal function.
  // We visit the entire stack because we want to return a CallInstruction
  // that was not reached via any KLEE internal functions.
  for (;it != itE; ++it) {
    // check calling instruction and if it is contained in a KLEE internal function
    const Function * f = (*it->caller).inst->getParent()->getParent();
    if (kmodule->internalFunctions.count(f)){
      ii = 0;
      continue;
    }
    if (!ii){
      ii = (*it->caller).info;
      *lastInstruction = (*it->caller).inst;
    }
  }

  if (!ii) {
    // something went wrong, play safe and return the current instruction info
    *lastInstruction = state.prevPC->inst;
    return *state.prevPC->info;
  }
  return *ii;
}

bool Executor::shouldExitOn(enum TerminateReason termReason) {
  std::vector<TerminateReason>::iterator s = ExitOnErrorType.begin();
  std::vector<TerminateReason>::iterator e = ExitOnErrorType.end();

  for (; s != e; ++s)
    if (termReason == *s)
      return true;

  return false;
}

void Executor::terminateStateOnError(ExecutionState &state,
                                     const llvm::Twine &messaget,
                                     enum TerminateReason termReason,
                                     const char *suffix,
                                     const llvm::Twine &info) {
  std::string message = messaget.str();
  static std::set< std::pair<Instruction*, std::string> > emittedErrors;
  Instruction * lastInst;
  const InstructionInfo &ii = getLastNonKleeInternalInstruction(state, &lastInst);
  
  if (EmitAllErrors ||
      emittedErrors.insert(std::make_pair(lastInst, message)).second) {
    if (ii.file != "") {
      klee_message("ERROR: %s:%d: %s", ii.file.c_str(), ii.line, message.c_str());
    } else {
      klee_message("ERROR: (location information missing) %s", message.c_str());
    }
    if (!EmitAllErrors)
      klee_message("NOTE: now ignoring this error at this location");

    std::string MsgString;
    llvm::raw_string_ostream msg(MsgString);
    msg << "Error: " << message << "\n";
    if (ii.file != "") {
      msg << "File: " << ii.file << "\n";
      msg << "Line: " << ii.line << "\n";
      msg << "assembly.ll line: " << ii.assemblyLine << "\n";
    }
    msg << "Stack: \n";
    state.dumpStack(msg);

    std::string info_str = info.str();
    if (info_str != "")
      msg << "Info: \n" << info_str;

    std::string suffix_buf;
    if (!suffix) {
      suffix_buf = TerminateReasonNames[termReason];
      suffix_buf += ".err";
      suffix = suffix_buf.c_str();
    }

    interpreterHandler->processTestCase(state, msg.str().c_str(), suffix);
  }
    
  terminateState(state);

  if (shouldExitOn(termReason))
    haltExecution = true;
}

// XXX shoot me
static const char *okExternalsList[] = { "printf", 
                                         "fprintf", 
                                         "puts",
                                         "getpid" };
static std::set<std::string> okExternals(okExternalsList,
                                         okExternalsList + 
                                         (sizeof(okExternalsList)/sizeof(okExternalsList[0])));

void Executor::callExternalFunction(ExecutionState &state,
                                    KInstruction *target,
                                    Function *function,
                                    std::vector< ref<Expr> > &arguments) {
  // check if specialFunctionHandler wants it
  if (specialFunctionHandler->handle(state, function, target, arguments))
    return;
  
  if (ExternalCalls == ExternalCallPolicy::None
      && !okExternals.count(function->getName())) {
    klee_warning("Disallowed call to external function: %s\n",
               function->getName().str().c_str());
    terminateStateOnError(state, "external calls disallowed", User);
    return;
  }

  // normal external function handling path
  // allocate 128 bits for each argument (+return value) to support fp80's;
  // we could iterate through all the arguments first and determine the exact
  // size we need, but this is faster, and the memory usage isn't significant.
  uint64_t *args = (uint64_t*) alloca(2*sizeof(*args) * (arguments.size() + 1));
  memset(args, 0, 2 * sizeof(*args) * (arguments.size() + 1));
  unsigned wordIndex = 2;
  for (std::vector<ref<Expr> >::iterator ai = arguments.begin(), 
       ae = arguments.end(); ai!=ae; ++ai) {
    if (ExternalCalls == ExternalCallPolicy::All) { // don't bother checking uniqueness
      *ai = optimizer.optimizeExpr(*ai, true);
      ref<ConstantExpr> ce;
      bool success = solver->getValue(state, *ai, ce);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      ce->toMemory(&args[wordIndex]);
      ObjectPair op;
      // Checking to see if the argument is a pointer to something
      if (ce->getWidth() == Context::get().getPointerWidth() &&
          state.addressSpace.resolveOne(ce, op)) {
        op.second->flushToConcreteStore(solver, state);
      }
      wordIndex += (ce->getWidth()+63)/64;
    } else {
      ref<Expr> arg = toUnique(state, *ai);
      if (ConstantExpr *ce = dyn_cast<ConstantExpr>(arg)) {
        // XXX kick toMemory functions from here
        ce->toMemory(&args[wordIndex]);
        wordIndex += (ce->getWidth()+63)/64;
      } else {
        terminateStateOnExecError(state, 
                                  "external call with symbolic argument: " + 
                                  function->getName());
        return;
      }
    }
  }

  // Prepare external memory for invoking the function
  state.addressSpace.copyOutConcretes();
#ifndef WINDOWS
  // Update external errno state with local state value
  int *errno_addr = getErrnoLocation(state);
  ObjectPair result;
  bool resolved = state.addressSpace.resolveOne(
      ConstantExpr::create((uint64_t)errno_addr, Expr::Int64), result);
  if (!resolved)
    klee_error("Could not resolve memory object for errno");
  ref<Expr> errValueExpr = result.second->read(0, sizeof(*errno_addr) * 8);
  ConstantExpr *errnoValue = dyn_cast<ConstantExpr>(errValueExpr);
  if (!errnoValue) {
    terminateStateOnExecError(state,
                              "external call with errno value symbolic: " +
                                  function->getName());
    return;
  }

  externalDispatcher->setLastErrno(
      errnoValue->getZExtValue(sizeof(*errno_addr) * 8));
#endif

  if (!SuppressExternalWarnings) {

    std::string TmpStr;
    llvm::raw_string_ostream os(TmpStr);
    os << "calling external: " << function->getName().str() << "(";
    for (unsigned i=0; i<arguments.size(); i++) {
      os << arguments[i];
      if (i != arguments.size()-1)
        os << ", ";
    }
    os << ") at " << state.pc->getSourceLocation();
    
    if (AllExternalWarnings)
      klee_warning("%s", os.str().c_str());
    else
      klee_warning_once(function, "%s", os.str().c_str());
  }

  bool success = externalDispatcher->executeCall(function, target->inst, args);
  if (!success) {
    terminateStateOnError(state, "failed external call: " + function->getName(),
                          External);
    return;
  }

  if (!state.addressSpace.copyInConcretes()) {
    terminateStateOnError(state, "external modified read-only object",
                          External);
    return;
  }

#ifndef WINDOWS
  // Update errno memory object with the errno value from the call
  int error = externalDispatcher->getLastErrno();
  state.addressSpace.copyInConcrete(result.first, result.second,
                                    (uint64_t)&error);
#endif

  Type *resultType = target->inst->getType();
  if (resultType != Type::getVoidTy(function->getContext())) {
    ref<Expr> e = ConstantExpr::fromMemory((void*) args, 
                                           getWidthForLLVMType(resultType));
    bindLocal(target, state, e);
  }
}

/***/

ref<Expr> Executor::replaceReadWithSymbolic(ExecutionState &state, 
                                            ref<Expr> e) {
  unsigned n = interpreterOpts.MakeConcreteSymbolic;
  if (!n || replayKTest || replayPath)
    return e;

  // right now, we don't replace symbolics (is there any reason to?)
  if (!isa<ConstantExpr>(e))
    return e;

  if (n != 1 && random() % n)
    return e;

  // create a new fresh location, assert it is equal to concrete value in e
  // and return it.
  
  static unsigned id;
  const Array *array =
      arrayCache.CreateArray("rrws_arr" + llvm::utostr(++id),
                             Expr::getMinBytesForWidth(e->getWidth()));
  ref<Expr> res = Expr::createTempRead(array, e->getWidth());
  ref<Expr> eq = NotOptimizedExpr::create(EqExpr::create(e, res));
  llvm::errs() << "Making symbolic: " << eq << "\n";
  state.addConstraint(eq);
  return res;
}

ObjectState *Executor::bindObjectInState(ExecutionState &state, 
                                         const MemoryObject *mo,
                                         bool isLocal,
                                         const Array *array) {
  ObjectState *os = array ? new ObjectState(mo, array) : new ObjectState(mo);
  state.addressSpace.bindObject(mo, os);

  // Its possible that multiple bindings of the same mo in the state
  // will put multiple copies on this list, but it doesn't really
  // matter because all we use this list for is to unbind the object
  // on function return.
  if (isLocal)
    state.stack.back().allocas.push_back(mo);

  return os;
}

void Executor::executeAlloc(ExecutionState &state,
                            ref<Expr> size,
                            bool isLocal,
                            KInstruction *target,
                            bool zeroMemory,
                            const ObjectState *reallocFrom,
                            size_t allocationAlignment) {
  size = toUnique(state, size);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(size)) {
    const llvm::Value *allocSite = state.prevPC->inst;
    if (allocationAlignment == 0) {
      allocationAlignment = getAllocationAlignment(allocSite);
    }
		/********************** Cache Modeling **************************/

    MemoryObject *mo = NULL;
    if (CacheModeling && isLocal) {
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "[+] Local alloc in size %lu\n", CE->getZExtValue());
#endif
      state.rsp -= CE->getZExtValue(); // little endian
      mo = memory->allocate(CE->getZExtValue(), isLocal, /*isGlobal=*/false,
          allocSite, allocationAlignment, state.rsp);
    } else {
      mo = memory->allocate(CE->getZExtValue(), isLocal, /*isGlobal=*/false,
          allocSite, allocationAlignment);
    }

    /********************** Cache Modeling **************************/
    if (!mo) {
      bindLocal(target, state, 
                ConstantExpr::alloc(0, Context::get().getPointerWidth()));
    } else {
      ObjectState *os = bindObjectInState(state, mo, isLocal);
      if (zeroMemory) {
        os->initializeToZero();
      } else {
        os->initializeToRandom();
      }
      bindLocal(target, state, mo->getBaseExpr());
      
      if (reallocFrom) {
        unsigned count = std::min(reallocFrom->size, os->size);
        for (unsigned i=0; i<count; i++)
          os->write(i, reallocFrom->read8(i));
        state.addressSpace.unbindObject(reallocFrom->getObject());
      }
    }
  } else {
    // XXX For now we just pick a size. Ideally we would support
    // symbolic sizes fully but even if we don't it would be better to
    // "smartly" pick a value, for example we could fork and pick the
    // min and max values and perhaps some intermediate (reasonable
    // value).
    // 
    // It would also be nice to recognize the case when size has
    // exactly two values and just fork (but we need to get rid of
    // return argument first). This shows up in pcre when llvm
    // collapses the size expression with a select.

    size = optimizer.optimizeExpr(size, true);

    ref<ConstantExpr> example;
    bool success = solver->getValue(state, size, example);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    
    // Try and start with a small example.
    Expr::Width W = example->getWidth();
    while (example->Ugt(ConstantExpr::alloc(128, W))->isTrue()) {
      ref<ConstantExpr> tmp = example->LShr(ConstantExpr::alloc(1, W));
      bool res;
      bool success = solver->mayBeTrue(state, EqExpr::create(tmp, size), res);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      if (!res)
        break;
      example = tmp;
    }

    StatePair fixedSize = fork(state, EqExpr::create(example, size), true);
    
    if (fixedSize.second) { 
      // Check for exactly two values
      ref<ConstantExpr> tmp;
      bool success = solver->getValue(*fixedSize.second, size, tmp);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      bool res;
      success = solver->mustBeTrue(*fixedSize.second, 
                                   EqExpr::create(tmp, size),
                                   res);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      if (res) {
        executeAlloc(*fixedSize.second, tmp, isLocal,
                     target, zeroMemory, reallocFrom);
      } else {
        // See if a *really* big value is possible. If so assume
        // malloc will fail for it, so lets fork and return 0.
        StatePair hugeSize = 
          fork(*fixedSize.second, 
               UltExpr::create(ConstantExpr::alloc(1U<<31, W), size),
               true);
        if (hugeSize.first) {
          klee_message("NOTE: found huge malloc, returning 0");
          bindLocal(target, *hugeSize.first, 
                    ConstantExpr::alloc(0, Context::get().getPointerWidth()));
        }
        
        if (hugeSize.second) {

          std::string Str;
          llvm::raw_string_ostream info(Str);
          ExprPPrinter::printOne(info, "  size expr", size);
          info << "  concretization : " << example << "\n";
          info << "  unbound example: " << tmp << "\n";
          terminateStateOnError(*hugeSize.second, "concretized symbolic size",
                                Model, NULL, info.str());
        }
      }
    }

    if (fixedSize.first) // can be zero when fork fails
      executeAlloc(*fixedSize.first, example, isLocal, 
                   target, zeroMemory, reallocFrom);
  }
}

void Executor::executeFree(ExecutionState &state,
                           ref<Expr> address,
                           KInstruction *target) {
  address = optimizer.optimizeExpr(address, true);
  StatePair zeroPointer = fork(state, Expr::createIsZero(address), true);
  if (zeroPointer.first) {
    if (target)
      bindLocal(target, *zeroPointer.first, Expr::createPointer(0));
  }
  if (zeroPointer.second) { // address != 0
    ExactResolutionList rl;
    resolveExact(*zeroPointer.second, address, rl, "free");
    
    for (Executor::ExactResolutionList::iterator it = rl.begin(), 
           ie = rl.end(); it != ie; ++it) {
      const MemoryObject *mo = it->first.first;
      if (mo->isLocal) {
        terminateStateOnError(*it->second, "free of alloca", Free, NULL,
                              getAddressInfo(*it->second, address));
      } else if (mo->isGlobal) {
        terminateStateOnError(*it->second, "free of global", Free, NULL,
                              getAddressInfo(*it->second, address));
      } else {
        it->second->addressSpace.unbindObject(mo);
        if (target)
          bindLocal(target, *it->second, Expr::createPointer(0));
      }
    }
  }
}

void Executor::resolveExact(ExecutionState &state,
                            ref<Expr> p,
                            ExactResolutionList &results, 
                            const std::string &name) {
  p = optimizer.optimizeExpr(p, true);
  // XXX we may want to be capping this?
  ResolutionList rl;
  state.addressSpace.resolve(state, solver, p, rl);
  
  ExecutionState *unbound = &state;
  for (ResolutionList::iterator it = rl.begin(), ie = rl.end(); 
       it != ie; ++it) {
    ref<Expr> inBounds = EqExpr::create(p, it->first->getBaseExpr());
    
    StatePair branches = fork(*unbound, inBounds, true);
    
    if (branches.first)
      results.push_back(std::make_pair(*it, branches.first));

    unbound = branches.second;
    if (!unbound) // Fork failure
      break;
  }

  if (unbound) {
    terminateStateOnError(*unbound, "memory error: invalid pointer: " + name,
                          Ptr, NULL, getAddressInfo(*unbound, p));
  }
}

void Executor::executeMemoryOperation(ExecutionState &state,
                                      bool isWrite,
                                      ref<Expr> address,
                                      ref<Expr> value /* undef if read */,
                                      KInstruction *target /* undef if write */, 
                                      bool recordObjName) {
  
  Expr::Width type = (isWrite ? value->getWidth() : 
                     getWidthForLLVMType(target->inst->getType()));
  unsigned bytes = Expr::getMinBytesForWidth(type);

  if (SimplifySymIndices) {
    if (!isa<ConstantExpr>(address))
      address = state.constraints.simplifyExpr(address);
    if (isWrite && !isa<ConstantExpr>(value))
      value = state.constraints.simplifyExpr(value);
  }

  address = optimizer.optimizeExpr(address, true);

  // fast path: single in-bounds resolution
  ObjectPair op;
  bool success;
  solver->setTimeout(coreSolverTimeout);
  if (!state.addressSpace.resolveOne(state, solver, address, op, success)) {
    address = toConstant(state, address, "resolveOne failure");
    success = state.addressSpace.resolveOne(cast<ConstantExpr>(address), op);
  }
  solver->setTimeout(time::Span());

  if (success) {
    const MemoryObject *mo = op.first;

    /********************** Cache Modeling **********************/
    // sjguo: record the object name for faster address comparion 
    // in cache analysis
    if (recordObjName) {
      assert(SpeculativeModeling && "SpeculativeModeling should be enabled.");
      if (mo->isRegObj) {
        state.regObj = true;
#ifdef Lewis_DEBUG_CACHE
        fprintf(stderr, "[+] memory object in register, skip logging and analysis\n");
#endif
      } else {
        state.objNames.push_back(mo->name);
        state.addrs.push_back(std::make_pair(address, true));
        state.pcInfos.push_back(state.pc->info);
        assert(state.addrs.size() == state.objNames.size() && 
            state.addrs.size() == state.pcInfos.size());
        state.regObj = false;
      }
    }
    /********************** End Cache Modeling ******************/

    if (MaxSymArraySize && mo->size >= MaxSymArraySize) {
      address = toConstant(state, address, "max-sym-array-size");
    }
    
    ref<Expr> offset = mo->getOffsetExpr(address);
    ref<Expr> check = mo->getBoundsCheckOffset(offset, bytes);
    check = optimizer.optimizeExpr(check, true);

    bool inBounds;
    solver->setTimeout(coreSolverTimeout);
    bool success = solver->mustBeTrue(state, check, inBounds);
    solver->setTimeout(time::Span());
    if (!success) {
      state.pc = state.prevPC;
      terminateStateEarly(state, "Query timed out (bounds check).");
      return;
    }

    if (inBounds) {
      const ObjectState *os = op.second;
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnError(state, "memory error: object read only",
                                ReadOnly);
        } else {
          ObjectState *wos = state.addressSpace.getWriteable(mo, os);
          wos->write(offset, value);
        }          
      } else {
        ref<Expr> result = os->read(offset, type);
        
        if (interpreterOpts.MakeConcreteSymbolic)
          result = replaceReadWithSymbolic(state, result);
        
        bindLocal(target, state, result);
      }

      return;
    }
  } 

  // we are on an error path (no resolution, multiple resolution, one
  // resolution with out of bounds)

  address = optimizer.optimizeExpr(address, true);
  ResolutionList rl;  
  solver->setTimeout(coreSolverTimeout);
  bool incomplete = state.addressSpace.resolve(state, solver, address, rl,
                                               0, coreSolverTimeout);
  solver->setTimeout(time::Span());
  
  // XXX there is some query wasteage here. who cares?
  ExecutionState *unbound = &state;
  
  for (ResolutionList::iterator i = rl.begin(), ie = rl.end(); i != ie; ++i) {
    const MemoryObject *mo = i->first;
    const ObjectState *os = i->second;
    ref<Expr> inBounds = mo->getBoundsCheckPointer(address, bytes);
    
    StatePair branches = fork(*unbound, inBounds, true);
    ExecutionState *bound = branches.first;

    // bound can be 0 on failure or overlapped 
    if (bound) {
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnError(*bound, "memory error: object read only",
                                ReadOnly);
        } else {
          ObjectState *wos = bound->addressSpace.getWriteable(mo, os);
          wos->write(mo->getOffsetExpr(address), value);
        }
      } else {
        ref<Expr> result = os->read(mo->getOffsetExpr(address), type);
        bindLocal(target, *bound, result);
      }
    }

    unbound = branches.second;
    if (!unbound)
      break;
  }
  
  // XXX should we distinguish out of bounds and overlapped cases?
  if (unbound) {
    if (incomplete) {
      terminateStateEarly(*unbound, "Query timed out (resolve).");
    } else {
      assert(0);
      terminateStateOnError(*unbound, "memory error: out of bound pointer", Ptr,
                            NULL, getAddressInfo(*unbound, address));
    }
  }
}

void Executor::executeMakeSymbolic(ExecutionState &state, 
                                   const MemoryObject *mo,
                                   const std::string &name) {
  // Create a new object state for the memory object (instead of a copy).
  if (!replayKTest) {
    // Find a unique name for this array.  First try the original name,
    // or if that fails try adding a unique identifier.
    unsigned id = 0;
    std::string uniqueName = name;
    while (!state.arrayNames.insert(uniqueName).second) {
      uniqueName = name + "_" + llvm::utostr(++id);
    }
    const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
    bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);
    
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
      seedMap.find(&state);
    if (it!=seedMap.end()) { // In seed mode we need to add this as a
                             // binding.
      for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
             siie = it->second.end(); siit != siie; ++siit) {
        SeedInfo &si = *siit;
        KTestObject *obj = si.getNextInput(mo, NamedSeedMatching);

        if (!obj) {
          if (ZeroSeedExtension) {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values = std::vector<unsigned char>(mo->size, '\0');
          } else if (!AllowSeedExtension) {
            terminateStateOnError(state, "ran out of inputs during seeding",
                                  User);
            break;
          }
        } else {
          if (obj->numBytes != mo->size &&
              ((!(AllowSeedExtension || ZeroSeedExtension)
                && obj->numBytes < mo->size) ||
               (!AllowSeedTruncation && obj->numBytes > mo->size))) {
	    std::stringstream msg;
	    msg << "replace size mismatch: "
		<< mo->name << "[" << mo->size << "]"
		<< " vs " << obj->name << "[" << obj->numBytes << "]"
		<< " in test\n";

            terminateStateOnError(state, msg.str(), User);
            break;
          } else {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values.insert(values.begin(), obj->bytes, 
                          obj->bytes + std::min(obj->numBytes, mo->size));
            if (ZeroSeedExtension) {
              for (unsigned i=obj->numBytes; i<mo->size; ++i)
                values.push_back('\0');
            }
          }
        }
      }
    }
  } else {
    ObjectState *os = bindObjectInState(state, mo, false);
    if (replayPosition >= replayKTest->numObjects) {
      terminateStateOnError(state, "replay count mismatch", User);
    } else {
      KTestObject *obj = &replayKTest->objects[replayPosition++];
      if (obj->numBytes != mo->size) {
        terminateStateOnError(state, "replay size mismatch", User);
      } else {
        for (unsigned i=0; i<mo->size; i++)
          os->write8(i, obj->bytes[i]);
      }
    }
  }
}

/***/

void Executor::runFunctionAsMain(Function *f,
				 int argc,
				 char **argv,
				 char **envp) {
  std::vector<ref<Expr> > arguments;

  // force deterministic initialization of memory objects
  srand(1);
  srandom(1);
  
  MemoryObject *argvMO = 0;

  // In order to make uclibc happy and be closer to what the system is
  // doing we lay out the environments at the end of the argv array
  // (both are terminated by a null). There is also a final terminating
  // null that uclibc seems to expect, possibly the ELF header?

  int envc;
  for (envc=0; envp[envc]; ++envc) ;

  unsigned NumPtrBytes = Context::get().getPointerWidth() / 8;
  KFunction *kf = kmodule->functionMap[f];
  assert(kf);
  Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
  if (ai!=ae) {
    arguments.push_back(ConstantExpr::alloc(argc, Expr::Int32));
    if (++ai!=ae) {
      Instruction *first = &*(f->begin()->begin());
			/********************** Cache Modeling **************************/

      if (CacheModeling) {
        fprintf(stderr, "[+] alloc arg/envc in size %u\n", (argc + 1 + envc + 1 + 1) * NumPtrBytes);
        argvMO = memory->allocate((argc + 1 + envc + 1 + 1) * NumPtrBytes,
            /*isLocal=*/false, /*isGlobal=*/true,
            /*allocSite=*/first, /*alignment=*/8,
            Executor::stackBase);
        Executor::stackBase -= (argc + 1 + envc + 1 + 1) * NumPtrBytes;
      } else {
        argvMO = memory->allocate((argc + 1 + envc + 1 + 1) * NumPtrBytes,
            /*isLocal=*/false, /*isGlobal=*/true,
            /*allocSite=*/first, /*alignment=*/8);
      }

      /******************** End Cache Modeling ************************/

      if (!argvMO)
        klee_error("Could not allocate memory for function arguments");

      arguments.push_back(argvMO->getBaseExpr());

      if (++ai!=ae) {
        uint64_t envp_start = argvMO->address + (argc+1)*NumPtrBytes;
        arguments.push_back(Expr::createPointer(envp_start));

        if (++ai!=ae)
          klee_error("invalid main function (expect 0-3 arguments)");
      }
    }
  }

  ExecutionState *state = new ExecutionState(kmodule->functionMap[f]);
  /********************** Cache Modeling **************************/
  state->rsp = Executor::stackBase;
  state->old_rsp.push_back(Executor::stackBase);
  /********************** End Modeling **************************/
  // Maintain a pointer of the executor in each ExecutionState
  state->exec = this;
  
  if (pathWriter) 
    state->pathOS = pathWriter->open();
  if (symPathWriter) 
    state->symPathOS = symPathWriter->open();


  if (statsTracker)
    statsTracker->framePushed(*state, 0);

  assert(arguments.size() == f->arg_size() && "wrong number of arguments");
  for (unsigned i = 0, e = f->arg_size(); i != e; ++i)
    bindArgument(kf, i, *state, arguments[i]);

  if (argvMO) {
    ObjectState *argvOS = bindObjectInState(*state, argvMO, false);

    for (int i=0; i<argc+1+envc+1+1; i++) {
      if (i==argc || i>=argc+1+envc) {
        // Write NULL pointer
        argvOS->write(i * NumPtrBytes, Expr::createPointer(0));
      } else {
        char *s = i<argc ? argv[i] : envp[i-(argc+1)];
        int j, len = strlen(s);

        MemoryObject *arg =
            memory->allocate(len + 1, /*isLocal=*/false, /*isGlobal=*/true,
                             /*allocSite=*/state->pc->inst, /*alignment=*/8);
        if (!arg)
          klee_error("Could not allocate memory for function arguments");
        ObjectState *os = bindObjectInState(*state, arg, false);
        for (j=0; j<len+1; j++)
          os->write8(j, s[j]);

        // Write pointer to newly allocated and initialised argv/envp c-string
        argvOS->write(i * NumPtrBytes, arg->getBaseExpr());
      }
    }
  }
  
  initializeGlobals(*state);

  processTree = new PTree(state);
  state->ptreeNode = processTree->root;
  run(*state);
  delete processTree;
  processTree = 0;

  // hack to clear memory objects
  delete memory;
  memory = new MemoryManager(NULL);

  globalObjects.clear();
  globalAddresses.clear();

  if (statsTracker)
    statsTracker->done();
}

unsigned Executor::getPathStreamID(const ExecutionState &state) {
  assert(pathWriter);
  return state.pathOS.getID();
}

unsigned Executor::getSymbolicPathStreamID(const ExecutionState &state) {
  assert(symPathWriter);
  return state.symPathOS.getID();
}

void Executor::getConstraintLog(const ExecutionState &state, std::string &res,
                                Interpreter::LogType logFormat) {

  switch (logFormat) {
  case STP: {
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool));
    char *log = solver->getConstraintLog(query);
    res = std::string(log);
    free(log);
  } break;

  case KQUERY: {
    std::string Str;
    llvm::raw_string_ostream info(Str);
    ExprPPrinter::printConstraints(info, state.constraints);
    res = info.str();
  } break;

  case SMTLIB2: {
    std::string Str;
    llvm::raw_string_ostream info(Str);
    ExprSMTLIBPrinter printer;
    printer.setOutput(info);
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool));
    printer.setQuery(query);
    printer.generateOutput();
    res = info.str();
  } break;

  default:
    klee_warning("Executor::getConstraintLog() : Log format not supported!");
  }
}

bool Executor::getSymbolicSolution(const ExecutionState &state,
                                   std::vector< 
                                   std::pair<std::string,
                                   std::vector<unsigned char> > >
                                   &res) {
  solver->setTimeout(coreSolverTimeout);

  ExecutionState tmp(state);

  // Go through each byte in every test case and attempt to restrict
  // it to the constraints contained in cexPreferences.  (Note:
  // usually this means trying to make it an ASCII character (0-127)
  // and therefore human readable. It is also possible to customize
  // the preferred constraints.  See test/Features/PreferCex.c for
  // an example) While this process can be very expensive, it can
  // also make understanding individual test cases much easier.
  for (unsigned i = 0; i != state.symbolics.size(); ++i) {
    const MemoryObject *mo = state.symbolics[i].first;
    std::vector< ref<Expr> >::const_iterator pi = 
      mo->cexPreferences.begin(), pie = mo->cexPreferences.end();
    for (; pi != pie; ++pi) {
      bool mustBeTrue;
      // Attempt to bound byte to constraints held in cexPreferences
      bool success = solver->mustBeTrue(tmp, Expr::createIsZero(*pi), 
					mustBeTrue);
      // If it isn't possible to constrain this particular byte in the desired
      // way (normally this would mean that the byte can't be constrained to
      // be between 0 and 127 without making the entire constraint list UNSAT)
      // then just continue on to the next byte.
      if (!success) break;
      // If the particular constraint operated on in this iteration through
      // the loop isn't implied then add it to the list of constraints.
      if (!mustBeTrue) tmp.addConstraint(*pi);
    }
    if (pi!=pie) break;
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);
  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(time::Span());
  if (!success) {
    klee_warning("unable to compute initial values (invalid constraints?)!");
    ExprPPrinter::printQuery(llvm::errs(), state.constraints,
                             ConstantExpr::alloc(0, Expr::Bool));
    return false;
  }
  
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    res.push_back(std::make_pair(state.symbolics[i].first->name, values[i]));
  return true;
}

void Executor::getCoveredLines(const ExecutionState &state,
                               std::map<const std::string*, std::set<unsigned> > &res) {
  res = state.coveredLines;
}

void Executor::doImpliedValueConcretization(ExecutionState &state,
                                            ref<Expr> e,
                                            ref<ConstantExpr> value) {
  abort(); // FIXME: Broken until we sort out how to do the write back.

  if (DebugCheckForImpliedValues)
    ImpliedValue::checkForImpliedValues(solver->solver, e, value);

  ImpliedValueList results;
  ImpliedValue::getImpliedValues(e, value, results);
  for (ImpliedValueList::iterator it = results.begin(), ie = results.end();
       it != ie; ++it) {
    ReadExpr *re = it->first.get();
    
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(re->index)) {
      // FIXME: This is the sole remaining usage of the Array object
      // variable. Kill me.
      const MemoryObject *mo = 0; //re->updates.root->object;
      const ObjectState *os = state.addressSpace.findObject(mo);

      if (!os) {
        // object has been free'd, no need to concretize (although as
        // in other cases we would like to concretize the outstanding
        // reads, but we have no facility for that yet)
      } else {
        assert(!os->readOnly && 
               "not possible? read only object with static read?");
        ObjectState *wos = state.addressSpace.getWriteable(mo, os);
        wos->write(CE, it->second);
      }
    }
  }
}

Expr::Width Executor::getWidthForLLVMType(llvm::Type *type) const {
  return kmodule->targetData->getTypeSizeInBits(type);
}

size_t Executor::getAllocationAlignment(const llvm::Value *allocSite) const {
  // FIXME: 8 was the previous default. We shouldn't hard code this
  // and should fetch the default from elsewhere.
  const size_t forcedAlignment = 8;
  size_t alignment = 0;
  llvm::Type *type = NULL;
  std::string allocationSiteName(allocSite->getName().str());
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(allocSite)) {
    alignment = GV->getAlignment();
    if (const GlobalVariable *globalVar = dyn_cast<GlobalVariable>(GV)) {
      // All GlobalVariables's have pointer type
      llvm::PointerType *ptrType =
          dyn_cast<llvm::PointerType>(globalVar->getType());
      assert(ptrType && "globalVar's type is not a pointer");
      type = ptrType->getElementType();
    } else {
      type = GV->getType();
    }
  } else if (const AllocaInst *AI = dyn_cast<AllocaInst>(allocSite)) {
    alignment = AI->getAlignment();
    type = AI->getAllocatedType();
  } else if (isa<InvokeInst>(allocSite) || isa<CallInst>(allocSite)) {
    // FIXME: Model the semantics of the call to use the right alignment
    llvm::Value *allocSiteNonConst = const_cast<llvm::Value *>(allocSite);
    const CallSite cs = (isa<InvokeInst>(allocSiteNonConst)
                             ? CallSite(cast<InvokeInst>(allocSiteNonConst))
                             : CallSite(cast<CallInst>(allocSiteNonConst)));
    llvm::Function *fn =
        klee::getDirectCallTarget(cs, /*moduleIsFullyLinked=*/true);
    if (fn)
      allocationSiteName = fn->getName().str();

    klee_warning_once(fn != NULL ? fn : allocSite,
                      "Alignment of memory from call \"%s\" is not "
                      "modelled. Using alignment of %zu.",
                      allocationSiteName.c_str(), forcedAlignment);
    alignment = forcedAlignment;
  } else {
    llvm_unreachable("Unhandled allocation site");
  }

  if (alignment == 0) {
    assert(type != NULL);
    // No specified alignment. Get the alignment for the type.
    if (type->isSized()) {
      alignment = kmodule->targetData->getPrefTypeAlignment(type);
    } else {
      klee_warning_once(allocSite, "Cannot determine memory alignment for "
                                   "\"%s\". Using alignment of %zu.",
                        allocationSiteName.c_str(), forcedAlignment);
      alignment = forcedAlignment;
    }
  }

  // Currently we require alignment be a power of 2
  if (!bits64::isPowerOfTwo(alignment)) {
    klee_warning_once(allocSite, "Alignment of %zu requested for %s but this "
                                 "not supported. Using alignment of %zu",
                      alignment, allocSite->getName().str().c_str(),
                      forcedAlignment);
    alignment = forcedAlignment;
  }
  assert(bits64::isPowerOfTwo(alignment) &&
         "Returned alignment must be a power of two");
  return alignment;
}

void Executor::prepareForEarlyExit() {
  if (statsTracker) {
    // Make sure stats get flushed out
    statsTracker->done();
  }
}

/// Returns the errno location in memory
int *Executor::getErrnoLocation(const ExecutionState &state) const {
#if !defined(__APPLE__) && !defined(__FreeBSD__)
  /* From /usr/include/errno.h: it [errno] is a per-thread variable. */
  return __errno_location();
#else
  return __error();
#endif
}

Interpreter *Interpreter::create(LLVMContext &ctx, const InterpreterOptions &opts,
                                 InterpreterHandler *ih) {
  return new Executor(ctx, opts, ih);
}

/******************************* Cache Modeling *********************************/

unsigned Executor::setNum = 0;
unsigned Executor::lineSize = 0;
unsigned Executor::nway = 0;
unsigned Executor::replace = 0;
uint64_t Executor::stackBase = 0;

void Executor::loadConfigFile(std::string path) {  
  cacheConfig.readFile(path.c_str());
  // we should handle exception here
  /*
  try {
    cacheConfig.readFile(path.c_str());
  } 
  catch (libconfig::ParseException& e) {
    llvm::errs() << path << " format is incorrect\n";
  } catch (libconfig::FileIOException& e) {
    llvm::errs() << path << " does not exists\n";
  }
  */

  if (cacheConfig.lookupValue("set", Executor::setNum) && 
      cacheConfig.lookupValue("line", Executor::lineSize) && 
      cacheConfig.lookupValue("asso", Executor::nway) && 
      cacheConfig.lookupValue("replace", Executor::replace)) {
    klee_message("Load cache config file successfully.");
    fprintf(stderr, "[+] set #:%u | line size:%u | asso:%u | replace:%u\n\n", 
        Executor::setNum, Executor::lineSize, Executor::nway, Executor::replace);
  } else {
    klee_error("Load mandatory cache config fields error, now exit.");
    exit(-1);
  }
}

void Executor::loadMemMapFile(std::string path) {
  
  FILE* fp = fopen(path.c_str(), "r");
  if (fp) {
    char line_str[256];
    fgets(line_str, sizeof(line_str), fp);
    sscanf(line_str, "stack:0x%lx", &Executor::stackBase);

    while(fgets(line_str, sizeof(line_str), fp)) {
      char name_str[32]; 
      uint64_t addr;
      if(!sscanf(line_str, "name:%s " "addr:0x%lx", name_str, &addr))
        break;
      std::string nstring(name_str);
      memoryMapFromFile[nstring] = addr;
    }
  }
  fclose(fp);
#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "###### Reading memory map from file ######\n");
  for (std::map<std::string, unsigned>::const_iterator it = memoryMapFromFile.begin(); 
     it != memoryMapFromFile.end(); it++) {
    std::string name = it->first;
    unsigned addr = it->second;
    fprintf(stderr, "[+] %s\t[0x%x]\n", name.c_str(), addr);
  }
  fprintf(stderr, "[+] Stack Base Address: \t[0x%lx]\n", Executor::stackBase);
  fprintf(stderr, "#### End reading memory map from file ####\n\n");
#endif
}

void Executor::loadKleeMemAccessFile(std::string path) {
  FILE* fp = fopen(path.c_str(), "r");
  if (fp) {
    char line_str[256];
    while(fgets(line_str, sizeof(line_str), fp)) {
      unsigned lineNum = 0;
      if(!sscanf(line_str, "%u", &lineNum))
        break;
      kleeMemAccessLine.push_back(lineNum);
    }
  }
  fclose(fp);
#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "###### Reading klee memory access from file ######\n");
  for (std::vector<unsigned>::const_iterator it = kleeMemAccessLine.begin(); it != kleeMemAccessLine.end(); it++)
    fprintf(stderr, "[+] line # %u\n", *it);
  fprintf(stderr, "#### End reading klee memory access from file ####\n\n");
#endif
}

int Executor::log_base2(int n){
  int power = 0;
  if (n<=0 || (n & (n-1)) != 0){
    assert(0 && "log2() only works for positive power of two values");
  }
  while(n >>= 1){
    power++;
  }
  return power;
}

ref<Expr> Executor::getSet(ref<Expr> addr){
  if(setCache.find(addr->hash()) != setCache.end()){
    return setCache[addr->hash()];
  }
  ref<Expr> set;
  if(addr->getKind() == Expr::Constant){
    uint64_t base = ((cast<ConstantExpr>(addr)->getZExtValue()) >> log_base2(Executor::lineSize)) & (Executor::setNum - 1);
    set = ConstantExpr::create(base, addr->getWidth());
  }else{
    ref<Expr> base = LShrExpr::create(addr, ConstantExpr::create(log_base2(Executor::lineSize), addr->getWidth()));
    set = AndExpr::create(base, ConstantExpr::create(Executor::setNum - 1, addr->getWidth()));
  }
  setCache[addr->hash()] = set;
  return set;
}


ref<Expr> Executor::cmpSet(ref<Expr> &addr1, ref<Expr> &addr2){
  std::pair<unsigned, unsigned> pair(addr1->hash(), addr2->hash());
  if(setPairCache.find(pair) != setPairCache.end()){
    return setPairCache[pair];
  }

  ref<Expr> cmp;
  if (addr1->getKind() == Expr::Constant && addr2->getKind() == Expr::Constant){
    unsigned set1 = cast<ConstantExpr>(getSet(addr1))->getZExtValue();
    unsigned set2 = cast<ConstantExpr>(getSet(addr2))->getZExtValue();
    if (set1 == set2){
      cmp = ConstantExpr::create(1, Expr::Bool);
    }else{
      cmp = ConstantExpr::create(0, Expr::Bool);
    }
  }else{
    ref<Expr> set1 = getSet(addr1);
    ref<Expr> set2 = getSet(addr2);
    cmp = EqExpr::create(set1, set2);
  }
  setPairCache[pair] = cmp;
  return cmp;
}

ref<Expr> Executor::getTag(ref<Expr> addr){
  if(tagCache.find(addr->hash()) != tagCache.end()){
      return tagCache[addr->hash()];
  }

  ref<Expr> tag;
  if(addr->getKind() == Expr::Constant){
    uint64_t base = ((cast<ConstantExpr>(addr)->getZExtValue()) >> (log_base2(Executor::lineSize) + log_base2(Executor::setNum)));
    tag = ConstantExpr::create(base, addr->getWidth());
  }else{
    ref<ConstantExpr> offset = ConstantExpr::create(log_base2(Executor::lineSize) + log_base2(Executor::setNum), addr->getWidth());
    tag = LShrExpr::create(addr, offset);
  }
  tagCache[addr->hash()] = tag;
  return tag;
}


ref<Expr> Executor::cmpTag(ref<Expr> &addr1, ref<Expr> &addr2){
  std::pair<unsigned, unsigned> pair(addr1->hash(), addr2->hash());
  if(tagPairCache.find(pair) != tagPairCache.end()){
    return tagPairCache[pair]; 
  }

  ref<Expr> cmp;
  if(addr1->getKind() == Expr::Constant && addr2->getKind() == Expr::Constant){
    unsigned tag1 = cast<ConstantExpr>(getTag(addr1))->getZExtValue();
    unsigned tag2 = cast<ConstantExpr>(getTag(addr2))->getZExtValue();
    if(tag1 == tag2){
      cmp = ConstantExpr::create(1, Expr::Bool);
    }else{
      cmp = ConstantExpr::create(0, Expr::Bool);
    }
  }else{
    ref<Expr> tag1 = getTag(addr1);
    ref<Expr> tag2 = getTag(addr2);
    cmp = EqExpr::create(tag1, tag2);
  }
  tagPairCache[pair] = cmp;
  return cmp;
}

// Lightweight comparion to find potential pair of same addresses
Solver::Validity Executor::isSameAddr(unsigned crt, unsigned pre, ExecutionState &state, TimingSolver *solver, ref<Expr>& uCnstr){
  assert(pre < crt);

  std::vector<std::pair<ref<Expr>, bool> > &addrs = state.addrs;
  ref<Expr> crtAddr = addrs[crt].first;
  ref<Expr> preAddr = addrs[pre].first;

#ifdef Lewis_DEBUG_CACHE
  if (crtAddr->getKind() == Expr::Constant) {
    ConstantExpr* CE = dyn_cast<ConstantExpr>(crtAddr);
    std::string value;
    CE->toString(value, 16);
    llvm::errs() << "[0x" << value << "]";
  } else  {
    fprintf(stderr, "symbolic [%u]",  crt);
    /*
    ExprSMTLIBPrinter printer;
    printer.setOutput(llvm::errs());
    printer.printMemoryExpression(crtAddr);
    */
  }
  llvm::errs() << " vs. ";

  if (preAddr->getKind() == Expr::Constant) {
    ConstantExpr* CE = dyn_cast<ConstantExpr>(preAddr);
    std::string value;
    CE->toString(value, 16);
    llvm::errs() << "[0x" << value << "]";
  } else  {
    fprintf(stderr, "symbolic [%u]",  pre);
    /*
    ExprSMTLIBPrinter printer;
    printer.setOutput(llvm::errs());
    printer.printMemoryExpression(preAddr);
    */
  }

  llvm::errs() << "\n";
#endif

#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "---- C1 ----\n");
  fprintf(stderr, "0. query the cached comparison results\n");
#endif
  // 0. Query the cached comparison results

#ifdef Lewis_O1
  std::pair<unsigned, unsigned> pair(crtAddr->hash(), preAddr->hash());
  if (cmpResultCache.find(pair) != cmpResultCache.end()) {
    if (cmpResultCache[pair] == Solver::Unknown)
      uCnstr = uCnstrCache[pair];
    return cmpResultCache[pair];
  }
#endif

  // 1. Compare the base MemoryObject name
  std::vector<std::string> &objNames = state.objNames;
  assert(addrs.size() == objNames.size());
  std::string crtName = objNames[crt];
  std::string preName = objNames[pre];
#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "1. compare object name ");
  llvm::errs() << "(" << crtName << ", " << preName << ")";
  fprintf(stderr, " ->> skipped for now\n");
#endif
  // XXX Lewis does not understand why unnamed indicates not same address
  // XXX even if not the same name, can still hit
  // if (crtName != preName || preName == "unnamed"){
  /*
  if (crtName != preName){
    cmpResultCache[pair] = Solver::False;
    return Solver::False;
  }
  */

#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "2. compare set\n");
#endif
  // 2. Compare the set 
  bool mustBeSameSet = false;
  ref<Expr> cmp1 = cmpSet(crtAddr, preAddr); 
  if(cmp1->getKind() == Expr::Constant){
    ref<ConstantExpr> tmp1 = cast<ConstantExpr>(cmp1);   
    if(tmp1->isZero()){
#ifdef Lewis_O1
      cmpResultCache[pair] = Solver::False;
#endif
      return Solver::False;
    }else if(tmp1->isOne()){
      mustBeSameSet = true;
    }
  }else{
    bool mustBeFalse = false; 
    solver->mustBeFalse(state, cmp1, mustBeFalse);
    if(mustBeFalse){
#ifdef Lewis_O1
      cmpResultCache[pair] = Solver::False;
#endif
      return Solver::False;
    }
  }

#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "3. compare tag\n");
#endif 
  // 3. Compare the tag
  bool mustBeSameTag = false;
  ref<Expr> cmp2 = cmpTag(crtAddr, preAddr); 
  if(cmp2->getKind() == Expr::Constant){
    ref<ConstantExpr> tmp2 = cast<ConstantExpr>(cmp2);
    if(tmp2->isZero()){
#ifdef Lewis_O1
      cmpResultCache[pair] = Solver::False;
#endif
      return Solver::False;
    }else if(tmp2->isOne()){
      mustBeSameTag = true;
    }
  }else{
    bool mustBeFalse = false;
    solver->mustBeFalse(state, cmp2, mustBeFalse);
    if(mustBeFalse){
#ifdef Lewis_O1
      cmpResultCache[pair] = Solver::False;
#endif
      return Solver::False;
    }
  }


  // 4. Might be the same addresses
  if(mustBeSameSet && mustBeSameTag){
#ifdef Lewis_O1
    cmpResultCache[pair] = Solver::True;
#endif
    return Solver::True;
  }else{
    uCnstr = AndExpr::create(cmp1, cmp2); // c1
#ifdef Lewis_O1
    cmpResultCache[pair] = Solver::Unknown;
    uCnstrCache[pair] = uCnstr;
#endif
    return Solver::Unknown;
  }
}

bool Executor::isEvictedLRU(unsigned crt, unsigned pre, unsigned ignoreCnt, ExecutionState& state, TimingSolver* solver, ref<Expr>& uCnstr) {
  
#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "0. compare distance\n");
#endif
  // uCnstr is NULL indicates that C1 & C2 = True
  // We further find C3 = True
  if (uCnstr.isNull() && (crt - pre - ignoreCnt) <= Executor::nway)
    return false;

  uint64_t maxFreeWay = Executor::nway;
  std::vector<std::pair<ref<Expr>, bool> > &addrs = state.addrs;
  ref<Expr> preAddr = addrs[pre].first;
  std::vector<ref<Expr> > conflictAddrs;
  ref<Expr> conflictCntCnstr = NULL;
  ref<Expr> conflictImpCnstr = NULL;
  static unsigned cnt = 0;

#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "1. go through access [%u, %u]\n", pre + 1, crt -1);
#endif
  for (unsigned i = pre + 1; i < crt && maxFreeWay > 0; i++) {
    if(addrs[i].second == false)
      continue;

    ref<Expr> idxAddr = addrs[i].first;
    ref<Expr> cmp1 = cmpSet(preAddr, idxAddr);

    bool isNotSameSet = false;
    if (cmp1->getKind() == Expr::Constant) {
      ref<ConstantExpr> tmp = cast<ConstantExpr>(cmp1);
      if (!tmp->isOne()) // not an evict
        isNotSameSet = true;
    } else {
      solver->mustBeFalse(state, cmp1, isNotSameSet);
    }

    if (isNotSameSet) {
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "  [%3u] not same set --> not evict, next\n", i);
#endif
      continue;
    }

    ref<Expr> conflictCnstr = cmp1;
    ref<Expr> tagCnstr = NULL;
    bool alreadyExistSameTag = false;

    for (std::vector<ref<Expr>>::iterator it = conflictAddrs.begin(); 
        it < conflictAddrs.end(); it++) {

      ref<Expr> cmp2 = cmpTag(*it, idxAddr);

      if (cmp2->getKind() == Expr::Constant) {
        ref<ConstantExpr> tmp = cast<ConstantExpr>(cmp2);
        if (tmp->isOne()) {
          alreadyExistSameTag = true;
          break;
        } else
          continue;
      } else {
        bool isSameTag = false;
        solver->mustBeTrue(state, cmp2, isSameTag);
        if (isSameTag) {
          alreadyExistSameTag = true;
          break;
        }
      }

      if (tagCnstr.isNull())
        tagCnstr = NotExpr::create(cmp2);
      else
        tagCnstr = AndExpr::create(tagCnstr, NotExpr::create(cmp2));
    }

    if(alreadyExistSameTag)
      continue;

    conflictAddrs.push_back(idxAddr);

    if (!tagCnstr.isNull()) // tagCnstr can be NULL if evictAddr is empty
      conflictCnstr = AndExpr::create(conflictCnstr, tagCnstr);
      
    bool isConflict = false;
    solver->mustBeTrue(state, conflictCnstr, isConflict);
    if (isConflict) {
      maxFreeWay--;
      continue;
    } 

    bool isNotConflict = false;
    solver->mustBeFalse(state, conflictCnstr, isNotConflict);
    if (isNotConflict) {
      continue;
    }

    std::string name = "conflictCnstr" + llvm::utostr(++cnt) + "_" + 
                  llvm::utostr(crt) + "_" + llvm::utostr(pre);
    ref<Expr> read = Expr::createTempRead(arrayCache.CreateArray(name, 1), 8);
    ref<Expr> conflict = EqExpr::create(read, ConstantExpr::create(1, read->getWidth()));
    
    if (conflictImpCnstr.isNull())
      conflictImpCnstr = Expr::createImplies(conflictCnstr, conflict);
    else
      conflictImpCnstr = AndExpr::create(conflictImpCnstr, 
          Expr::createImplies(conflictCnstr, conflict));

    if (conflictCntCnstr.isNull())
      conflictCntCnstr = read;
    else
      conflictCntCnstr = AddExpr::create(conflictCntCnstr, read);
  }

  if (maxFreeWay == 0) // must conflict
    return true;

  if (conflictImpCnstr.isNull()) // no conflict
    return false;

  // force to 64 bits width
  conflictCntCnstr = SExtExpr::create(conflictCntCnstr, 64);
#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "maxFreeWay: %lu, conflictCntCnstr width %u\n", maxFreeWay, conflictCntCnstr->getWidth());
#endif

  ref<Expr> max = ConstantExpr::create(maxFreeWay, conflictCntCnstr->getWidth());
  ref<Expr> C3 = AndExpr::create(conflictImpCnstr, UleExpr::create(conflictCntCnstr, max));

  bool isEvict = false;
  solver->mustBeFalse(state, C3, isEvict);
  if (isEvict)
    return true;

  if (uCnstr.isNull())
    uCnstr = C3;
  else
    uCnstr = AndExpr::create(uCnstr, C3);

  return false;
}

bool Executor::simpleAnalyzeMemCache(ExecutionState &state, TimingSolver* solver) {
  if (state.regObj) {
#ifdef Lewis_DEBUG_SPECU
    fprintf(stderr, "[+] branch comparison between registers\n");
#endif
    return true;
  }
  
  if (state.addrs.empty()) {
#ifdef Lewis_DEBUG_SPECU
    fprintf(stderr, "[+] branch comparison between constant\n");
#endif
    return true;
  }

// #if defined(Lewis_DEBUG_SPECU) && defined(Lewis_DEBUG_CACHE)
  // printMemoryAddr(state);
// #endif

  // state.marks.empty() means no speculative exec till now
  if(state.marks.empty()) {
    state.simpleModeFlag = true;
    unsigned crt = state.addrs.size() - 1;
    bool isHit = analyzeThisAccess(state, solver, crt);
    state.simpleModeFlag = false;
#ifdef Lewis_DEBUG_SPECU
    if (isHit)
      fprintf(stderr, "[+] branch comparison's memory access hits\n");
    else 
      fprintf(stderr, "[+] branch comparison's memory access miss\n");
#endif
    return isHit;
  }

  // reserve
  std::vector<std::pair<ref<Expr>, bool> > copyAddrs = state.addrs;
  std::vector<std::string> copyObjNames = state.objNames;
  std::vector<const InstructionInfo*> copyPcInfos = state.pcInfos;
  std::vector<std::pair<uint64_t, uint64_t>> copyMarks = state.marks;
  
  // reshape
  if (state.nestedCnt == 0) { // simple mode only consider last piece of normal execution
    uint64_t offset = state.marks.back().second;
    state.addrs.erase(state.addrs.begin(), state.addrs.begin()+offset);
    state.objNames.erase(state.objNames.begin(), state.objNames.begin()+offset);
    state.pcInfos.erase(state.pcInfos.begin(), state.pcInfos.begin()+offset);
    state.marks.clear();
  } else { // simple mode only consider memory access in current speculative
    state.marks.clear();
  }

  bool isHit = false;
  if (state.addrs.empty()) {
    isHit = false;
  } else {
    state.simpleModeFlag = true;
    unsigned crt = state.addrs.size() - 1;
    isHit = analyzeThisAccess(state, solver, crt);
    state.simpleModeFlag = false;
  }

  // restore
  state.addrs = copyAddrs;
  state.objNames = copyObjNames;
  state.pcInfos = copyPcInfos;
  state.marks = copyMarks;
  if (isHit)
    fprintf(stderr, "[+] branch comparison's memory access hits\n");
  else 
    fprintf(stderr, "[+] branch comparison's memory access miss\n");
  return isHit;
}

// off-line analysis
void Executor::analyzeMemCache(ExecutionState &state, TimingSolver *solver) {

  printMemoryAddr(state);
  unsigned specNum = state.marks.size();
  if (specNum == 0) {
    fprintf(stderr, "[+] No speculative execution in state %lu, skip analysis\n", state.id);
    return;
  }
  fprintf(stderr, "\n##################### Analyzing State %lu ##################\n", state.id);

  state.analyzeNormalFlag = false;
  state.normalExecResult.clear();
  state.simpleModeFlag = false;


  unsigned maxEnabledSpec = specNum/10 > 1 ? specNum/10 : 1;
  // unsigned specNum = 3;
  // unsigned maxEnabledSpec = 2;

  // Using LRU policy, aff (A+B) >= aff(B) while aff(A+B) and aff(A) are independent,
  // we can do some optimization to reduce # of combination of speculative exec for analysis
  // Given specNum = 4, max = 3, following algorithm runs as
  // 1, 12, 13, 123, 14, 124, 134, 234
  std::vector<std::vector<unsigned>> analyzeQ;
  std::vector<unsigned> first(1, 0);
  analyzeQ.push_back(first);

  for(unsigned i = 1; i < specNum; i++) {
    if (maxEnabledSpec == 1) {
      std::vector<unsigned> single(1, i);
      analyzeQ.push_back(single);
    } else {
      std::vector<std::vector<unsigned>> oldAnalyzeQ = analyzeQ;
      unsigned high = 0;
      for (std::vector<std::vector<unsigned>>::iterator II = oldAnalyzeQ.begin(); 
          II != oldAnalyzeQ.end(); II++) {
        std::vector<unsigned> newone;
        if (II->size() == maxEnabledSpec) {
          if (II->front() <= high) { // to prevent duplication
            newone.insert(newone.begin(), II->begin()+1, II->end());
            newone.push_back(i);
            analyzeQ.push_back(newone);
          }
          high = II->front();
        } else {
          newone.insert(newone.begin(), II->begin(), II->end());
          newone.push_back(i);
          analyzeQ.push_back(newone);
        }
      }
    }
  }

  if (specNum == 0) { // no spec
    analyzeQ.clear();
    maxEnabledSpec = 0;
  }
  std::vector<unsigned> ept; // empty means no speculative exec
  analyzeQ.insert(analyzeQ.begin(), ept);

  fprintf(stderr, "[+] At most  %u/%u to be enabled\n", maxEnabledSpec, specNum);
  fprintf(stderr, "[+] In total %lu situations to analyze (including normal exec)\n", 
      analyzeQ.size());

  for (std::vector<std::vector<unsigned>>::iterator II = analyzeQ.begin(); 
      II != analyzeQ.end(); II++) {

    // allow all speculative exec
    for (std::vector<std::pair<ref<Expr>, bool> >::iterator JJ = state.addrs.begin(); 
        JJ != state.addrs.end(); JJ++)
      JJ->second = true;

    unsigned startP = 0;
    if (II->empty())
      state.analyzeNormalFlag = true;
    fprintf(stderr, "\n------------- Enable ");
    for (std::vector<std::pair<uint64_t, uint64_t>>::iterator JJ = state.marks.begin();
        JJ != state.marks.end(); JJ++) {
      unsigned offset = JJ - state.marks.begin();
      unsigned start = JJ->first;
      unsigned end = JJ->second;
      if (std::find(II->begin(), II->end(), offset)!=II->end()) {
        fprintf(stderr, "%u/[%u, %u), ", offset, start, end);
        startP = JJ->second;
      } else {
        // mask disabled speculative exec
        for (std::vector<std::pair<ref<Expr>, bool> >::iterator KK = 
            state.addrs.begin() + start; KK != (state.addrs.begin() + end); KK++)
          KK->second = false;
      }
    }
    fprintf(stderr, "-------------\n");

    assert(state.addrs.size() == state.pcInfos.size());
    for (unsigned i = startP; i < state.addrs.size(); i++) {
      bool isDuplicated = false;
#ifdef Lewis_O1
      for (std::vector<unsigned>::iterator II = detectedLocation.begin();
          II != detectedLocation.end(); II++) {
        if (*II == state.pcInfos[i]->assemblyLine) {
          fprintf(stderr, "[%u] duplicated memory access, skip\n", i);
          state.normalExecResult.push_back(false);
          isDuplicated = true;
          break;
        }
      }
#endif
      if (!isDuplicated)
        analyzeThisAccess(state, solver, i);
    }

    // as long as detecting leakage in at least one speculative execution behavior
//    state.detectedLocation.clear();

    fprintf(stderr, "------------- End Enable ");
    for (std::vector<unsigned>::iterator JJ = II->begin(); JJ!= II->end(); JJ++) {
      fprintf(stderr, "%u ", *JJ);
    }
    if (II->empty())
      state.analyzeNormalFlag = false;
    fprintf(stderr, "-------------\n");
  }

  fprintf(stderr, "################# End Analyzing State %lu ##################\n\n", state.id);
}

// return value is used iff. state is in simple mode
bool Executor::analyzeThisAccess(ExecutionState &state, TimingSolver *solver, unsigned crt) {

  // simpleModeFlag is online, analyzeNormalFlag is offline
  // never be true at the same time
  assert(!(state.simpleModeFlag && state.analyzeNormalFlag));

  std::vector<std::pair<ref<Expr>, bool> > &addrs = state.addrs;

  if (crt == 0) {
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(false);
    }
    return false;
  }
  // memory access in speculative exec, ignore
  if (addrs[crt].second == false) {
#ifdef Lewis_DEBUG_CACHE
    fprintf(stderr, "[%u] memory access in speculative exec, ignore\n", crt);
#endif
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(false);
    } else {
      ;
    }
    return false;
  }

#ifdef Lewis_O2
  ref<Expr> crtAddr = addrs[crt].first;
  bool hasFoundNearest = false;
  ref<Expr> gC2 = NULL;
  ref<Expr> gUCnstr = NULL;
  unsigned missCnt = 0;
  unsigned ignoreCnt = 0;

  // to cater to constraint solver
  int traverseThrehold = (crt < Executor::nway*10) ? 0 : (crt - Executor::nway*10);

  // for (int pre=crt-1; pre>=0; pre--) {
  for (int pre=crt-1; pre>=traverseThrehold;pre--) {
    if (addrs[pre].second == false) { // ignore this and regard as miss
      ignoreCnt++;
      continue;
    }
    // all local uCnstr before nearest is false because C2 is always false
    if (hasFoundNearest)
      break; 

    ref<Expr> preAddr = addrs[pre].first;
    ref<Expr> uCnstr = NULL;

    // uCnstr = C1 (same address)
    Solver::Validity validity = isSameAddr(crt, pre, state, solver, uCnstr); 
    if (validity == Solver::False) { 
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "---- C1 is false ----\n");
#endif
      missCnt++;
      continue; // optimize 1: c1 doesn't hold
    } else if (validity == Solver::True) {
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "---- C1 is true ----\n");
#endif
      // uCnstr = C1 & C2 = True & C2 = C2 (nearest)
      uCnstr = gC2; 
      hasFoundNearest = true; // at worst this one is
    } else {
      assert(validity == Solver::Unknown && !uCnstr.isNull());
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "---- C1 is unknown ----\n");
#endif
      // optimize 2: reuse gC2 from previous iteration
      ref<Expr> lC2 = NotExpr::create(uCnstr);
      if (gC2.isNull()) {
        gC2 = lC2;
      } else {
        // uCnstr = C1 & C2
        uCnstr = AndExpr::create(uCnstr, gC2); 

        // update gC2 for next iteration
        gC2 = AndExpr::create(gC2, lC2); 

        // optimize 3: introducing extra C1 may ease solver to determine nearest
        // Lewis is not sure if this is a real optimization. check it in experiment
        /* AES test indicates that it is not a real optimization, skip
        bool c1c2isTrue = false;
        solver->mustBeTrue(state, uCnstr, c1c2isTrue);
        if(c1c2isTrue) {
          uCnstr = NULL;
          hasFoundNearest = true;
          fprintf(stderr, "[+] optimize 3 is a real optimization,  grap beer !!!\n");
        }
        */
      }
    }

#ifdef Lewis_DEBUG_CACHE
    if (uCnstr.isNull())
      fprintf(stderr, "---- C1 & C2 is true ----\n");
    else
      fprintf(stderr, "---- C1 & C2 is unknown ----\n");

    fprintf(stderr, "---- C1 & C2 & C3 ----\n");
#endif
    // uCnstr = C1 & C2 & C3 (not Evicted)
    // if true, all local uCnstr before is fasle because C3 is always false
    if (isEvictedLRU(crt, pre, ignoreCnt, state, solver, uCnstr)) {
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "---- C1 & C2 & C3 is false ----\n");
#endif
      break;
    }

    // uCnstr is surely true
    if (uCnstr.isNull()) {
      if (state.simpleModeFlag) {
        ; // simple mode never archives results
      } else if (state.analyzeNormalFlag) {
        assert(state.normalExecResult.size() == crt);
        state.normalExecResult.push_back(true);
      } else if (!state.normalExecResult[crt]) { // miss in normal exec
          fprintf(stderr, "[%u] Detect Opposite! normal-miss, spec-hit\n", crt);
          detectedLocation.push_back(state.pcInfos[crt]->assemblyLine);
      } else {
        ; 
      }
#ifdef Lewis_DEBUG_CACHE
      fprintf(stderr, "---- C1 & C2 & C3 is true ----\n");
      fprintf(stderr, "[%u] always hit in state %lu\n", crt, state.id);
#endif
      return true; // always hit, no leak, cheers
    }

#ifdef Lewis_DEBUG_CACHE
    fprintf(stderr, "---- C1 & C2 & C3 is unknown ----\n");
#endif
    
    // update gUCnstr by OR local uCnstr
    if (gUCnstr.isNull()) 
      gUCnstr = uCnstr;
    else
      gUCnstr = OrExpr::create(gUCnstr, uCnstr);
  }

  // simple mode avoid bothering constraint solver too much
  if (state.simpleModeFlag)
    return false;

  // if ((missCnt+ignoreCnt) == crt) {
  if ((missCnt+ignoreCnt) == crt || gUCnstr.isNull()) {
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(false);
    } else if (state.normalExecResult[crt]) { // hit in normal exec
        fprintf(stderr, "[%u] Detect Opposite! normal-hit, spec-miss\n", crt);
        detectedLocation.push_back(state.pcInfos[crt]->assemblyLine);
    } else {
      ;
    }
#ifdef Lewis_DEBUG_CACHE
    fprintf(stderr, "[%u] always miss in state %lu\n", crt, state.id);
#endif
    return false;
  }


  // uConstr == NULL means always false
  /*
  if (gUCnstr.isNull()) {
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(false);
    }
    return false;
  }
  */

  fprintf(stderr, "[%u] begin detecting @ %s:%u\n", crt, 
      state.pcInfos[crt]->file.c_str(), state.pcInfos[crt]->line);
  if (state.analyzeNormalFlag) {
    fprintf(stderr, "[+] assume hit in state %lu due to normal exec\n", state.id);
    assert(state.normalExecResult.size() == crt);
    state.normalExecResult.push_back(true);
  } else {
    if (detectLeak(state, gUCnstr, solver))
      detectedLocation.push_back(state.pcInfos[crt]->assemblyLine);
  }

  fprintf(stderr, "[%u] end detecting leakage\n", crt);
  return false;
#else // Lewis_O2
  ref<Expr> crtAddr = addrs[crt].first;
  ref<Expr> gC2 = NULL;
  ref<Expr> gUCnstr = NULL;
  unsigned ignoreCnt = 0;
  int traverseThrehold = (crt < Executor::nway*10) ? 0 : (crt - Executor::nway*10);

  for (int pre=crt-1; pre>=traverseThrehold;pre--) {
    if (addrs[pre].second == false) { // ignore this and regard as miss
      ignoreCnt++;
      continue;
    }
    ref<Expr> preAddr = addrs[pre].first;
    ref<Expr> uCnstr = NULL;
    Solver::Validity validity = isSameAddr(crt, pre, state, solver, uCnstr);
    if (validity == Solver::False) {
      uCnstr = ConstantExpr::create(0, Expr::Bool);
    } else if (validity == Solver::True) {
      uCnstr = ConstantExpr::create(1, Expr::Bool);
    } else {
      assert(validity == Solver::Unknown && !uCnstr.isNull());
    }
    ref<Expr> lC2 = NotExpr::create(uCnstr);
    if (gC2.isNull()) {
      gC2 = lC2;
    } else {
      uCnstr = AndExpr::create(uCnstr, gC2);
      gC2 = AndExpr::create(gC2, lC2);
    }
    isEvictedLRU(crt, pre, ignoreCnt, state, solver, uCnstr);
    if (gUCnstr.isNull())
      gUCnstr = uCnstr;
    else
      gUCnstr = OrExpr::create(gUCnstr, uCnstr);
  }

  if (gUCnstr.isNull()) {
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(true);
    }
    return true;
  }

  bool mustbeFalse = false;
  solver->mustBeFalse(state, gUCnstr, mustbeFalse);
  if (mustbeFalse) {
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(false);
    }
    return false;
  }

  bool mustbeTrue = false;
  solver->mustBeTrue(state, gUCnstr, mustbeTrue);
  if (mustbeTrue) {
    if (state.analyzeNormalFlag) {
      assert(state.normalExecResult.size() == crt);
      state.normalExecResult.push_back(true);
    }
    return true;
  }

  if (state.analyzeNormalFlag) {
    assert(state.normalExecResult.size() == crt);
    state.normalExecResult.push_back(true);
    return true;
  }

  if (state.simpleModeFlag) {
    ExecutionState* cState = new ExecutionState(state);
    cState->addConstraint(gUCnstr);
    std::vector< std::vector<unsigned char> > hitValues;
    std::vector<const Array*> hitObjects;
    bool success = solver->getInitialValues(*cState, hitObjects, hitValues);
    delete(cState);
    return success;
  }

  detectLeak(state, gUCnstr, solver);
  return false;
#endif // Lewis_O2
}

/******************************* End Cache Modeling *********************************/

/********************** Speculative Execution Modeling ***************/

void Executor::stopSpeculativeExecution(ExecutionState &state) {
	ExecutionState *par= state.parState;	
  // assert(par->stateType == ExecutionState::SYMBOLIC && "Parent state type must be SYMBOLIC.");	
  assert(par->nestedCnt < MaxSpeculativeDepth && "Out of speculative execution depth.");
  assert(!par->continueFlag && "Parent state must be paused.");
  assert(par->addrs.size() == par->objNames.size() &&
      par->addrs.size() == par->pcInfos.size());

  if (par->nestedCnt == 0) {
    std::pair<uint64_t, uint64_t> pr = std::make_pair(par->addrs.size(), par->addrs.size() + state.addrs.size());
    par->marks.push_back(pr);
  }

  par->addrs.reserve(par->addrs.size() + state.addrs.size());	
  par->addrs.insert(par->addrs.end(), state.addrs.begin(), state.addrs.end());	

  par->objNames.reserve(par->objNames.size() + state.objNames.size());	
  par->objNames.insert(par->objNames.end(), state.objNames.begin(), state.objNames.end());	

  par->pcInfos.reserve(par->pcInfos.size() + state.pcInfos.size());	
  par->pcInfos.insert(par->pcInfos.end(), state.pcInfos.begin(), state.pcInfos.end());	

  par->continueFlag = true;
#ifdef Lewis_DEBUG_SPECU
  fprintf(stderr, "[+] Continue symbolic state %lu\n", par->id);
#endif 
}

Executor::StatePair 	
Executor::fork(ExecutionState &current, ref<Expr> condition, bool isInternal, ExecutionState*& specu1, ExecutionState*& specu2) {	
  assert(SpeculativeModeling);	

   Solver::Validity res;	
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 	
    seedMap.find(&current);	
  bool isSeeding = it != seedMap.end();	

   if (!isSeeding && !isa<ConstantExpr>(condition) && 	
      (MaxStaticForkPct!=1. || MaxStaticSolvePct != 1. ||	
       MaxStaticCPForkPct!=1. || MaxStaticCPSolvePct != 1.) &&	
      statsTracker->elapsed() > time::seconds(60)) {	
    StatisticManager &sm = *theStatisticManager;	
    CallPathNode *cpn = current.stack.back().callPathNode;	
    if ((MaxStaticForkPct<1. &&	
         sm.getIndexedValue(stats::forks, sm.getIndex()) > 	
         stats::forks*MaxStaticForkPct) ||	
        (MaxStaticCPForkPct<1. &&	
         cpn && (cpn->statistics.getValue(stats::forks) > 	
                 stats::forks*MaxStaticCPForkPct)) ||	
        (MaxStaticSolvePct<1 &&	
         sm.getIndexedValue(stats::solverTime, sm.getIndex()) > 	
         stats::solverTime*MaxStaticSolvePct) ||	
        (MaxStaticCPForkPct<1. &&	
         cpn && (cpn->statistics.getValue(stats::solverTime) > 	
                 stats::solverTime*MaxStaticCPSolvePct))) {	
      ref<ConstantExpr> value; 	
      bool success = solver->getValue(current, condition, value);	
      assert(success && "FIXME: Unhandled solver failure");	
      (void) success;	
      addConstraint(current, EqExpr::create(value, condition));	
      condition = value;	
    }	
  }	

   time::Span timeout = coreSolverTimeout;	
  if (isSeeding)	
    timeout *= static_cast<unsigned>(it->second.size());	
  solver->setTimeout(timeout);	
  bool success = solver->evaluate(current, condition, res);	
  solver->setTimeout(time::Span());	
  if (!success) {	
    current.pc = current.prevPC;	
    terminateStateEarly(current, "Query timed out (fork).");	
    return StatePair(0, 0);	
  }	

   if (!isSeeding) {	
    if (replayPath && !isInternal) {	
      assert(replayPosition<replayPath->size() &&	
             "ran out of branches in replay path mode");	
      bool branch = (*replayPath)[replayPosition++];	

       if (res==Solver::True) {	
        assert(branch && "hit invalid branch in replay path mode");	
      } else if (res==Solver::False) {	
        assert(!branch && "hit invalid branch in replay path mode");	
      } else {	
        // add constraints	
        if(branch) {	
          res = Solver::True;	
          addConstraint(current, condition);	
        } else  {	
          res = Solver::False;	
          addConstraint(current, Expr::createIsZero(condition));	
        }	
      }	
    } else if (res==Solver::Unknown) {	
      assert(!replayKTest && "in replay mode, only one branch can be true.");	

       if ((MaxMemoryInhibit && atMemoryLimit) || 	
          current.forkDisabled ||	
          inhibitForking || 	
          (MaxForks!=~0u && stats::forks >= MaxForks)) {	

 	if (MaxMemoryInhibit && atMemoryLimit)	
	  klee_warning_once(0, "skipping fork (memory cap exceeded)");	
	else if (current.forkDisabled)	
	  klee_warning_once(0, "skipping fork (fork disabled on current path)");	
	else if (inhibitForking)	
	  klee_warning_once(0, "skipping fork (fork disabled globally)");	
	else 	
	  klee_warning_once(0, "skipping fork (max-forks reached)");	

         TimerStatIncrementer timer(stats::forkTime);	
        if (theRNG.getBool()) {	
          addConstraint(current, condition);	
          res = Solver::True;        	
        } else {	
          addConstraint(current, Expr::createIsZero(condition));	
          res = Solver::False;	
        }	
      }	
    }	
  }	

   // Fix branch in only-replay-seed mode, if we don't have both true	
  // and false seeds.	
  if (isSeeding && 	
      (current.forkDisabled || OnlyReplaySeeds) && 	
      res == Solver::Unknown) {	
    bool trueSeed=false, falseSeed=false;	
    // Is seed extension still ok here?	
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 	
           siie = it->second.end(); siit != siie; ++siit) {	
      ref<ConstantExpr> res;	
      bool success = 	
        solver->getValue(current, siit->assignment.evaluate(condition), res);	
      assert(success && "FIXME: Unhandled solver failure");	
      (void) success;	
      if (res->isTrue()) {	
        trueSeed = true;	
      } else {	
        falseSeed = true;	
      }	
      if (trueSeed && falseSeed)	
        break;	
    }	
    if (!(trueSeed && falseSeed)) {	
      assert(trueSeed || falseSeed);	

       res = trueSeed ? Solver::True : Solver::False;	
      addConstraint(current, trueSeed ? condition : Expr::createIsZero(condition));	
    }	
  }	

/******************************* Speculative Execution Modeling *****************/    
  bool isHit = simpleAnalyzeMemCache(current, solver);
  /*
  if (isHit) {
    fprintf(stderr, "[+] no speculative fork\n");
  }
  */
/*************************** End Speculative Execution Modeling *****************/    

   // XXX - even if the constraint is provable one way or the other we	
  // can probably benefit by adding this constraint and allowing it to	
  // reduce the other constraints. For example, if we do a binary	
  // search on a particular value, and then see a comparison against	
  // the value it has been fixed at, we should take this as a nice	
  // hint to just use the single constraint instead of all the binary	
  // search ones. If that makes sense.	
  if (res==Solver::True) {	
    if (!isInternal) {	
      if (pathWriter) {	
        current.pathOS << "1";	
      }	
    }	
/******************************* Speculative Execution Modeling *****************/    
    if (!isHit && current.nestedCnt < MaxSpeculativeDepth) {
      ExecutionState *trueState = &current;
      specu1 = trueState->branch();
      trueState->childId = specu1->id;
      specu1->stateType = ExecutionState::SPECULATIVE;
      specu1->addrs.clear();
      specu1->objNames.clear();
      specu1->pcInfos.clear();
      specu1->nestedCnt++;
      addedStates.push_back(specu1);

      trueState->ptreeNode->data = 0;
      std::pair<PTree::Node*, PTree::Node*> res =
        processTree->split(trueState->ptreeNode, specu1, trueState);
      specu1->ptreeNode = res.first;
      trueState->ptreeNode = res.second;
    }
/*************************** End Speculative Execution Modeling *****************/    
     return StatePair(&current, 0);	
  } else if (res==Solver::False) {	
    if (!isInternal) {	
      if (pathWriter) {	
        current.pathOS << "0";	
      }	
    }	
/******************************* Speculative Execution Modeling *****************/    
    if (!isHit && current.nestedCnt < MaxSpeculativeDepth) {
      ExecutionState *falseState = &current;
      specu2 = falseState->branch();
      falseState->childId = specu2->id;
      specu2->stateType = ExecutionState::SPECULATIVE;
      specu2->addrs.clear();
      specu2->objNames.clear();
      specu2->pcInfos.clear();
      specu2->nestedCnt++;
      addedStates.push_back(specu2);

      falseState->ptreeNode->data = 0;
      std::pair<PTree::Node*, PTree::Node*> res =
        processTree->split(falseState->ptreeNode, specu2, falseState);
      specu2->ptreeNode = res.first;
      falseState->ptreeNode = res.second;
    }
/*************************** End Speculative Execution Modeling *****************/    
     return StatePair(0, &current);	
  } else {	
    TimerStatIncrementer timer(stats::forkTime);	
    ExecutionState *falseState, *trueState = &current;	

    ++stats::forks;	

    falseState = trueState->branch();	

/******************************* Speculative Execution Modeling *****************/    

    if (!isHit && trueState->nestedCnt < MaxSpeculativeDepth) {
      // Set parent state 	
      specu1 = trueState->branch();	
      specu2 = falseState->branch();	

      // set parent's childId
      trueState->childId = specu1->id;
      falseState->childId = specu2->id;

      // Update the state type 	
      specu1->stateType = ExecutionState::SPECULATIVE;	
      specu2->stateType = ExecutionState::SPECULATIVE;	

      // clear addrs in speculative state	
      specu1->addrs.clear();	
      specu2->addrs.clear();	

      // clear objNames in speculative state	
      specu1->objNames.clear();	
      specu2->objNames.clear();	

      // clear objNames in speculative state	
      specu1->pcInfos.clear();	
      specu2->pcInfos.clear();	

      specu1->nestedCnt++;
      specu2->nestedCnt++;

      addedStates.push_back(specu1);	
      addedStates.push_back(specu2);	
    }

/************************** End Speculative Execution Modeling *****************/
    // Add new states into the state pool	
    addedStates.push_back(falseState);	

    if (it != seedMap.end()) {
      std::vector<SeedInfo> seeds = it->second;	
      it->second.clear();	
      std::vector<SeedInfo> &trueSeeds = seedMap[trueState];	
      std::vector<SeedInfo> &falseSeeds = seedMap[falseState];	
      for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 	
             siie = seeds.end(); siit != siie; ++siit) {	
        ref<ConstantExpr> res;	
        bool success = 	
          solver->getValue(current, siit->assignment.evaluate(condition), res);	
        assert(success && "FIXME: Unhandled solver failure");	
        (void) success;	
        if (res->isTrue()) {	
          trueSeeds.push_back(*siit);	
        } else {	
          falseSeeds.push_back(*siit);	
        }	
      }	

       bool swapInfo = false;	
      if (trueSeeds.empty()) {	
        if (&current == trueState) swapInfo = true;	
        seedMap.erase(trueState);	
      }	
      if (falseSeeds.empty()) {	
        if (&current == falseState) swapInfo = true;	
        seedMap.erase(falseState);	
      }	
      if (swapInfo) {	
        std::swap(trueState->coveredNew, falseState->coveredNew);	
        std::swap(trueState->coveredLines, falseState->coveredLines);	
      }	
    }	

    current.ptreeNode->data = 0;	
    std::pair<PTree::Node*, PTree::Node*> res =	
    processTree->split(current.ptreeNode, falseState, trueState);	
    falseState->ptreeNode = res.first;	
    trueState->ptreeNode = res.second;	

/******************************* Speculative Execution Modeling *****************/

    if (!isHit && trueState->nestedCnt < MaxSpeculativeDepth) {
      // set processTree
      trueState->ptreeNode->data = 0;
      res = processTree->split(trueState->ptreeNode, specu1, trueState);
      specu1->ptreeNode = res.first;
      trueState->ptreeNode = res.second;

      falseState->ptreeNode->data = 0;
      res = processTree->split(falseState->ptreeNode, specu2, falseState);
      specu2->ptreeNode = res.first;
      falseState->ptreeNode = res.second;
    }

/************************** End Speculative Execution Modeling *****************/

     if (pathWriter) {	
      // Need to update the pathOS.id field of falseState, otherwise the same id	
      // is used for both falseState and trueState.	
      falseState->pathOS = pathWriter->open(current.pathOS);	
      if (!isInternal) {	
        trueState->pathOS << "1";	
        falseState->pathOS << "0";	
      }	
    }	
    if (symPathWriter) {	
      falseState->symPathOS = symPathWriter->open(current.symPathOS);	
      if (!isInternal) {	
        trueState->symPathOS << "1";	
        falseState->symPathOS << "0";	
      }	
    }	

    addConstraint(*trueState, condition);	
    addConstraint(*falseState, Expr::createIsZero(condition));	

     // Kinda gross, do we even really still want this option?	
    if (MaxDepth && MaxDepth<=trueState->depth) {	
      terminateStateEarly(*trueState, "max-depth exceeded.");	
      terminateStateEarly(*falseState, "max-depth exceeded.");	
      return StatePair(0, 0);	
    }	

    if (!isHit && trueState->nestedCnt < MaxSpeculativeDepth)
      assert(specu1 && specu2);	
    return StatePair(trueState, falseState);	
  }	
}

/********************** End Speculative Execution Modeling ************/

/*********************** Cache Behavior Analysis **********************/

bool Executor::detectLeak(ExecutionState& state, ref<Expr>& hitCnstr, TimingSolver* solver) {

  /* ===== hitCnstr & pathCnstr ======*/
  ExecutionState* cState = new ExecutionState(state);
  cState->addConstraint(hitCnstr);

#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "///////// printing hitCnstr & pathCnstr /////////\n");
  ConstraintManager cm = cState->constraints;
  ExprSMTLIBPrinter printer;
  printer.setOutput(llvm::errs());
  Query query(cm, ConstantExpr::alloc(0, Expr::Bool));
  printer.setQuery(query);
  printer.generateOutput();
  fprintf(stderr, "/////// end printing hitCnstr & pathCnstr //////\n");
#endif

  std::vector< std::vector<unsigned char> > hitValues;
  std::vector<const Array*> hitObjects;
  for (unsigned SI = 0; SI != state.symbolics.size(); SI++) {
#ifdef Lewis_DEBUG_CACHE
    llvm::errs() << "\"" << state.symbolics[SI].first->name << "\" ";
    fprintf(stderr, "[0x%lX]\n", state.symbolics[SI].first->address);
#endif
    hitObjects.push_back(state.symbolics[SI].second);
  }

  bool success = solver->getInitialValues(*cState, hitObjects, hitValues);
  if (!success) {
    if (state.analyzeNormalFlag)
      state.normalExecResult.push_back(false);
    fprintf(stderr, "[-] no solution for hitCnstr & pathCnstr\n");
    delete(cState);
    return false;
  }
  delete(cState);

  /* ===== notHitCnstr & pathCnstr ======*/
  ref<Expr> notHitCnstr = NotExpr::create(hitCnstr);
  cState = new ExecutionState(state);
  cState->addConstraint(notHitCnstr);

#ifdef Lewis_DEBUG_CACHE
  fprintf(stderr, "///////// printing notHitCnstr & pathCnstr /////////\n");
  ConstraintManager cm2 = cState->constraints;
  Query query2(cm2, ConstantExpr::alloc(0, Expr::Bool));
  printer.setQuery(query2);
  printer.generateOutput();
  fprintf(stderr, "/////// end printing notHitCnstr & pathCnstr //////\n");
#endif

  std::vector< std::vector<unsigned char> > notHitValues;
  std::vector<const Array*> notHitObjects;
  for (unsigned SI = 0; SI != state.symbolics.size(); SI++) {
#ifdef Lewis_DEBUG_CACHE
    llvm::errs() << "\"" << state.symbolics[SI].first->name << "\" ";
    fprintf(stderr, "[0x%lX]\n", state.symbolics[SI].first->address);
#endif
    notHitObjects.push_back(state.symbolics[SI].second);
  }
  
  success = solver->getInitialValues(*cState, notHitObjects, notHitValues);
  if (!success) {
    if (state.analyzeNormalFlag)
      state.normalExecResult.push_back(true);
    fprintf(stderr, "[-] no solution for notHitCnstr & pathCnstr\n");
    delete(cState);
    return false;
  }
  delete(cState);

  if (state.analyzeNormalFlag) {
    // Given no leak in normal execution, Lewis directly assume hit to avoid
    // bothering constraint solver.
    state.normalExecResult.push_back(true);
    fprintf(stderr, "[+] assume hit in state %lu due to normal exec\n", state.id);
    return false;
  }

  /* ===== Leak Print solution ===== */
  fprintf(stderr, "[+] printing one solution for HIT\n");
  for (unsigned SI = 0; SI < state.symbolics.size(); SI++) {
    llvm::errs() << state.symbolics[SI].first->name << " = ";
    // llvm::errs() << "(size " << state.symbolics[SI].second->size << ", little-endian)\n";
    /*
    for (unsigned II = 0; II < hitValues[SI].size(); II++)
      fprintf(stderr, "%d", hitValues[SI][II]);
     */
    fprintf(stderr, "0x");
    for (int II = hitValues[SI].size() - 1; II > -1; II--)
      fprintf(stderr, "%x", hitValues[SI][II]);
    fprintf(stderr, "\n");
    fflush(stderr);
  }

  fprintf(stderr, "[+] printing one solution for MISS\n");
  for (unsigned SI = 0; SI < state.symbolics.size(); SI++) {
    llvm::errs() << state.symbolics[SI].first->name << " = ";
    // llvm::errs() << "(size " << state.symbolics[SI].second->size << ", little-endian)\n";
    /*
    for (unsigned II = 0; II < notHitValues[SI].size(); II++)
      fprintf(stderr, "%d", notHitValues[SI][II]);
     */
    fprintf(stderr, "0x");
    for (int II = notHitValues[SI].size() -1 ; II > -1; II--)
      fprintf(stderr, "%x", notHitValues[SI][II]);
    fprintf(stderr, "\n");
    fflush(stderr);
  }

	return true;
}

/*********************** End Cache Behavior Analysis *******************/

/*********************** Debug Util *****************************/

void Executor::printMemoryAddr(ExecutionState& state) {
  fprintf(stderr, "=================== Printing all addresses in state %lu =================\n", state.id);
  fflush(stderr);
  unsigned int cnt = 0;
  
  for (std::vector<std::pair<ref<Expr>, bool> >::iterator II = state.addrs.begin(); II != state.addrs.end(); II++, cnt++) {
    ref<Expr> addr = II->first;
    if (addr->getKind() == Expr::Constant) {
      ConstantExpr* CE = dyn_cast<ConstantExpr>(addr);
      std::string value;
      CE->toString(value, 16);
      fprintf(stderr, "[%3u] Constant memory address::=>", cnt);
      llvm::errs() << "[0x" << value << "]";
      fprintf(stderr, " -- %lX|%lu", 
          cast<ConstantExpr>(getTag(addr))->getZExtValue(),
          cast<ConstantExpr>(getSet(addr))->getZExtValue());
      llvm::errs().flush();
    } else {
      fprintf(stderr, "[%3u] Symbolic memory address::=>", cnt);
      fprintf(stderr, "[skip]");
      /*
#ifdef Lewis_DEBUG_CACHE
      ExprSMTLIBPrinter printer;
      printer.setOutput(llvm::errs());
      printer.printMemoryExpression(addr);
#else 
      fprintf(stderr, "[skip]");
#endif
      */
    }

    bool isNormal = true;
    for (std::vector<std::pair<uint64_t, uint64_t>>::iterator JJ = state.marks.begin(); JJ != state.marks.end(); JJ++) {
      uint64_t start = JJ->first;
      uint64_t end = JJ->second;
      // [start, end)
      if (start <= cnt && cnt < end) {
        fprintf(stderr, " -- S");
        isNormal = false;
      }
    }
    
    if (isNormal)
      fprintf(stderr, " -- N");

    if (II->second == false) {
      fprintf(stderr, " -- Ignore");
    }

    fprintf(stderr, " -- %s:%u", state.pcInfos[cnt]->file.c_str(), state.pcInfos[cnt]->line);

    llvm::errs() << "\n";
  }

  fprintf(stderr, "===================== end printing all addresses =======================\n");
  fflush(stderr);
}

/*********************** End Debug Util *****************************/
