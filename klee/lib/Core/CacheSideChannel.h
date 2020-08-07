#ifndef __CACHE_SIDE_CHANNEL_H_
#define __CACHE_SIDE_CHANNEL_H_

#include "Executor.h"
#include "Context.h"
#include "CoreStats.h"
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
#include "ExecutorTimerInfo.h"

#include "klee/SolverStats.h"

#include "klee/ExecutionState.h"
#include "klee/Expr.h"
#include "klee/Interpreter.h"
#include "klee/TimerStatIncrementer.h"
#include "llvm/Support/CommandLine.h"
#include "klee/Common.h"
#include "klee/util/Assignment.h"
#include "klee/util/ExprPPrinter.h"
#include "klee/util/ExprSMTLIBPrinter.h"
#include "klee/util/ExprUtil.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "klee/Config/Version.h"
#include "klee/Internal/ADT/KTest.h"
#include "klee/Internal/ADT/RNG.h"
#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "klee/Internal/Support/FloatEvaluation.h"
#include "klee/Internal/System/Time.h"
#include "klee/Internal/System/MemoryUsage.h"

#define BSIZE 32
#define NSETS 1024

using namespace klee;
typedef std::vector< ref<Expr> > addrT;
typedef std::pair< ref<Expr>, ref<Expr> > pairT;
typedef std::map<unsigned, pairT> logT;
typedef std::map< pairT, ref<Expr> > hashT;
typedef std::map< ref<Expr>, ref<Expr> > cacheT;
typedef std::map< unsigned, addrT > seqT;
typedef std::map< unsigned, std::vector<pairT> > seqPairT;

class CacheChannel {
  private:
    /* optimization tricks to reduce memory consumption during constraint building */
    static hashT setPairCache;
    static cacheT setCache;
    static hashT tagPairCache;
    static cacheT tagCache;


    /* access-based observer models */
    static void processDirMapAccessCnstr(ExecutionState& state, TimingSolver* solver);
    static void processSetAssocAccessCnstr(ExecutionState& state, TimingSolver* solver);

    /* generate reload constraints  */
    static ref<Expr> generateCnstrForInterReload(ExecutionState& state, int hStatr, int hEnd, int target, bool& reload);

    /* encodes misses */
    static void processColdMissCnstr(ExecutionState& state, TimingSolver* solver);
    static void processConflictMissCnstr(ExecutionState& state, TimingSolver* solver);

    /* log constraints related to only set-associative caches */
    static void logSetAssociativeCnstr(ref<Expr> constraint, unsigned accessID, bool reload = false);

		/* construct tag/set constraints 
		 * suffix "CC" stands for <constant, constant> address pairs
		 * suffix "SC" stands for <symbolic, constant> or <constant, symbolic> address pairs
		 * suffix "SS" stands for <symbolic, symbolic> address pairs */ 
    static bool generateSetCnstrCC(ref<Expr>& addressI, ref<Expr>& addressJ);
    static bool generateTagCnstrCC(ref<Expr>& addressI, ref<Expr>& addressJ);
    static ref<Expr> generateSetCnstrSC(ref<Expr>& addressI, ref<Expr>& addressJ);
    static ref<Expr> generateTagCnstrSC(ref<Expr>& addressI, ref<Expr>& addressJ);
    static ref<Expr> generateSetCnstrSS(ref<Expr>& addressI, ref<Expr>& addressJ);
    static ref<Expr> generateTagCnstrSS(ref<Expr>& addressI, ref<Expr>& addressJ);


    /* return log of a number to the base 2 */
    static int log_base2(int n) {
      int power = 0;
      if (n <= 0 || (n & (n-1)) != 0)
        assert(0 && "log2() only works for positive power of two values");
      while (n >>= 1)
        power++;
      return power;
    }

  public:
    /* cache configuration */
    static int nset;
    static int line;
    static int nassoc;
    static char policy[8];
    static int replace;
    static unsigned long observedCacheMiss;

    /* fixed number of cold misses (independent of input) */
    static int fixedColdMisses;
    /* fixed number of conflict misses (independent of input) */
    static unsigned long fixedConflictMisses;
    /* number of atomic constraints in the constraint system */
    static unsigned long atomicCnstr;

    /* store all generated constraints in the constraint log */
    static logT missConstraintLog;

    /* data structures related to set-associative caches */
    static std::map<unsigned, ref<Expr> > setAssocReloadCnstrLog;
    static seqT setAssocCnstrLog;
    static seqPairT setAssocConflictCnstrLog;




  public:
    /* processing memory address */
    static void printMemoryAddr(ExecutionState& state);
    static void processMemoryAddr(ExecutionState& state, TimingSolver* solver);
    static void printCacheCnstr(ExecutionState& state);
};


class CacheChannelLRU : public CacheChannel {
  public:
  static void processLRUConflictMissCnstr(ExecutionState& state, TimingSolver* solver);
};

class CacheChannelFIFO : public CacheChannel {
  public:
  static void processFIFOConflictMissCnstr(unsigned accessID, ExecutionState& state, TimingSolver* solver);
};

#endif // __CACHE_SIDE_CHANNEL_H_
