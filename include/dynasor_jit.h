// ============================================================================
//  dynasor_jit.h -- opt-in runtime specialization of the hot MTTKRP kernel.
//
//  Stage-3 optimization.  The baked binary already specializes
//  `process_range_fiber_csr_T<N, Rp, TGT>` across the grid
//      N in {3,4,5} x Rp in {16,32,64,128,256} x TGT in [0, N-1]
//  so a tensor that hits the grid runs a fully-unrolled, type-correct
//  kernel with no dispatch-time branches.
//
//  What the JIT adds on top:
//    1. Rare-grid shapes (N=2, N in {6,7,8}; Rp in {48,96,192,384}) -- the
//       baked binary falls through to the generic runtime-bound variant
//       (N=0, Rp=0, TGT=-1), which has un-unrolled inner loops.  The JIT
//       recompiles with real constants and closes the 3-5x gap.
//    2. Constant-propagates the L2 far-prefetch switch: `pf_far_on` is a
//       LICM-hoisted runtime load in the baked binary, which keeps the
//       dead T2-prefetch code path resident.  The JIT bakes it as
//       `if constexpr (pf_far)` so the entire block is DCE'd when off.
//    3. Tensor-specific DYN_PF_DIST / DYN_PF_DIST_FAR tuning (future
//       hook; currently the defaults are baked unchanged).
//
//  The JIT is OPT-IN (--jit or DYN_JIT=1) and compiles lazily -- one
//  shell-out to $(CXX) per unique (N, Rp, TGT, pf_far) tuple used by the
//  run.  Typical compile time is 2-4 seconds per variant, so the payoff
//  horizon is ~20+ CP-ALS iterations.  Failure to compile or load the
//  shared library degrades gracefully to the baked template dispatch
//  with a single-line notice.
//
//  Output is cached under .dyn_jit_cache/<key>.dll (Windows) or
//  .dyn_jit_cache/<key>.so (POSIX).  The cache key mixes
//    * ISA token (matches DYN_SIMD_NAME)
//    * N, Rp, TGT, pf_far
//    * $(CXX) --version sha1 prefix
//  so toolchain changes invalidate the cache automatically.
// ============================================================================
#ifndef DYNASOR_JIT_H
#define DYNASOR_JIT_H

#include "dynasor_common.h"

namespace dynasor {

// ---------------------------------------------------------------------------
//  JIT-compiled fiber_csr kernel.  Matches the signature of the baked
//  dynasor_process_shard_fiber_csr but with N, Rp, TGT, pf_far baked
//  into the function body -- so these parameters are no longer accepted
//  as arguments.  The driver selects the correct function pointer via
//  dyn_jit_get_fiber_csr() before entering the parallel region.
// ---------------------------------------------------------------------------
using jit_fiber_csr_fn = void (*)(
    const value_t*         vals,
    const idx_t* const*    idx,
    const uint64_t*        rowptr,
    idx_t                  row_b,
    idx_t                  row_e,
    const value_t* const*  Y,
    value_t*               Yhat_n);

// Master switch.  Checks DYN_JIT env var; can be forced on/off via
// dyn_jit_force_enable().  The first call is memoized.
bool dyn_jit_enabled();

// Force-enable / force-disable JIT for the remainder of the run.  Called
// by main.cpp when --jit is present so users don't also have to set
// DYN_JIT=1.  Must be called before the first dyn_jit_get_* query.
void dyn_jit_force_enable(bool on);

// Resolve (or compile + load) a fiber_csr specialization.  Returns
// nullptr if the JIT is disabled, the toolchain is missing, or the
// compile/dlopen failed.  On nullptr the driver should fall back to the
// baked template path.  Thread-safe (serialized internally on first
// miss; cache hits are lock-free).
jit_fiber_csr_fn dyn_jit_get_fiber_csr(int num_modes,
                                       int rank_padded,
                                       int target_mode,
                                       bool pf_far);

// ---------------------------------------------------------------------------
//  Stage 4 -- JIT-compiled all-modes Morton kernel.
//
//  Signature matches process_range_all_modes_morton_T<N, Rp>:
//    (const value_t* mvals,
//     const uint64_t* mkeys,
//     uint64_t b, uint64_t e,
//     const value_t* const* Y,
//     value_t* const* Yout)
//
//  The baked kernel in dynasor_morton.cpp reads per-mode masks from a
//  runtime `uint64_t mask[N]` array.  The JIT variant bakes the N masks
//  as `static constexpr uint64_t` so `_pext_u64(key, kMask0)` becomes a
//  single-cycle immediate-mask instruction -- no stack load, no cache
//  miss on mask array.
//
//  Cache key: (N, Rp, mask_0, mask_1, ..., mask_{N-1}, ISA, cxx_tag).
// ---------------------------------------------------------------------------
using jit_all_modes_morton_fn = void (*)(
    const value_t*         mvals,
    const uint64_t*        mkeys,
    uint64_t b, uint64_t e,
    const value_t* const*  Y,
    value_t* const*        Yout);

// Resolve (or compile + load) an all-modes Morton specialization.  Returns
// nullptr if the JIT is disabled, the toolchain is missing, or the
// compile/dlopen failed.  On nullptr the driver should fall back to the
// baked templated path.  Thread-safe; first miss serialized on the JIT
// mutex, subsequent hits are lock-free through the cache.
jit_all_modes_morton_fn dyn_jit_get_all_modes_morton(
    int             num_modes,
    int             rank_padded,
    const uint64_t* masks);

// Diagnostic: compiled specializations so far (hits + misses).  Used by
// the SpmttkrpSweep banner.
struct JitStats {
    int compiled   = 0;  // how many unique variants the JIT built
    int cache_hits = 0;  // cached-on-disk variants reused this run
    int failures   = 0;  // compile/load failures (silent fallbacks)
};
JitStats dyn_jit_stats();

} // namespace dynasor

#endif // DYNASOR_JIT_H
