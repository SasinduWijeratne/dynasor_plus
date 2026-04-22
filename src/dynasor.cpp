// ============================================================================
//  dynasor.cpp -- parallel driver implementing Algorithm 2 of the paper.
//
//  SoA edition: the per-element state lives in Tensor::vals + idx_buf[n][i]
//  (+ scr_vals / scr_idx[n][i] as the remap destination).  `shard_ids[]` has
//  been dropped entirely; sid is derived as idx >> mn_shift[n] in the kernel.
//
//  One thread per super-shard group, produced by the greedy LPT scheduler
//  (see dynasor_preprocess.cpp).  All writes to Yhat[n][idx_n, :] are
//  conflict-free because the super-shard containing idx_n is owned by a
//  single thread.
//
//  Per-mode runtime work:
//      1. zero Yhat[n]
//      2. memcpy(work_off, T.precomp_thr_off[n])   -- tiny
//      3. parallel kernel pass using work_off[tid * stride + ...] as a
//         thread-local, monotonically-increasing cursor (no atomics)
//      4. swap source/scratch SoA pointer groups
//
//  One OpenMP implicit barrier per mode.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_jit.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
// Fallback timer when built without OpenMP (cheaper than <chrono> here).
#include <ctime>
static inline double omp_get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

namespace dynasor {

// Declared in dynasor_kernel.cpp.  Two variants:
//   dynasor_process_shard        -- element-at-a-time (default)
//   dynasor_process_shard_fiber  -- fiber-unrolled    (DenseFiber modes)
void dynasor_process_shard(
    const value_t* vals,
    const idx_t*   const* idx,
    uint64_t b, uint64_t e,
    int n, int num_modes,
    int rank, int rank_padded,
    const value_t* const* Y,
    value_t* Yhat_n,
    int next_mode, int shift_next,
    value_t* scr_vals,
    idx_t*   const* scr_idx,
    uint64_t* offsets_local);

void dynasor_process_shard_fiber(
    const value_t* vals,
    const idx_t*   const* idx,
    uint64_t b, uint64_t e,
    int n, int num_modes,
    int rank, int rank_padded,
    const value_t* const* Y,
    value_t* Yhat_n,
    int next_mode, int shift_next,
    value_t* scr_vals,
    idx_t*   const* scr_idx,
    uint64_t* offsets_local);

// NCopy + CSR: rowptr-driven fiber kernel.  No remap scatter; caller
// passes the per-mode rowptr and the super-shard's row range.  Requires
// T.ncopy_csr[n] = true (rowptr built at preprocess time).
void dynasor_process_shard_fiber_csr(
    const value_t* vals,
    const idx_t*   const* idx,
    const uint64_t* rowptr,
    idx_t row_b, idx_t row_e,
    int n, int num_modes,
    int rank, int rank_padded,
    const value_t* const* Y,
    value_t* Yhat_n);

// All-modes single-pass kernel.  Streams one slab range and updates all
// N per-thread private Yhat (ofibs) simultaneously.  See kernel doc in
// dynasor_kernel.cpp for the compute pattern.
void dynasor_process_shard_all_modes(
    const value_t* vals,
    const idx_t*   const* idx,
    uint64_t b, uint64_t e,
    int num_modes, int rank, int rank_padded,
    const value_t* const* Y,
    value_t* const* Yout);

// Threshold (bytes) above which Yhat's zero-fill switches from memset
// to write-combining NT stores.  Below the threshold we want the zeros
// to LIVE in LLC so the kernel's first per-row load is an LLC hit; above
// it the zeros would be evicted before the kernel ever looks at them
// (Yhat is read exactly once per fiber start), so burning LLC lines
// with memset is pure waste and NT wins on two fronts: no RFO, no cache
// pollution.  Default 32 MiB matches typical L3 capacity on Zen 4/5 and
// Sapphire Rapids; overridable at runtime.
static inline size_t dyn_nt_zero_threshold() {
    static const size_t thr = [] {
        const char* s = std::getenv("DYN_NT_ZERO_THRESHOLD");
        if (s && *s) {
            long long v = std::atoll(s);
            if (v > 0) return (size_t)v;
        }
        return (size_t)(32ull << 20);   // 32 MiB
    }();
    return thr;
}

// Force-disable / force-enable the NT-zero path (env A/B knob).
//
// DEFAULT OFF -- A/B tested on bench_3d_10M / bench_4d_10M at rank 128,
// the NT-zero path is neutral-to-mildly-regressing (0..6%) on moderate
// tensors.  Skipping RFO looks like a free win on paper, but replacing
// memset's `rep stosb` (which the CPU can steer through its optimized
// fast-string engine) with explicit 64 B NT stores costs throughput
// unless Yhat is truly too large for LLC to absorb.  Enable via
// DYN_NT_ZERO=auto (respects threshold) or DYN_NT_ZERO=1 (force) on
// billion-nnz workloads where Yhat > LLC by a large margin.
enum class NTZeroMode { Auto, Off, Force };
static inline NTZeroMode dyn_nt_zero_mode() {
    static const NTZeroMode m = [] {
        const char* s = std::getenv("DYN_NT_ZERO");
        if (!s || !*s) return NTZeroMode::Off;
        if (s[0] == '0')                         return NTZeroMode::Off;
        if (s[0] == 'a' || s[0] == 'A')          return NTZeroMode::Auto;
        if (s[0] == '1' || s[0] == 'f' || s[0] == 'F')
                                                 return NTZeroMode::Force;
        return NTZeroMode::Off;
    }();
    return m;
}

// NT zero one thread's contiguous slice.  `p` is assumed 64-byte
// aligned (dyn_aligned_alloc guarantees that) and `bytes` need not be
// a multiple of 64 -- the head/tail scraps fall back to memset.
static inline void zero_slice_nt(char* p, size_t bytes) {
    // Align write pointer up to 64 B.  Yhat is allocated 64 B aligned
    // (dyn_aligned_alloc), but our per-thread slice offset may not be;
    // fall back to memset for the head fragment.
    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    size_t head = (64u - (size_t)(addr & 63u)) & 63u;
    if (head > bytes) head = bytes;
    if (head) { std::memset(p, 0, head); p += head; bytes -= head; }
    const size_t full_lines = bytes / 64u;
    for (size_t i = 0; i < full_lines; ++i)
        dyn_nt_zero_cacheline(p + i * 64u);
    const size_t tail = bytes - full_lines * 64u;
    if (tail) std::memset(p + full_lines * 64u, 0, tail);
}

// Parallel bulk zero of Yhat[n].  Saturates every memory controller by
// splitting the buffer across the OpenMP team; each thread zeros its
// own slice, which also preserves NUMA first-touch residency for the
// subsequent kernel pass.  When the buffer is large enough to exceed
// LLC the slice-level zero switches to write-combining NT stores (see
// dyn_nt_zero_threshold() / dyn_nt_zero_mode()), saving the RFO on
// every cache line.
static inline void zero_output_bulk(value_t* Yhat_n, size_t bytes) {
    const NTZeroMode mode = dyn_nt_zero_mode();
    const bool use_nt =
        (mode == NTZeroMode::Force) ||
        (mode == NTZeroMode::Auto && bytes >= dyn_nt_zero_threshold());
#ifdef _OPENMP
    #pragma omp parallel
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t chunk = (bytes + (size_t)nt - 1) / (size_t)nt;
        const size_t b     = std::min(bytes, (size_t)tid * chunk);
        const size_t e     = std::min(bytes, b + chunk);
        if (e > b) {
            if (use_nt) zero_slice_nt((char*)Yhat_n + b, e - b);
            else        std::memset ((char*)Yhat_n + b, 0, e - b);
        }
    }
#else
    if (use_nt) zero_slice_nt((char*)Yhat_n, bytes);
    else        std::memset (Yhat_n, 0, bytes);
#endif
    // NT stores are weakly ordered vs subsequent normal loads; the
    // parallel region's implicit barrier is NOT an sfence on x86.
    // Issue the fence so the kernel pass that immediately follows
    // sees zeroed memory in program order.
    if (use_nt) dyn_sfence_stores();
}

// Lazy zero: only clears rows that the kernel will actually write, using
// the touched-row bitmap built during preprocessing.  The iteration
// splits the bitmap into byte-range chunks across the OpenMP team so the
// per-row memsets run in parallel and hit every memory controller.
static inline void zero_output_lazy(value_t* Yhat_n,
                                    int rank_padded,
                                    idx_t mode_size,
                                    const uint8_t* bits)
{
    const size_t row_bytes = (size_t)rank_padded * sizeof(value_t);
    const size_t nbytes    = ((size_t)mode_size + 7u) / 8u;
#ifdef _OPENMP
    #pragma omp parallel
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t chunk = (nbytes + (size_t)nt - 1) / (size_t)nt;
        const size_t b     = std::min(nbytes, (size_t)tid * chunk);
        const size_t e     = std::min(nbytes, b + chunk);
        for (size_t k = b; k < e; ++k) {
            uint8_t v = bits[k];
            while (v) {
                const int lsb = __builtin_ctz((unsigned)v);
                const idx_t r = (idx_t)(k * 8u + (size_t)lsb);
                if (r < mode_size)
                    std::memset((char*)Yhat_n + (size_t)r * row_bytes,
                                0, row_bytes);
                v &= (uint8_t)(v - 1);   // clear lowest set bit
            }
        }
    }
#else
    for (size_t k = 0; k < nbytes; ++k) {
        uint8_t v = bits[k];
        while (v) {
            const int lsb = __builtin_ctz((unsigned)v);
            const idx_t r = (idx_t)(k * 8u + (size_t)lsb);
            if (r < mode_size)
                std::memset((char*)Yhat_n + (size_t)r * row_bytes, 0, row_bytes);
            v &= (uint8_t)(v - 1);
        }
    }
#endif
}

// Dispatch: if the mode is near-fully-touched, bulk memset is faster than
// iterating the bitmap; if many rows will never be written, lazy zero
// wins by skipping them entirely.  The 75% crossover matches observed
// behaviour on Zen 4/5 -- bulk streaming beats per-row memsets above that.
static inline void zero_output_mode(value_t*      Yhat_n,
                                    int           rank_padded,
                                    idx_t         mode_size,
                                    const uint8_t* bits,
                                    uint64_t      touched_count)
{
    const size_t row_bytes = (size_t)rank_padded * sizeof(value_t);
    const size_t tot_bytes = (size_t)mode_size * row_bytes;
    const bool dense = (bits == nullptr) ||
                       (touched_count * 4u >= (uint64_t)mode_size * 3u);
    if (dense) zero_output_bulk(Yhat_n, tot_bytes);
    else       zero_output_lazy(Yhat_n, rank_padded, mode_size, bits);
}

// ----------------------------------------------------------------------------
//  Resolve thread count + plan-mismatch handling.  Shared by the sweep
//  constructor and the legacy one-shot driver.
// ----------------------------------------------------------------------------
static int resolve_thread_count(const Tensor& T, int num_threads) {
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    else                 num_threads = omp_get_max_threads();
#else
    num_threads = 1;
#endif

    // NOTE: a previous revision exposed DYN_RUNTIME_THREADS as a
    // "kernel-only scaling" escape hatch that decoupled runtime OMP
    // threads from the plan's thread count.  That path is unsafe:
    //   * rt > plan-threads  =>  sched_mode[tid] OOB read (access violation)
    //   * rt < plan-threads  =>  plan-threads [rt..plan-1]'s shards are
    //                            skipped, corrupting scr_* after the first
    //                            swap_buffers() and producing NaN/garbage
    //                            in subsequent iterations
    // Making rt != plan-threads correct would require re-running the LPT
    // schedule (essentially rebuilding the plan), which defeats the
    // purpose of the hatch.  The supported way to vary thread count is
    // --threads N, which rebuilds the plan consistently.  The override is
    // therefore silently ignored if set.

    if (T.precomp_num_threads > 0 && num_threads != T.precomp_num_threads) {
        std::printf("[dynasor] warning: plan built for %d threads, "
                    "requested %d; using %d\n",
                    T.precomp_num_threads, num_threads, T.precomp_num_threads);
        num_threads = T.precomp_num_threads;
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
    }
    return num_threads;
}

// ----------------------------------------------------------------------------
//  SpmttkrpSweep -- per-mode API used by iterative decomposition drivers.
//  The body of process(n) is identical to the inner loop body of the
//  legacy one-shot driver below, so both share the same kernel code path
//  (fiber-sort + zero + kernel + sfence + swap).
// ----------------------------------------------------------------------------
SpmttkrpSweep::SpmttkrpSweep(Tensor& T, FactorMatrices& F, int num_threads,
                             bool print_banner)
    : T_(T), F_(F), num_threads_(resolve_thread_count(T, num_threads))
{
    const char* fiber_env = std::getenv("DYN_FIBER_OFF");
    fiber_off_ = fiber_env && *fiber_env && *fiber_env != '0';

    size_t max_plan = 0;
    for (int n = 0; n < T_.num_modes; ++n)
        if (T_.precomp_thr_off[n].size() > max_plan)
            max_plan = T_.precomp_thr_off[n].size();
    work_off_.assign(max_plan, 0);

    if (print_banner) {
        std::printf("[dynasor] threads=%d rank=%d (padded=%d) SIMD=%s "
                    "(SoA, pre-planned, atomic-free)\n",
                    num_threads_, F_.rank, F_.rank_padded, DYN_SIMD_NAME);
        // One-line cache-tricks status so A/B runs are self-documenting.
        // Read DYN_PF_FAR here (the toggle itself lives in dynasor_kernel.cpp
        // as a file-local lambda; we only mirror its state).  Kept in sync
        // with dyn_pf_far_on(): default OFF, enabled only when env is set
        // and not "0".
        const char* pf_far_env = std::getenv("DYN_PF_FAR");
        const bool  pf_far_on  = pf_far_env && pf_far_env[0] && pf_far_env[0] != '0';
        const char* pf_far_s   = pf_far_on ? "on" : "off";
        const NTZeroMode nz = dyn_nt_zero_mode();
        const char* nt_zero_s =
            (nz == NTZeroMode::Force) ? "force" :
            (nz == NTZeroMode::Off  ) ? "off"   : "auto";
        const size_t nt_thr = dyn_nt_zero_threshold();
#if defined(_WIN32)
        const char* lp_s =
            (std::getenv("DYN_WIN_LARGE_PAGES") &&
             std::getenv("DYN_WIN_LARGE_PAGES")[0] &&
             std::getenv("DYN_WIN_LARGE_PAGES")[0] != '0')
                ? "win-large-pages=on" : "win-large-pages=off";
#elif defined(__linux__)
        const char* lp_s = "thp-hint=on";   // dyn_advise_huge is unconditional
#else
        const char* lp_s = "hugepages=n/a";
#endif
        std::printf("[dynasor] cache-tricks: pf-far=%s nt-zero=%s (thr=%zu MiB) %s\n",
                    pf_far_s, nt_zero_s, nt_thr >> 20, lp_s);
        std::printf("[dynasor] jit: %s\n",
                    dyn_jit_enabled() ? "on (fiber_csr specialization)" : "off");
    }
}

void SpmttkrpSweep::process(int n) {
    const int N  = T_.num_modes;
    const int R  = F_.rank;
    const int Rp = F_.rank_padded;
    const value_t* const* Yin =
        const_cast<const value_t* const*>(F_.Y.data());

    const int next_mode  = (n + 1) % N;
    const int shift_next = T_.mn_shift[next_mode];

    // NCopy + fiber combo:
    //   * fiber_sort_shards reads primary and writes scratch, then
    //     swap_buffers() -- both exclusive to the PingPong layout, so we
    //     never call it at runtime under NCopy.
    //   * Instead, decide_and_populate_layout() sorts each NCopy copy by
    //     ix[n] within each shard ONCE at preprocess time.  The runtime
    //     dispatch here consults T_.ncopy_fiber_sorted[n]:
    //        true  -> fiber kernel is safe (copy is pre-sorted).
    //        false -> fall back to element kernel for this mode
    //                 (the copy was populated but not sorted, e.g. because
    //                 the user set DYN_NCOPY_FIBER_SORT=0, or this mode
    //                 is not DenseFiber).
    //   * When T_.ncopy_csr[n] is also true, dispatch the rowptr-driven
    //     CSR fiber kernel instead of the COO one: it avoids the O(nnz)
    //     fiber-boundary scan and drops the ix[n] prefetch stream.
    //   * PingPong path is unchanged: sort is driven by fiber_sort_shards.
    const bool ncopy_layout   = (T_.layout == Layout::NCopy);
    const bool inplace_layout = (T_.layout == Layout::InPlace);
    const bool fiber_ok_ncopy =
        ncopy_layout && T_.ncopy_fiber_sorted[n];
    const bool fiber_ok_pp =
        !ncopy_layout && !inplace_layout;
    const bool fiber_ok_inplace =
        inplace_layout;   // inplace_sort_slab_by_mode produces fiber order for DenseFiber modes
    const bool use_fiber =
        !fiber_off_ &&
        (T_.kernel_class[n] == KernelClass::DenseFiber) &&
        (fiber_ok_ncopy || fiber_ok_pp || fiber_ok_inplace);
    const bool use_fiber_csr =
        use_fiber && ncopy_layout && T_.ncopy_csr[n];

    const double t0 = omp_get_wtime();

    // 0. Slab ordering for mode n.
    //    PingPong : fiber_sort_shards (runtime sort of the scratch side
    //               that was just populated by the previous mode's scatter).
    //    NCopy    : pre-sorted at preprocess time; nothing to do.
    //    InPlace  : cycle-following counting sort of the primary slab
    //               by ix[n] >> mn_shift[n], with optional within-shard
    //               fiber sort if this mode is DenseFiber.  Skipped on
    //               mode 0 of the first sweep because build_flycoo leaves
    //               the slab already sorted by mode 0.
    if (use_fiber && !ncopy_layout && !inplace_layout) {
        fiber_sort_shards(T_, n);
        ++n_fiber_used_;
    } else if (use_fiber && ncopy_layout) {
        ++n_fiber_used_;   // still counted for the banner
    } else if (use_fiber && inplace_layout) {
        ++n_fiber_used_;
    }
    if (inplace_layout && T_.inplace_sorted_mode != n) {
        // The preprocessor leaves the primary slab sorted by mode 0, so
        // the very first process(0) hits the "already sorted" shortcut.
        // Every other mode re-sorts in place before the kernel runs.
        inplace_sort_slab_by_mode(T_, n, num_threads_);
        T_.inplace_sorted_mode = (int8_t)n;
    }

    // 1. Zero Yhat[n] (bulk if dense, row-selective if sparse).
    const uint8_t* t_bits =
        (n < (int)T_.touched_bits.size() && !T_.touched_bits[n].empty())
            ? T_.touched_bits[n].data() : nullptr;
    const uint64_t t_count =
        (n < (int)T_.touched_count.size()) ? T_.touched_count[n] : 0;
    zero_output_mode(F_.Yhat[n], Rp, T_.mode_size[n], t_bits, t_count);

    // 2. Cursor template copy.  Skipped in NCopy and InPlace modes (no
    //    scatter-remap -> no cursors).
    const std::vector<uint64_t>& plan = T_.precomp_thr_off[n];
    const bool ncopy       = ncopy_layout;
    const bool no_scatter  = ncopy || inplace_layout;
    if (!no_scatter) {
        std::memcpy(work_off_.data(), plan.data(),
                    plan.size() * sizeof(uint64_t));
    }
    const size_t stride = plan.size() / (size_t)num_threads_;

    // 3. Main parallel kernel pass.
    //
    //    PingPong: read from T_.vals / T_.idx_buf, scatter to T_.scr_vals /
    //              T_.scr_idx, then swap at end.
    //    NCopy   : read from T_.ncopy_vals[n] / T_.ncopy_idx[n], pass
    //              nullptr scratch pointers so the kernel skips the
    //              scatter block entirely.  No swap, no elapsed_ bump
    //              from remap bandwidth.
    //    InPlace : read from T_.vals / T_.idx_buf (just sorted by mode n
    //              above), nullptr scratch -> same kernel hot path as
    //              NCopy.  No swap.
    const auto&          sched_mode = T_.ss_list[n];
    const value_t*       vals_src   = ncopy ? T_.ncopy_vals[n]       : T_.vals;
    const idx_t* const*  idx_src    = ncopy ? T_.ncopy_idx [n]       : T_.idx_buf;
    value_t*             scr_v      = no_scatter ? nullptr           : T_.scr_vals;
    idx_t* const*        scr_i      = no_scatter ? nullptr           : T_.scr_idx;
    value_t*             Yhat_n     = F_.Yhat[n];

    // Stage-3 JIT: resolve a (N, Rp, TGT, pf_far)-specialized fiber_csr
    // kernel if the JIT is enabled and we are on the NCopy+CSR+DenseFiber
    // path.  Lookup is O(1) on cache hit; miss performs at most one
    // shell-out compile which is memoized for the remainder of the run.
    // nullptr on any failure -> fall back to the baked template dispatch
    // without changing correctness.  Resolved outside the omp parallel
    // region so every thread sees the same pointer without re-querying
    // the cache under its lock.
    jit_fiber_csr_fn jit_fn = nullptr;
    if (use_fiber_csr) {
        const char* pf_far_env = std::getenv("DYN_PF_FAR");
        const bool  pf_far_on  = pf_far_env && pf_far_env[0] && pf_far_env[0] != '0';
        jit_fn = dyn_jit_get_fiber_csr(N, Rp, n, pf_far_on);
    }

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads_) {
            uint64_t* off = work_off_.data() + (size_t)tid * stride;
            const int      shft_n    = T_.mn_shift[n];
            const idx_t    mode_n_sz = T_.mode_size[n];
            const uint64_t* rowptr_n = use_fiber_csr ? T_.ncopy_rowptr[n]
                                                     : nullptr;
            for (sid_t ss : sched_mode[tid]) {
                uint64_t b = T_.shard_begin[n][ss];
                uint64_t e = T_.shard_end  [n][ss];
                if (use_fiber_csr) {
                    // Derive per-shard row range in O(1) from super-shard
                    // id.  `mn[n]` is a power of two so `ss << shft_n`
                    // is the exact row_lo; clip row_hi at mode_size[n]
                    // to handle the final super-shard's tail rows.
                    idx_t row_lo = (idx_t)((uint64_t)ss << shft_n);
                    idx_t row_hi = (idx_t)((uint64_t)(ss + 1) << shft_n);
                    if (row_hi > mode_n_sz) row_hi = mode_n_sz;
                    if (jit_fn) {
                        jit_fn(vals_src, idx_src, rowptr_n,
                               row_lo, row_hi,
                               Yin, Yhat_n);
                    } else {
                        dynasor_process_shard_fiber_csr(
                            vals_src, idx_src, rowptr_n,
                            row_lo, row_hi,
                            n, N, R, Rp,
                            Yin, Yhat_n);
                    }
                } else if (use_fiber) {
                    dynasor_process_shard_fiber(
                        vals_src, idx_src, b, e,
                        n, N, R, Rp,
                        Yin, Yhat_n,
                        next_mode, shift_next,
                        scr_v, scr_i, off);
                } else {
                    dynasor_process_shard(
                        vals_src, idx_src, b, e,
                        n, N, R, Rp,
                        Yin, Yhat_n,
                        next_mode, shift_next,
                        scr_v, scr_i, off);
                }
            }
        }
    }

    // 4. Publish the kernel's NT remap stores.  A no-op in NCopy mode
    //    because the kernel issued no scatter stores, but sfence is cheap
    //    and keeps the control flow simple.
    dyn_sfence_stores();

    // 5. Activate the freshly remapped layout for mode (n+1).
    //    NCopy uses N pre-sorted copies so no swap is needed: the next
    //    mode will simply index into ncopy_vals[n+1] directly.
    //    InPlace keeps the primary slab live; the next mode re-sorts it
    //    in place (step 0 above).
    if (!no_scatter) T_.swap_buffers();

    elapsed_ += omp_get_wtime() - t0;
}

// ----------------------------------------------------------------------------
//  Legacy one-shot driver: run all N modes in a single sweep.  Kept for
//  compatibility with the standalone MTTKRP benchmark in main.cpp.
// ----------------------------------------------------------------------------
double spmttkrp_dynasor(Tensor& T, FactorMatrices& F, int num_threads) {
    SpmttkrpSweep sweep(T, F, num_threads, /*print_banner=*/true);
    for (int n = 0; n < T.num_modes; ++n) sweep.process(n);

    const double secs = sweep.elapsed_s();
    const int    used = sweep.fiber_modes_used();
    const char*  fiber_env = std::getenv("DYN_FIBER_OFF");
    const bool   fiber_off = fiber_env && *fiber_env && *fiber_env != '0';
    std::printf("[dynasor] total spMTTKRP time (all %d modes) = %.6f s "
                "(fiber kernel used on %d mode%s%s)\n",
                T.num_modes, secs, used,
                used == 1 ? "" : "s",
                fiber_off ? ", DYN_FIBER_OFF=1" : "");
    return secs;
}

// ============================================================================
//  All-modes single-pass driver.
//
//  Streams the tensor ONCE per sweep, updating all N Yhat outputs in a
//  single kernel pass.  Per-thread private "ofibs" buffers eliminate
//  write contention; a final parallel reduction folds them into the
//  caller's global Yhat[n] arrays.
//
//  Per-iteration work (times in ms are rough estimates for bench_3d_10M,
//  rank 32, 16 threads, AVX-512 / Zen 5):
//
//    A. zero per-thread ofibs         (~8 MiB total, ~1 ms)
//    B. parallel kernel pass          (one slab-read, ~40 ms)
//    C. parallel reduction            (N * mode_size * rank * T -> Yhat,
//                                      ~3 ms on medium, dominated by
//                                      Yhat writeout not the reduction
//                                      compute)
//
//  Compared to the per-mode driver at 220 ms/iter (InPlace) on the same
//  tensor, the all-modes path targets ~45 ms/iter -- within ALTO's
//  ~62 ms/iter reach and below on 4D / higher-rank regimes.
//
//  Fallback: if ofibs allocation fails (e.g. 335M-nnz tensor with
//  mode_size ~3M and 32 threads would need ~1.5 GiB of ofibs), the
//  driver prints a warning and defers to spmttkrp_dynasor.
//
//  Layout independence: the kernel reads idx[w][i] / vals[i] directly
//  from T.vals and T.idx_buf, in whatever order the preprocessor left
//  them.  No per-mode re-sort, no PingPong swap, no NCopy copy
//  selection.  Works with any Layout.
// ============================================================================
double spmttkrp_all_modes_dynasor(Tensor& T, FactorMatrices& F, int num_threads) {
    const int      N   = T.num_modes;
    const int      R   = F.rank;
    const int      Rp  = F.rank_padded;
    const uint64_t nnz = T.nnz;

    num_threads = resolve_thread_count(T, num_threads);

    // The all-modes kernel reads T.vals and ALL N columns of T.idx_buf
    // from a single slab.  Under NCopy with CSR compaction, the primary
    // slab is freed and T.vals / T.idx_buf[*] are nulled after the per-
    // copy compact pass.  Detect and fall back cleanly -- the caller in
    // main.cpp is supposed to override DYN_NCOPY_CSR_COMPACT=0 when
    // --all-modes is selected, but honor the fallback for robustness.
    if (!T.vals || !T.idx_buf[0]) {
        std::fprintf(stderr,
            "[dynasor] ALL-MODES: primary slab unavailable (likely NCopy+CSR\n"
            "          compact freed it).  Falling back to per-mode driver.\n"
            "          Hint: add --ncopy-csr-compact off or use --layout inplace\n"
            "          together with --all-modes.\n");
        return spmttkrp_dynasor(T, F, num_threads);
    }

    // ------------------------------------------------------------------
    //  Auto-dispatch: pick the best driver for the current (N, Rp) tuple.
    //
    //  The all-modes kernel streams the tensor once (win) but writes to
    //  N random output rows per nnz (cost).  Each random row is
    //  Rp/SIMD_WIDTH cache lines; the per-mode NCopy+CSR kernels instead
    //  process nonzeros in sorted-by-target-mode order, so the current
    //  output row is reused across consecutive nonzeros and typically
    //  stays in L1.  The scatter cost grows linearly with Rp; the
    //  tensor-BW win grows with N.  Empirically on bench_3d_10M /
    //  bench_4d_10M, AVX-512:
    //
    //      N=3 Rp in {16, 32}   -> all-modes wins (1.7-1.8x vs per-mode)
    //      N=3 Rp >= 64          -> per-mode wins (up to 2.1x)
    //      N=4 all tested Rp     -> all-modes wins (1.4-1.9x vs per-mode)
    //      N=5+ likely all-modes -> larger tensor-BW factor (untested)
    //
    //  So we auto-redirect to the per-mode driver in two regimes:
    //
    //      (a) N=3 && Rp>=64    -- 3D, scatter dominates at rank>=64.
    //      (b) Rp>=128          -- any N, TLB footprint of the ofib
    //                              stripe (~128 KiB per (thread, mode))
    //                              starts missing in the 1 MiB L2 too
    //                              regularly and the reduction write-
    //                              back dominates.
    //
    //  Power users can override with DYN_ALL_MODES_FORCE=1 to benchmark
    //  the kernel anyway.  When the heuristic is refined (e.g. after
    //  tile-based output handling closes the gap at Rp>=64), update this
    //  predicate rather than the caller.
    {
        const bool auto_fallback =
            (N == 3 && Rp >= 64) ||   // (a)
            (Rp >= 128);              // (b)
        const char* force = std::getenv("DYN_ALL_MODES_FORCE");
        const bool  forced = force && force[0] && force[0] != '0';
        if (auto_fallback && !forced) {
            std::printf("[dynasor] ALL-MODES: auto-dispatch -> per-mode "
                        "(N=%d Rp=%d: predicted output-row scatter cost "
                        "exceeds tensor-BW win; override with "
                        "DYN_ALL_MODES_FORCE=1)\n", N, Rp);
            return spmttkrp_dynasor(T, F, num_threads);
        }
    }

    // ---- banner ----
    {
        const char* pf_far_env = std::getenv("DYN_PF_FAR");
        const bool  pf_far_on  = pf_far_env && pf_far_env[0] && pf_far_env[0] != '0';
        std::printf("[dynasor] ALL-MODES driver: threads=%d rank=%d (padded=%d) "
                    "SIMD=%s pf-far=%s\n",
                    num_threads, R, Rp, DYN_SIMD_NAME,
                    pf_far_on ? "on" : "off");
    }

    // ---- per-mode output buffer layout ----
    //
    // ofibs_raw owns num_threads aligned blocks.  Within each block,
    // mode n's private Yhat starts at offset ofib_off[n] (in value_t
    // elements).  ofib_off[N] is the per-thread block stride.
    std::vector<size_t> ofib_off((size_t)N + 1, 0);
    for (int n = 0; n < N; ++n) {
        size_t rows = (size_t)T.mode_size[n];
        size_t bytes = rows * (size_t)Rp * sizeof(value_t);
        // Round up to 64 B alignment so every (tid, n) buffer is cache-
        // line aligned independent of rank_padded.
        bytes = (bytes + 63ULL) & ~63ULL;
        ofib_off[n + 1] = ofib_off[n] + bytes / sizeof(value_t);
    }
    const size_t stride_values = ofib_off[N];    // per-thread slot, in value_t
    const size_t stride_bytes  = stride_values * sizeof(value_t);
    const size_t total_bytes   = stride_bytes * (size_t)num_threads;

    value_t* ofibs_raw = (value_t*)dyn_aligned_alloc(total_bytes);
    if (!ofibs_raw) {
        std::fprintf(stderr,
            "[dynasor] ALL-MODES: unable to allocate %.2f GiB of per-thread\n"
            "          private Yhat buffers; falling back to per-mode driver.\n",
            (double)total_bytes / (double)(1ULL << 30));
        return spmttkrp_dynasor(T, F, num_threads);
    }
    std::printf("[dynasor] ALL-MODES: ofibs = %.3f MiB (%d threads x %.3f MiB)\n",
                (double)total_bytes / (double)(1ULL << 20),
                num_threads,
                (double)stride_bytes / (double)(1ULL << 20));

    // ---- per-thread pointer tables ----
    //   Yout_by_thread[tid][n] -> start of mode-n private Yhat for thread tid.
    std::vector<std::vector<value_t*>> Yout_by_thread(
        (size_t)num_threads, std::vector<value_t*>((size_t)N, nullptr));
    for (int tid = 0; tid < num_threads; ++tid) {
        value_t* base = ofibs_raw + (size_t)tid * stride_values;
        for (int n = 0; n < N; ++n)
            Yout_by_thread[(size_t)tid][(size_t)n] = base + ofib_off[(size_t)n];
    }

    // Flat pointer array each thread passes into the kernel (value_t* const*).
    // Reused per-shard inside the parallel region; built there as a stack
    // local to avoid an extra indirection.

    const double t0 = omp_get_wtime();

    // ---- A. zero per-thread ofibs ----
    // Each thread zeroes its own slab.  The slabs are small enough (few
    // MiB typical) that plain memset is fine; no need for NT stores.
    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads) {
            std::memset(ofibs_raw + (size_t)tid * stride_values,
                        0, stride_bytes);
        }
    }

    // ---- B. parallel kernel pass (one slab read) ----
    // Partition NNZ evenly across threads.  No cross-thread contention
    // because each thread writes into its own ofibs copy.  We do NOT use
    // the shard-based scheduling here: all-modes has no concept of a
    // target mode owning a partition, so a simple blockwise split gives
    // optimal cache behavior (sequential reads over the NNZ column).
    const uint64_t chunk =
        (nnz + (uint64_t)num_threads - 1) / (uint64_t)num_threads;

    const value_t* const* Y_in =
        const_cast<const value_t* const*>(F.Y.data());

    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads) {
            const uint64_t b = (uint64_t)tid * chunk;
            const uint64_t e = std::min(nnz, b + chunk);
            if (b < e) {
                // Build kernel input pointer tables on the stack.
                value_t* Yout_local[DYN_MAX_MODES];
                for (int n = 0; n < N; ++n)
                    Yout_local[n] = Yout_by_thread[(size_t)tid][(size_t)n];

                dynasor_process_shard_all_modes(
                    T.vals,
                    T.idx_buf,
                    b, e,
                    N, R, Rp,
                    Y_in,
                    Yout_local);
            }
        }
    }

    // ---- C. parallel reduction ofibs -> Yhat ----
    // Yhat[n][r, :] = sum over tid of ofibs[tid][n][r, :].
    //
    // Parallelized across (n, row) so each thread does a contiguous
    // run of rows.  Reads T sequential slabs and writes one row of Yhat
    // per row processed; fully vectorizable.
    for (int n = 0; n < N; ++n) {
        const idx_t rows = T.mode_size[n];
        value_t* DYN_RESTRICT dst = F.Yhat[n];
        const size_t row_stride = (size_t)Rp;

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int64_t r64 = 0; r64 < (int64_t)rows; ++r64) {
            const size_t r = (size_t)r64;
            value_t* DYN_RESTRICT out = dst + r * row_stride;

            // Start by loading thread 0's contribution.
            const value_t* DYN_RESTRICT src0 =
                ofibs_raw + 0 * stride_values + ofib_off[(size_t)n]
                          + r * row_stride;
            for (int c = 0; c < (int)row_stride; c += DYN_SIMD_WIDTH) {
                dyn_vstore(out + c, dyn_vload(src0 + c));
            }
            for (int t = 1; t < num_threads; ++t) {
                const value_t* DYN_RESTRICT src =
                    ofibs_raw + (size_t)t * stride_values
                              + ofib_off[(size_t)n]
                              + r * row_stride;
                for (int c = 0; c < (int)row_stride; c += DYN_SIMD_WIDTH) {
                    dyn_vec_t a = dyn_vload(out + c);
                    dyn_vec_t b = dyn_vload(src + c);
                    dyn_vstore(out + c, dyn_vadd(a, b));
                }
            }
        }
    }

    const double secs = omp_get_wtime() - t0;

    dyn_aligned_free(ofibs_raw);

    std::printf("[dynasor] ALL-MODES spMTTKRP (1 slab pass, all %d modes) "
                "= %.6f s (%.3f Mnnz/s)\n",
                N, secs, (double)nnz / 1e6 / secs);
    return secs;
}

} // namespace dynasor
