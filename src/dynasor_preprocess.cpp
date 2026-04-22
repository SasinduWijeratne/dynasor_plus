// ============================================================================
//  dynasor_preprocess.cpp
//
//  Transforms an in-memory COO tensor (SoA layout) into the FLYCOO
//  representation (Section III of the paper).  Steps, with the number of
//  full-tensor scans each incurs:
//
//     1. pick power-of-2 m_n per mode  (=> mn_shift[n])                  0
//     2. FUSED sort-and-count-all-modes over the tensor                  2
//        * Pass A: one OMP-parallel scan building per-worker histograms
//          for EVERY mode simultaneously.  Mode 0's portion also drives
//          the scatter base; modes 1..N-1 become the shared count_flat
//          that downstream steps reuse without re-scanning.
//        * Pass B: the scatter into the SoA scratch (then swap_buffers).
//     3. greedy LPT schedule from the already-computed counts             0
//     4. remap-plan histogram pass (tid, sid_next) only                   1
//        * shard_begin/end come from count_flat, no extra scan.
//     --------------------------------------------------------------------
//     TOTAL tensor scans: 3  (was 3 + num_modes before this refactor)
//
//  Shard ids are DERIVED from indices at every access point (via mn_shift),
//  so there is no shard_ids[] array anywhere in memory.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_cpals.h"          // CpKernel enum (for ComputePlan)
#include "dynasor_jit.h"            // dyn_jit_enabled()
#include "dynasor_simd.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#include <ctime>
static inline double omp_get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

namespace dynasor {

static inline int dyn_omp_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static inline int dyn_omp_tid() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Round x up to the next power of two (>=1).
static inline idx_t round_up_pow2(idx_t x) {
    if (x <= 1) return 1;
    idx_t r = 1;
    while (r < x) r <<= 1;
    return r;
}
static inline int log2_pow2(idx_t x) {
    int k = 0;
    while ((idx_t(1) << k) < x) ++k;
    return k;
}

// ---------------------------------------------------------------------------
//  Choose m_n (rows per super-shard) for mode n.  Returns a POWER OF TWO
//  in [16, 16384] (clamped against I_n).  Power-of-two lets the hot loop
//  compute sid = idx >> mn_shift[n] instead of idx / mn[n].
// ---------------------------------------------------------------------------
static idx_t pick_mn(idx_t I_n, int num_threads) {
    constexpr idx_t kMinRows = 16;
    constexpr idx_t kMaxRows = 16384;

    if (I_n <= (idx_t)num_threads) return 1;

    idx_t target_k = (idx_t)num_threads * 4;
    idx_t mn = std::max<idx_t>(1u, I_n / target_k);
    mn = std::max(mn, kMinRows);
    mn = std::min(mn, kMaxRows);
    if (mn > I_n) mn = I_n;
    return round_up_pow2(mn);
}

// ---------------------------------------------------------------------------
//  Fused: sort by mode 0 AND build per-mode shard counts for all modes.
//
//  Single OMP-parallel scan (Pass A) populates thread-local all-mode
//  histograms.  The mode-0 slice drives the shard-major scatter base; the
//  other modes become `count_flat` and are consumed by build_schedule and
//  build_remap_plan without any further full-tensor scans.
//
//  This replaces the old sort_by_shard + N-times-nnz counting loop in
//  build_schedule.  Total element scans drop from 2+N to 2 (for the whole
//  counting/sorting phase).
// ---------------------------------------------------------------------------
static void sort_and_count_all(
    Tensor&                 T,
    std::vector<uint64_t>&  count_flat,
    std::vector<size_t>&    cnt_off)
{
    const int       NM   = T.num_modes;
    const uint64_t  N    = T.nnz;
    const int       OMPT = dyn_omp_threads();

    // Packed per-mode offsets into the flat count array (size N+1).
    cnt_off.assign(NM + 1, 0);
    for (int n = 0; n < NM; ++n)
        cnt_off[n + 1] = cnt_off[n] + T.shards_per_mode[n];
    const size_t cnt_total = cnt_off[NM];

    // Hoist per-mode pointers and shifts into stack arrays so the hot loop
    // sees constants.  DYN_MAX_MODES is typically 8.
    const idx_t* idx_ptr[DYN_MAX_MODES];
    int          shft   [DYN_MAX_MODES];
    for (int n = 0; n < NM; ++n) { idx_ptr[n] = T.idx_buf[n]; shft[n] = T.mn_shift[n]; }

    // One contiguous allocation for per-worker all-mode histograms.  For
    // OMPT=64, cnt_total=10K this is 5 MiB -- trivial compared to tensor.
    std::vector<uint64_t> w_cnt((size_t)OMPT * cnt_total, 0);

    // Per-worker touched-row bitmaps, packed as
    //     [w * bit_total + bit_off[n] + (row >> 3)]
    // Each bit set iff worker w observed a nonzero at (mode n, row).  OR-
    // reduced at the end of this function into T.touched_bits[n].  The
    // extra memory is ceil(sum(I_n)/8) * OMPT bytes; for Amazon (sum(I) ~
    // 10 M) it's ~80 MiB with 64 workers, trivial relative to the tensor.
    std::vector<size_t> bit_off(NM + 1, 0);
    for (int n = 0; n < NM; ++n) {
        const size_t bytes = ((size_t)T.mode_size[n] + 7u) / 8u;
        bit_off[n + 1] = bit_off[n] + bytes;
    }
    const size_t bit_total = bit_off[NM];
    std::vector<uint8_t> w_bits((size_t)OMPT * bit_total, 0);

    // ------------------------------------------------------------------
    //  Pass A: histogram every mode in one element scan, and mark touched
    //  rows in the per-worker bitmap.  Zero extra scans over the tensor.
    // ------------------------------------------------------------------
    #pragma omp parallel
    {
        const int      wid   = dyn_omp_tid();
        const uint64_t chunk = (N + OMPT - 1) / OMPT;
        const uint64_t b     = std::min<uint64_t>((uint64_t)wid * chunk, N);
        const uint64_t e     = std::min<uint64_t>(b + chunk, N);
        uint64_t* myc        = w_cnt .data() + (size_t)wid * cnt_total;
        uint8_t*  myb        = w_bits.data() + (size_t)wid * bit_total;

        for (uint64_t i = b; i < e; ++i) {
            #pragma GCC unroll 8
            for (int n = 0; n < NM; ++n) {
                const idx_t r = idx_ptr[n][i];
                const sid_t s = (sid_t)(r >> shft[n]);
                myc[cnt_off[n] + s]++;
                myb[bit_off[n] + (size_t)(r >> 3)] |= (uint8_t)(1u << (r & 7u));
            }
        }
    }

    // ------------------------------------------------------------------
    //  Compute per-worker base offsets for the mode-0 scatter.
    //  Shard-major: all elements of shard s come before shard s+1.
    //  This prefix scan is O(K0 * OMPT) -- tiny, but we parallelize it
    //  across shards to stay cache-friendly when K0 is large.
    // ------------------------------------------------------------------
    const idx_t K0 = T.shards_per_mode[0];
    std::vector<uint64_t> base((size_t)OMPT * K0, 0);
    {
        // Step 1: shard totals (sum over workers for each shard).
        std::vector<uint64_t> shard_total(K0, 0);
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < (int64_t)K0; ++s) {
            uint64_t sum = 0;
            for (int w = 0; w < OMPT; ++w)
                sum += w_cnt[(size_t)w * cnt_total + (size_t)s];
            shard_total[s] = sum;
        }
        // Step 2: serial exclusive prefix scan over K0 (K0 <= ~16K typically).
        uint64_t cum = 0;
        std::vector<uint64_t> shard_base(K0, 0);
        for (idx_t s = 0; s < K0; ++s) {
            shard_base[s] = cum;
            cum          += shard_total[s];
        }
        // Step 3: parallel over shards, serial over workers (OMPT is small).
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < (int64_t)K0; ++s) {
            uint64_t c = shard_base[s];
            for (int w = 0; w < OMPT; ++w) {
                base[(size_t)w * K0 + (size_t)s] = c;
                c += w_cnt[(size_t)w * cnt_total + (size_t)s];
            }
        }
    }

    // ------------------------------------------------------------------
    //  Pass B: scatter every SoA array using the per-worker base.
    // ------------------------------------------------------------------
    const int     shift0  = shft[0];
    const idx_t*  ix_sort = idx_ptr[0];
    #pragma omp parallel
    {
        const int      wid   = dyn_omp_tid();
        const uint64_t chunk = (N + OMPT - 1) / OMPT;
        const uint64_t b     = std::min<uint64_t>((uint64_t)wid * chunk, N);
        const uint64_t e     = std::min<uint64_t>(b + chunk, N);
        uint64_t* myb        = base.data() + (size_t)wid * K0;

        for (uint64_t i = b; i < e; ++i) {
            const sid_t    s = (sid_t)(ix_sort[i] >> shift0);
            const uint64_t d = myb[s]++;
            // NT scalar store: skip RFO on every slot we touch here --
            // these buffers are read exclusively after swap_buffers() and
            // after the surrounding parallel region's implicit barrier.
            dyn_stream_f32(T.scr_vals + d, T.vals[i]);
            #pragma GCC unroll 8
            for (int w = 0; w < NM; ++w)
                dyn_stream_u32((uint32_t*)&T.scr_idx[w][d],
                               (uint32_t)T.idx_buf[w][i]);
        }
    }
    // OMP barrier already synchronizes the writers; the sfence makes the
    // NT stores visible to the subsequent swap_buffers() consumer.
    dyn_sfence_stores();
    T.swap_buffers();

    // ------------------------------------------------------------------
    //  Reduce per-worker histograms into the shared count_flat.
    //  Parallel over cnt_total -- trivially bandwidth bound.
    // ------------------------------------------------------------------
    count_flat.assign(cnt_total, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < (int64_t)cnt_total; ++k) {
        uint64_t sum = 0;
        for (int w = 0; w < OMPT; ++w)
            sum += w_cnt[(size_t)w * cnt_total + (size_t)k];
        count_flat[k] = sum;
    }

    // ------------------------------------------------------------------
    //  OR-reduce per-worker touched-row bitmaps into T.touched_bits[n]
    //  and count set bits for the lazy-zero density decision.  The OR is
    //  parallel per byte; the popcount is a single parallel pass.
    // ------------------------------------------------------------------
    T.touched_bits .assign((size_t)NM, std::vector<uint8_t>());
    T.touched_count.assign((size_t)NM, 0);
    for (int n = 0; n < NM; ++n) {
        const size_t nbytes = bit_off[n + 1] - bit_off[n];
        T.touched_bits[n].assign(nbytes, 0);
        uint8_t* out = T.touched_bits[n].data();
        const size_t off = bit_off[n];
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < (int64_t)nbytes; ++k) {
            uint8_t v = 0;
            for (int w = 0; w < OMPT; ++w)
                v |= w_bits[(size_t)w * bit_total + off + (size_t)k];
            out[k] = v;
        }
        uint64_t cnt = 0;
        #pragma omp parallel for reduction(+:cnt) schedule(static)
        for (int64_t k = 0; k < (int64_t)nbytes; ++k)
            cnt += (uint64_t)__builtin_popcount((unsigned)out[k]);
        T.touched_count[n] = cnt;
    }
}

// ---------------------------------------------------------------------------
//  Greedy LPT schedule (Dynasor Algorithm 3).
//
//  Consumes the precomputed `count_flat` built by sort_and_count_all -- no
//  tensor scanning happens here.  Total cost is O(sum_n K_n * log K_n) for
//  the per-mode descending sort plus O(sum_n K_n * log T) for the priority
//  queue, which is negligible against the element-scan work we just saved.
// ---------------------------------------------------------------------------
static void build_schedule_from_counts(Tensor&                     T,
                                       const std::vector<uint64_t>& count_flat,
                                       const std::vector<size_t>&   cnt_off,
                                       int                          num_threads)
{
    T.ss_list.assign(T.num_modes,
                     std::vector<std::vector<sid_t>>(num_threads));

    for (int n = 0; n < T.num_modes; ++n) {
        const idx_t     num_SS = T.shards_per_mode[n];
        const uint64_t* cn     = count_flat.data() + cnt_off[n];

        // Order super-shards by decreasing nnz count (LPT).
        std::vector<sid_t> order(num_SS);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](sid_t a, sid_t b) { return cn[a] > cn[b]; });

        // Min-heap of (current-load, tid).  Assign each shard to the
        // currently-least-loaded thread.
        using QE = std::pair<uint64_t, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> heap;
        for (int t = 0; t < num_threads; ++t) heap.push({0ULL, t});

        for (sid_t ss : order) {
            if (cn[ss] == 0) continue;          // empty shard -- skip entirely
            auto top = heap.top(); heap.pop();
            const int tid = top.second;
            T.ss_list[n][tid].push_back(ss);
            heap.push({top.first + cn[ss], tid});
        }
    }
}

// ---------------------------------------------------------------------------
//  Pre-compute the remap plan from already-known per-mode shard counts.
//
//  The counts in `count_flat` populate shard_begin/end with no re-scan.
//  Only the (tid, sid_next) histogram still requires a full element scan;
//  everything else is O(sum K).
// ---------------------------------------------------------------------------
static void build_remap_plan_from_counts(
    Tensor&                      T,
    const std::vector<uint64_t>& count_flat,
    const std::vector<size_t>&   cnt_off,
    int                          num_threads)
{
    const int      N    = T.num_modes;
    const uint64_t nnz  = T.nnz;
    const int      OMPT = dyn_omp_threads();
    T.precomp_num_threads = num_threads;

    constexpr size_t kLineBytes    = 64;
    constexpr size_t kUintsPerLine = kLineBytes / sizeof(uint64_t);

    // Per-mode cache-line-padded row stride (for the tid * stride + sid_next
    // addressing) and packed histogram offsets.
    std::vector<size_t> row_stride(N);
    std::vector<size_t> hist_off(N + 1, 0);
    for (int n = 0; n < N; ++n) {
        const idx_t K_next = T.shards_per_mode[(n + 1) % N];
        row_stride[n]   = ((size_t)K_next + kUintsPerLine - 1) & ~(kUintsPerLine - 1);
        hist_off[n + 1] = hist_off[n] + (size_t)num_threads * row_stride[n];
    }
    T.remap_row_stride = row_stride[0];
    const size_t hist_total = hist_off[N];

    // --------------------------------------------------------------------
    //  shard_begin / shard_end from count_flat.  No scanning.  We also
    //  build a flattened tid_of[] ownership map in the same loop, so the
    //  histogram pass touches only a single-level lookup for the tid.
    // --------------------------------------------------------------------
    T.shard_begin.assign(N, {});
    T.shard_end  .assign(N, {});

    std::vector<size_t> tid_off(N + 1, 0);
    for (int n = 0; n < N; ++n) tid_off[n + 1] = tid_off[n] + T.shards_per_mode[n];
    std::vector<int> tid_of_flat(tid_off[N], -1);

    for (int n = 0; n < N; ++n) {
        const idx_t     K  = T.shards_per_mode[n];
        const uint64_t* cn = count_flat.data() + cnt_off[n];
        T.shard_begin[n].assign(K, 0);
        T.shard_end  [n].assign(K, 0);
        uint64_t cum = 0;
        for (idx_t s = 0; s < K; ++s) {
            T.shard_begin[n][s] = cum;
            cum                += cn[s];
            T.shard_end  [n][s] = cum;
        }
        int* tflat = tid_of_flat.data() + tid_off[n];
        for (int tid = 0; tid < num_threads; ++tid)
            for (sid_t ss : T.ss_list[n][tid]) tflat[ss] = tid;
    }

    // Hoist per-mode pointers / shifts / tid-owner-map pointers.
    const idx_t* idx_ptr  [DYN_MAX_MODES];
    int          shft     [DYN_MAX_MODES];
    const int*   tid_ofN  [DYN_MAX_MODES];
    for (int n = 0; n < N; ++n) {
        idx_ptr[n] = T.idx_buf[n];
        shft   [n] = T.mn_shift[n];
        tid_ofN[n] = tid_of_flat.data() + tid_off[n];
    }

    // --------------------------------------------------------------------
    //  Histogram pass: for each element, for each mode n, bump the
    //  (tid_of_n, sid_next) bucket.  No counting here -- counts already
    //  live in `count_flat`.  This is the only full-tensor scan in this
    //  function.
    // --------------------------------------------------------------------
    // --------------------------------------------------------------------
    //  Histogram pass: for each element, for each mode n, bump the
    //  (tid_of_n, sid_next) bucket.  No counting here -- counts already
    //  live in `count_flat`.  This is the only full-tensor scan in this
    //  function.
    // --------------------------------------------------------------------
    std::vector<uint64_t> w_hist((size_t)OMPT * hist_total, 0);

    #pragma omp parallel
    {
        const int wid = dyn_omp_tid();
        uint64_t* myh = w_hist.data() + (size_t)wid * hist_total;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)nnz; ++i) {
            #pragma GCC unroll 8
            for (int n = 0; n < N; ++n) {
                const sid_t sid_n = (sid_t)(idx_ptr[n][i] >> shft[n]);
                const int   tid   = tid_ofN[n][sid_n];
                if (tid >= 0) {
                    const int   nm       = (n + 1) % N;
                    const sid_t sid_next = (sid_t)(idx_ptr[nm][i] >> shft[nm]);
                    myh[hist_off[n] + (size_t)tid * row_stride[n] + sid_next]++;
                }
            }
        }
    }

    std::vector<uint64_t> hist_flat(hist_total, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < (int64_t)hist_total; ++k) {
        uint64_t sum = 0;
        for (int w = 0; w < OMPT; ++w) sum += w_hist[(size_t)w * hist_total + k];
        hist_flat[k] = sum;
    }
    std::vector<uint64_t>().swap(w_hist);

    // --------------------------------------------------------------------
    //  Per-mode cursor table: precomp_thr_off[n][tid * stride + sid_next].
    //  Parallel over next-mode shards; the inner loop over tids is short.
    // --------------------------------------------------------------------
    T.precomp_thr_off.assign(N, {});
    for (int n = 0; n < N; ++n) {
        const int    nm     = (n + 1) % N;
        const idx_t  K_nm   = T.shards_per_mode[nm];
        const size_t stride = row_stride[n];
        T.precomp_thr_off[n].assign((size_t)num_threads * stride, 0);

        uint64_t*       dst = T.precomp_thr_off[n].data();
        const uint64_t* hn  = hist_flat.data() + hist_off[n];
        const uint64_t* bn  = T.shard_begin[nm].data();

        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < (int64_t)K_nm; ++s) {
            uint64_t cum = bn[s];
            for (int tid = 0; tid < num_threads; ++tid) {
                dst[(size_t)tid * stride + s] = cum;
                cum += hn[(size_t)tid * stride + s];
            }
        }
    }
}

// Inlined stable counting sort of the SoA slab [b, e) of `src_*` into
// the slab [b, e) of `dst_*`, using `key(i)` as the bucket id.
// `cnt` must be sized >= key_max + 1 and is used destructively.
//
// Hoisted above both NCopy populate (fiber-sort pass) and fiber_sort_shards
// so both call sites can reuse it without a forward declaration.
template<typename KeyFn>
static inline void counting_sort_one_shard(
    const value_t* DYN_RESTRICT        src_vals,
    const idx_t* const* DYN_RESTRICT   src_idx,
    value_t* DYN_RESTRICT              dst_vals,
    idx_t* const* DYN_RESTRICT         dst_idx,
    int                                NM,
    uint64_t b, uint64_t e,
    uint32_t key_max,
    std::vector<uint32_t>&             cnt,
    KeyFn                              key)
{
    const uint64_t K_s = e - b;
    if (K_s == 0) return;

    if (K_s == 1) {
        // Degenerate shard: just copy-through so the dst slot is populated.
        dyn_stream_f32(dst_vals + b, src_vals[b]);
        #pragma GCC unroll 8
        for (int w = 0; w < NM; ++w)
            dyn_stream_u32((uint32_t*)&dst_idx[w][b],
                           (uint32_t)src_idx[w][b]);
        return;
    }

    cnt.assign((size_t)key_max + 1, 0);

    // Pass A: histogram.
    for (uint64_t i = b; i < e; ++i) ++cnt[(size_t)key(i)];

    // Pass B: exclusive prefix scan.  cnt[k] now holds the first dst slot
    // (offset from b) reserved for bucket k.
    uint32_t acc = 0;
    for (uint32_t k = 0; k <= key_max; ++k) {
        uint32_t v = cnt[k];
        cnt[k] = acc;
        acc += v;
    }

    // Pass C: stable scatter.  NT stores skip the Read-For-Ownership on
    // dst cache lines, halving the effective write bandwidth.
    for (uint64_t i = b; i < e; ++i) {
        uint32_t k = (uint32_t)key(i);
        uint64_t d = b + (uint64_t)(cnt[k]++);
        dyn_stream_f32(dst_vals + d, src_vals[i]);
        #pragma GCC unroll 8
        for (int w = 0; w < NM; ++w)
            dyn_stream_u32((uint32_t*)&dst_idx[w][d],
                           (uint32_t)src_idx[w][i]);
    }
}

// ===========================================================================
//  Hybrid-layout helpers: shard-only remap (no FMA) + decision logic.
//
//  The PingPong layout keeps ONE primary slab and ONE scratch slab, and
//  every mode-MTTKRP rewrites (val + N idx) per nnz into the scratch slab
//  so the next mode can read from the freshly remapped layout.  That
//  extra 16-20 B/nnz of streamed writes shows up as measurable hot-path
//  bandwidth on multi-iteration ALS.
//
//  NCopy replaces the ping-pong with N physical copies of the tensor,
//  each sorted by one mode's shard order, all precomputed once.  The
//  kernel then reads copy[n] for mode n and does NOT remap.
//
//  To populate copies 1..N-1 we run the exact same shard scatter the
//  kernel would do, but WITHOUT the FMA accumulate.  The helper below
//  mirrors the kernel's work-off partitioning exactly so that the
//  resulting copy lands in the same format the kernel expects.
// ===========================================================================
static void dyn_remap_shard_only_all(
    const Tensor& T,
    int from_n,
    const value_t* DYN_RESTRICT src_vals,
    const idx_t* const*         src_idx,
    value_t* DYN_RESTRICT       dst_vals,
    idx_t* const*               dst_idx,
    int num_threads)
{
    const int    N          = T.num_modes;
    const int    next_mode  = (from_n + 1) % N;
    const int    shift_next = T.mn_shift[next_mode];
    const idx_t* DYN_RESTRICT ix_next = src_idx[next_mode];
    const auto&  sched_mode = T.ss_list[from_n];
    const std::vector<uint64_t>& plan = T.precomp_thr_off[from_n];

    // Stride matches the runtime kernel: plan.size() == num_threads * num_shards_next.
    // Each thread owns its slice of the cursor array, so two threads never
    // collide on the same target shard -- this is what makes the whole
    // remap pipeline atomic-free.
    if (num_threads <= 0) num_threads = 1;
    const size_t stride = plan.size() / (size_t)num_threads;

    // Per-run cursor template (memcpy of plan into our local scratch).
    std::vector<uint64_t> work_off(plan.begin(), plan.end());

    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads) {
            uint64_t* off = work_off.data() + (size_t)tid * stride;
            for (sid_t ss : sched_mode[tid]) {
                const uint64_t b = T.shard_begin[from_n][ss];
                const uint64_t e = T.shard_end  [from_n][ss];
                for (uint64_t i = b; i < e; ++i) {
                    const sid_t    sid_nxt = (sid_t)(ix_next[i] >> shift_next);
                    const uint64_t d       = (uint64_t)(off[sid_nxt]++);
                    dyn_stream_f32(dst_vals + d, src_vals[i]);
                    #pragma GCC unroll 8
                    for (int w = 0; w < N; ++w)
                        dyn_stream_u32((uint32_t*)&dst_idx[w][d],
                                       (uint32_t)src_idx[w][i]);
                }
            }
        }
        dyn_sfence_stores();
    }
    dyn_sfence_stores();
}

// ===========================================================================
//  InPlace layout: cycle-following in-place counting sort of the primary
//  SoA by ix[n].  Two-level radix (shard then intra-shard row) so that
//  the histogram counter arrays stay small even when mode_size[n] is in
//  the 10^7+ range -- we never allocate I_n counters.
//
//  Memory overhead: 2 * (nshards+1) uint64 for the shard pass (tiny), plus
//  2 * (m_n+1) uint64 PER THREAD for the within-shard pass (also tiny --
//  m_n is the power-of-2 shard width, typically 2^13 = 8192).
//
//  Parallelism:
//    Pass 1 (shard bucket): sequential.  Cycle-following on O(nshards)
//      buckets is inherently sequential because swaps span the whole slab
//      and any parallel strategy requires cross-thread coordination that
//      costs more than the pass itself.  Empirically this pass runs at
//      ~3-4 GB/s of swap bandwidth on a single core -- ~80 ms on 100M
//      nnz for a 3D tensor -- which is comparable to PingPong's scatter
//      traffic once the factor-gather cost dominates.
//
//    Pass 2 (intra-shard row): parallel across shards (disjoint ranges).
//      Each thread cycle-sorts its shards into fiber-contiguous order
//      using a local histogram of size m_n+1.  Skipped when the mode
//      is NOT classified as DenseFiber -- the element kernel only needs
//      shard-level grouping, which Pass 1 already provides.
//
//  Not supported under InPlace:
//    - NCopy + CSR slab compaction (needs all copies present).
//    - Per-mode ix[n] drop optimization (needs the column for subsequent
//      re-sorts).
// ===========================================================================
void inplace_sort_slab_by_mode(Tensor& T, int n, int num_threads) {
    const uint64_t nnz  = T.nnz;
    const int      N    = T.num_modes;
    const int      shft = T.mn_shift[n];
    const idx_t    nsh  = T.shards_per_mode[n];
    const idx_t    m_n  = T.mn[n];
    idx_t*   DYN_RESTRICT ixn = T.idx_buf[n];
    value_t* DYN_RESTRICT vals = T.vals;

    if (nnz <= 1 || nsh <= 1) return;
    if (num_threads <= 0) num_threads = 1;

    // --- Pass 1: shard bucket sort ---------------------------------------
    //
    //  Two code paths:
    //
    //   (a) Serial American-flag sort (cycle-following counting sort).
    //       Each element moves at most twice, O(nnz) swaps total.  Used
    //       when nnz is small, nnz exceeds 2^31 (the src[] array can't
    //       index past uint32 without going to 8 B/nnz), num_threads==1,
    //       or DYN_INPLACE_PARALLEL=0 forces it.
    //
    //   (b) Parallel 3-step pass: per-thread histogram -> shared src[]
    //       permutation via disjoint thread cursors -> parallel in-place
    //       permutation with cycle leader-election.  Uses 4 B/nnz of
    //       transient scratch (the src[] array) -- the InPlace layout
    //       is still "1 x NNZ steady" because src[] is released before
    //       Pass 2 runs.  Scales near-linearly in num_threads for the
    //       histogram and scatter steps; the cycle-apply step is O(nnz
    //       + sum_of_cycle_lengths^2 / avg_cycle_length) with perfect
    //       parallelism via min-index leader election.
    //
    //  Pass 2 runs unchanged in both cases (already parallel over shards).

    const char* parallel_env = std::getenv("DYN_INPLACE_PARALLEL");
    const bool  parallel_off = parallel_env && parallel_env[0] == '0';
    const size_t scr_need    = sizeof(value_t) * (size_t)nnz;
    const bool  scr_ok       = (T.inplace_scr_raw != nullptr)
                             && (T.inplace_scr_bytes >= scr_need);
    const bool  use_parallel = !parallel_off
                             && (num_threads > 1)
                             && (nnz >= 16384ULL)
                             && scr_ok;

    std::vector<uint64_t> cnt((size_t)nsh + 1, 0);

    if (!use_parallel) {
        // ---- Path (a): serial cycle-following sort --------------------
        for (uint64_t i = 0; i < nnz; ++i) {
            const idx_t s = (idx_t)(ixn[i] >> shft);
            ++cnt[(size_t)s + 1];
        }
        for (idx_t s = 1; s <= nsh; ++s) cnt[s] += cnt[s - 1];

        std::vector<uint64_t> beg(cnt);

        for (idx_t s = 0; s < nsh; ++s) {
            const uint64_t e_s = cnt[s + 1];
            while (beg[s] < e_s) {
                const uint64_t i = beg[s];
                const idx_t    cur = (idx_t)(ixn[i] >> shft);
                if (cur == s) { ++beg[s]; continue; }
                const uint64_t j = beg[cur]++;
                if (i != j) {
                    std::swap(vals[i], vals[j]);
                    #pragma GCC unroll 8
                    for (int w = 0; w < N; ++w)
                        std::swap(T.idx_buf[w][i], T.idx_buf[w][j]);
                }
            }
        }
    } else {
        // ---- Path (b): parallel counting sort via scatter + copy-back.
        //
        //  Idea: do (N+1) cache-friendly passes over the slab, one per
        //  SoA column.  Each pass:
        //    1. Parallel scatter.  Each thread reads its slice of the
        //       source column sequentially, looks up the destination
        //       from a per-thread shard cursor, and writes into a
        //       shared scratch buffer.  Per-thread cursor ranges are
        //       pairwise disjoint (built from the per-thread histogram),
        //       so no atomics are needed.
        //    2. Parallel copy-back scratch -> column.  Both reads and
        //       writes are sequential.
        //
        //  Why this beats an in-place cycle sort at scale:  the bucket-
        //  sort permutation implied by our shard-id tends to have very
        //  long cycles for a mode-transition (e.g. mode 1 -> mode 2 on
        //  bench_3d_10M), which makes cycle-follow sorts cache-hostile.
        //  The scatter + copy-back pattern does 2 x nnz x sizeof(col)
        //  sequential passes per column with write-locality per shard,
        //  trading ~1 extra column-pass of bandwidth per column for
        //  O(T)-parallel execution and clean cache behaviour.  Uses
        //  sizeof(value_t) * nnz of scratch (allocated once in
        //  decide_and_populate_layout, reused across all sorts).
        //
        //  The ix[n] column is processed LAST because the scatter's
        //  shard-id lookup reads ix[n].  Doing it last preserves the
        //  original ix[n] values for all prior columns.  Reading and
        //  writing within the same pass is still safe: each iteration
        //  reads ix[n][i] (source), computes its destination d, and
        //  writes scr[d] = ix[n][i].  scr is completely separate from
        //  ix[n], so no RAW hazard.

        // Step A: per-thread histogram.
        std::vector<std::vector<uint64_t>> histo(
            (size_t)num_threads, std::vector<uint64_t>((size_t)nsh, 0));
        const uint64_t chunk =
            (nnz + (uint64_t)num_threads - 1) / (uint64_t)num_threads;

        #pragma omp parallel num_threads(num_threads)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            if (tid < num_threads) {
                const uint64_t lo = (uint64_t)tid * chunk;
                const uint64_t hi = std::min(nnz, lo + chunk);
                uint64_t* DYN_RESTRICT h = histo[tid].data();
                for (uint64_t i = lo; i < hi; ++i) {
                    ++h[(size_t)(ixn[i] >> shft)];
                }
            }
        }

        // Step B: build per-thread starting cursors (pairwise disjoint)
        //         and the global shard-start table cnt[].
        //         We keep one "initial" snapshot so we can reset cursors
        //         between columns; they get bumped by each scatter.
        std::vector<std::vector<uint64_t>> thr_off0(
            (size_t)num_threads, std::vector<uint64_t>((size_t)nsh, 0));
        std::vector<std::vector<uint64_t>> thr_off(
            (size_t)num_threads, std::vector<uint64_t>((size_t)nsh, 0));
        {
            uint64_t running = 0;
            for (idx_t s = 0; s < nsh; ++s) {
                cnt[s] = running;
                for (int t = 0; t < num_threads; ++t) {
                    thr_off0[t][s] = running;
                    running += histo[t][s];
                }
            }
            cnt[nsh] = running;
        }

        // Step C/D: scatter + copy-back per column.
        //  Use the single scratch buffer cast to value_t* or idx_t*.
        value_t* const scr_f = (value_t*)T.inplace_scr_raw;
        idx_t*   const scr_i = (idx_t*)  T.inplace_scr_raw;

        auto reset_cursors = [&] {
            for (int t = 0; t < num_threads; ++t) {
                std::memcpy(thr_off[t].data(), thr_off0[t].data(),
                            sizeof(uint64_t) * (size_t)nsh);
            }
        };

        auto scatter_f = [&](const value_t* DYN_RESTRICT col) {
            reset_cursors();
            #pragma omp parallel num_threads(num_threads)
            {
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                if (tid < num_threads) {
                    const uint64_t lo = (uint64_t)tid * chunk;
                    const uint64_t hi = std::min(nnz, lo + chunk);
                    uint64_t* DYN_RESTRICT cur = thr_off[tid].data();
                    for (uint64_t i = lo; i < hi; ++i) {
                        const idx_t s = (idx_t)(ixn[i] >> shft);
                        scr_f[cur[(size_t)s]++] = col[i];
                    }
                }
            }
            #pragma omp parallel for num_threads(num_threads) \
                    schedule(static)
            for (int64_t j = 0; j < (int64_t)nnz; ++j) {
                const_cast<value_t*>(col)[j] = scr_f[j];
            }
        };
        auto scatter_i = [&](idx_t* DYN_RESTRICT col) {
            reset_cursors();
            #pragma omp parallel num_threads(num_threads)
            {
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                if (tid < num_threads) {
                    const uint64_t lo = (uint64_t)tid * chunk;
                    const uint64_t hi = std::min(nnz, lo + chunk);
                    uint64_t* DYN_RESTRICT cur = thr_off[tid].data();
                    for (uint64_t i = lo; i < hi; ++i) {
                        const idx_t s = (idx_t)(ixn[i] >> shft);
                        scr_i[cur[(size_t)s]++] = col[i];
                    }
                }
            }
            #pragma omp parallel for num_threads(num_threads) \
                    schedule(static)
            for (int64_t j = 0; j < (int64_t)nnz; ++j) {
                col[j] = scr_i[j];
            }
        };

        // Process every column except ix[n] first ...
        scatter_f(vals);
        for (int w = 0; w < N; ++w) {
            if (w == n) continue;
            scatter_i(T.idx_buf[w]);
        }
        // ... then ix[n] last, so the shard-id lookup in each prior
        // pass reads the original order of ix[n].
        scatter_i(T.idx_buf[n]);
    }

    // --- Pass 2: within-shard in-place cycle-follow by intra-shard row. ---
    // Required so the fiber kernel can fold each fiber into SIMD regs.
    // Safe to skip for non-fiber modes: the element kernel only partitions
    // work by shard id, which Pass 1 already established.
    if (T.kernel_class[n] != KernelClass::DenseFiber) return;

    if (num_threads <= 0) num_threads = 1;
    #pragma omp parallel num_threads(num_threads)
    {
        std::vector<uint64_t> shcnt;
        std::vector<uint64_t> shbeg;
        shcnt.reserve((size_t)m_n + 1);
        shbeg.reserve((size_t)m_n + 1);

        #pragma omp for schedule(dynamic)
        for (int64_t s64 = 0; s64 < (int64_t)nsh; ++s64) {
            const idx_t    s        = (idx_t)s64;
            const uint64_t b        = cnt[s];
            const uint64_t e        = cnt[s + 1];
            if (e - b <= 1) continue;
            const idx_t    row_base = (idx_t)(s << shft);

            shcnt.assign((size_t)m_n + 1, 0);
            for (uint64_t i = b; i < e; ++i) {
                const idx_t k = (idx_t)(ixn[i] - row_base);
                ++shcnt[(size_t)k + 1];
            }
            for (idx_t k = 1; k <= m_n; ++k) shcnt[k] += shcnt[k - 1];
            shbeg = shcnt;  // copy

            for (idx_t k = 0; k < m_n; ++k) {
                const uint64_t e_k = shcnt[k + 1];
                while (shbeg[k] < e_k) {
                    const uint64_t i_rel = shbeg[k];
                    const uint64_t i     = b + i_rel;
                    const idx_t    cur   = (idx_t)(ixn[i] - row_base);
                    if (cur == k) { ++shbeg[k]; continue; }
                    const uint64_t j_rel = shbeg[cur]++;
                    const uint64_t j     = b + j_rel;
                    if (i != j) {
                        std::swap(vals[i], vals[j]);
                        #pragma GCC unroll 8
                        for (int w = 0; w < N; ++w)
                            std::swap(T.idx_buf[w][i], T.idx_buf[w][j]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  dyn_plan_storage  --  unified cross-layout planner.
//
//  Shared by:
//    * main.cpp (pre-load, to decide OOC vs in-core before touching nnz)
//    * decide_and_populate_layout (post-load, to pick among in-core tiers)
//
//  Override precedence (highest first):
//    1. --layout ooc | DYN_LAYOUT=ooc   -> OutOfCore
//    2. --ooc on     | DYN_OOC=on       -> OutOfCore
//    3. --layout {pingpong,ncopy,inplace} | DYN_LAYOUT=...  -> that layout
//    4. Auto:
//         a. if --ooc != "off" AND inplace_need > budget   -> OutOfCore
//         b. else prefer NCopy -> PingPong -> InPlace (first that fits)
//         c. if even InPlace doesn't fit AND --ooc == "off"
//               -> fall through with reason "DOES NOT FIT" so the caller
//                  can emit the OOM banner.
// ---------------------------------------------------------------------------
namespace {

inline std::string tolower_copy(const std::string& s) {
    std::string out = s;
    for (auto& c : out) c = (char)std::tolower((unsigned char)c);
    return out;
}

// Normalize a user-supplied layout override string.  Returns one of
// {"", "auto", "pingpong", "ncopy", "inplace", "ooc"}; "" for unknown.
inline std::string normalize_layout_override(const std::string& in) {
    std::string v = tolower_copy(in);
    if (v.empty() || v == "auto")                   return "auto";
    if (v == "pingpong" || v == "pp")               return "pingpong";
    if (v == "ncopy"    || v == "nc")               return "ncopy";
    if (v == "inplace"  || v == "ip")               return "inplace";
    if (v == "ooc"      || v == "outofcore" ||
        v == "out-of-core")                         return "ooc";
    return "";  // unknown -- caller should treat as "auto"
}

inline std::string normalize_ooc_mode(const std::string& in) {
    std::string v = tolower_copy(in);
    if (v.empty())         return "auto";
    if (v == "on"  ||
        v == "yes" ||
        v == "1")          return "on";
    if (v == "off" ||
        v == "no"  ||
        v == "0")          return "off";
    if (v == "auto")       return "auto";
    return "auto";         // unknown -> auto
}

} // namespace

StoragePlan dyn_plan_storage(uint64_t           nnz,
                             int                num_modes,
                             const idx_t*       mode_sizes,
                             int                rank_padded_est,
                             bool               has_dense_fiber_hint,
                             const std::string& layout_cli,
                             const std::string& ooc_cli)
{
    StoragePlan p{};

    // 1. Footprints per layout (tensor-only).
    p.pingpong_bytes = dyn_tensor_pingpong_bytes(nnz, num_modes);
    p.ncopy_bytes    = dyn_tensor_ncopy_bytes   (nnz, num_modes);
    p.inplace_bytes  = dyn_tensor_inplace_bytes (nnz, num_modes);

    // 2. Factor-matrix + scratch overhead.
    //    Each mode n contributes one I_n x R slab (Yhat) and one I_n x R
    //    slab (gram/LS scratch).  Pre-load we may not know mode_sizes;
    //    the caller passes nullptr and we estimate from a small default.
    size_t sum_In = 0;
    if (mode_sizes) {
        for (int n = 0; n < num_modes; ++n) sum_In += (size_t)mode_sizes[n];
    } else {
        sum_In = (size_t)num_modes * 16384ULL;  // 16K per mode is a generous upper bound for FROSTT
    }
    const int    Rp           = (rank_padded_est > 0) ? rank_padded_est : 64;
    const size_t factors_est  = 2ULL * sum_In * (size_t)Rp * sizeof(value_t);

    // If any mode is DenseFiber the NCopy populate pass transiently keeps
    // ONE extra copy-sized slab for the counting-sort-back-copy.  The peak
    // is freed before build_flycoo returns but might OOM us meanwhile.
    const size_t per_nnz = sizeof(value_t) + (size_t)num_modes * sizeof(idx_t);
    const size_t sort_temp_bytes = has_dense_fiber_hint
                                 ? per_nnz * (size_t)nnz : 0ULL;

    p.overhead_bytes = factors_est + sort_temp_bytes + (64ULL << 20);

    p.pingpong_need = p.pingpong_bytes + p.overhead_bytes;
    p.ncopy_need    = p.ncopy_bytes    + p.overhead_bytes;
    p.inplace_need  = p.inplace_bytes  + p.overhead_bytes;

    // OOC: factors + factor-matrix ofibs + one chunk buffer + slack.  The
    // exact chunk size is picked later, use 256 MiB as a planning value.
    const size_t ooc_factor_ofibs = 2ULL * sum_In * (size_t)Rp
                                  * sizeof(value_t);
    p.ooc_need = factors_est + ooc_factor_ofibs
               + (256ULL << 20)          // chunk buffer
               + (256ULL << 20);         // slack

    // 3. Budget.
    p.avail_bytes = dyn_memory_available_bytes();
    p.total_bytes = dyn_memory_total_bytes();
    constexpr double kUsableFraction = 0.70;
    const size_t headroom = std::max<size_t>(
        (size_t)(2ULL << 30),
        p.total_bytes ? (size_t)(0.25 * (double)p.total_bytes)
                      : (size_t)(2ULL << 30));
    size_t usable = 0;
    if (p.avail_bytes)      usable = (size_t)((double)p.avail_bytes * kUsableFraction);
    else if (p.total_bytes) usable = (size_t)((double)p.total_bytes * 0.50);
    p.budget_bytes = (usable > headroom) ? usable - headroom : 0;

    // 4. Resolve overrides.
    const std::string cli_l = normalize_layout_override(layout_cli);
    const std::string cli_o = normalize_ooc_mode(ooc_cli);

    // DYN_LAYOUT env wins over CLI only when CLI is "auto" or empty.
    std::string env_l;
    if (const char* e = std::getenv("DYN_LAYOUT"); e && *e)
        env_l = normalize_layout_override(e);
    std::string env_o;
    if (const char* e = std::getenv("DYN_OOC"); e && *e)
        env_o = normalize_ooc_mode(e);

    const std::string layout_eff =
        (cli_l != "auto" && !cli_l.empty()) ? cli_l :
        (!env_l.empty() ? env_l : "auto");
    const std::string ooc_eff =
        (cli_o != "auto") ? cli_o :
        (!env_o.empty() ? env_o : "auto");

    p.override_source = "auto";

    // --layout ooc  /  DYN_LAYOUT=ooc
    if (layout_eff == "ooc") {
        p.choice          = Layout::OutOfCore;
        p.reason          = "layout override = ooc";
        p.override_source = (cli_l == "ooc") ? "cli-layout" : "env-layout";
        return p;
    }
    // --ooc on  /  DYN_OOC=on
    if (ooc_eff == "on") {
        p.choice          = Layout::OutOfCore;
        p.reason          = "--ooc on override";
        p.override_source = (cli_o == "on") ? "cli-ooc" : "env-ooc";
        return p;
    }
    // --layout {pingpong,ncopy,inplace}  /  DYN_LAYOUT=...
    if (layout_eff == "pingpong") {
        p.choice          = Layout::PingPong;
        p.reason          = "layout override = pingpong";
        p.override_source = (cli_l == "pingpong") ? "cli-layout" : "env-layout";
        return p;
    }
    if (layout_eff == "ncopy") {
        p.choice          = Layout::NCopy;
        p.reason          = "layout override = ncopy";
        p.override_source = (cli_l == "ncopy") ? "cli-layout" : "env-layout";
        return p;
    }
    if (layout_eff == "inplace") {
        p.choice          = Layout::InPlace;
        p.reason          = "layout override = inplace";
        p.override_source = (cli_l == "inplace") ? "cli-layout" : "env-layout";
        return p;
    }

    // 5. Auto heuristic.
    //    First: would ANY in-core layout fit?  If not (and OOC isn't
    //    explicitly disabled), go OOC before attempting to load.
    if (ooc_eff != "off" && p.inplace_need > p.budget_bytes) {
        p.choice = Layout::OutOfCore;
        p.reason = "auto: no in-core layout fits (inplace_need > budget)";
        return p;
    }

    // In-core tiering (mirrors the original decide_layout preference).
    if (num_modes < 2) {
        p.choice = Layout::PingPong;
        p.reason = "num_modes < 2 (no benefit from NCopy)";
        return p;
    }
    if (p.ncopy_need <= p.budget_bytes) {
        p.choice = Layout::NCopy;
        p.reason = "fits in memory budget";
        return p;
    }
    if (p.pingpong_need <= p.budget_bytes) {
        p.choice = Layout::PingPong;
        p.reason = "NCopy too large; PingPong fits";
        return p;
    }
    if (p.inplace_need <= p.budget_bytes) {
        p.choice = Layout::InPlace;
        p.reason = "PingPong too large; InPlace is minimum footprint";
        return p;
    }

    // Even 1 x NNZ + factors + slack doesn't fit AND OOC is disabled.
    // Emit a cosmetic choice; decide_and_populate_layout will abort loudly.
    p.choice = Layout::InPlace;
    p.reason = "TENSOR DOES NOT FIT IN RAM (even at 1 x NNZ) and --ooc off";
    return p;
}

void dyn_print_plan(const StoragePlan& p, const char* prefix) {
    auto gib = [](size_t b) {
        return (double)b / (1024.0 * 1024.0 * 1024.0);
    };
    const char* choice_str =
        (p.choice == Layout::NCopy)     ? "NCopy"     :
        (p.choice == Layout::PingPong)  ? "PingPong"  :
        (p.choice == Layout::InPlace)   ? "InPlace"   :
        (p.choice == Layout::Morton)    ? "Morton"    :
        (p.choice == Layout::OutOfCore) ? "OutOfCore" : "<unknown>";

    std::printf(
        "%s memory : total=%.2f GiB  available=%.2f GiB  budget=%.2f GiB\n"
        "%s tensor : pingpong=%.2f GiB  ncopy=%.2f GiB  "
        "inplace=%.2f GiB  overhead=%.2f GiB\n"
        "%s candidates : nc_need=%.2f  pp_need=%.2f  "
        "ip_need=%.2f  ooc_need=%.2f (GiB)\n"
        "%s decision : %s  (%s; src=%s)\n",
        prefix, gib(p.total_bytes), gib(p.avail_bytes), gib(p.budget_bytes),
        prefix, gib(p.pingpong_bytes), gib(p.ncopy_bytes),
                gib(p.inplace_bytes), gib(p.overhead_bytes),
        prefix, gib(p.ncopy_need), gib(p.pingpong_need),
                gib(p.inplace_need), gib(p.ooc_need),
        prefix, choice_str, p.reason, p.override_source);
}

// ===========================================================================
//  Stage 4 -- unified ComputePlan builder.
//
//  Encodes the decision tree described in the Stage 4 plan:
//
//    [kernel-algorithm]
//      * OOC / Morton            -> Jacobi (only supported path)
//      * nnz < 1e6               -> GaussSeidel (linalg dominates)
//      * all-modes fallback hit  -> GaussSeidel (N==3 & Rp>=64, or Rp>=128)
//      * ofibs > RAM/4, cap 8GiB -> GaussSeidel (memory)
//      * else                    -> Jacobi
//
//    [per-mode kernel impl]  (GS only; Jacobi does not use per-mode)
//      * NCopy AND ncopy_csr[n]  -> FiberCSR
//      * kernel_class == DenseFiber OR avg_fiber_len>=32 -> Fiber
//      * else                    -> Element
//
//    [pf_far]
//      * 2 * sum(I_n)*Rp*4 B > 32 MiB  -> on  (factors overflow L3)
//      * else                          -> off
//
//    [use_jit]
//      * DYN_JIT / --jit user-enabled AND:
//        - layout == Morton AND nnz >= 1e6  (baked masks pay back)
//        - OR N > 5 (baked-grid miss)
//        - OR Rp not in {16,32,64,128,256}
//
//  All thresholds are tuned against the Stage 3 benchmark corpus and
//  documented inline.  Overrides preserved: layout_cli / ooc_cli /
//  kernel_cli match the exact keywords main.cpp used to parse before.
// ===========================================================================

// Local helper: case-insensitive compare on a single keyword.
static inline bool s4_ieq(const std::string& s, const char* lit) {
    if (s.size() != std::strlen(lit)) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        char a = (char)std::tolower((unsigned char)s[i]);
        if (a != lit[i]) return false;
    }
    return true;
}
static inline std::string s4_lower(const std::string& s) {
    std::string r; r.reserve(s.size());
    for (char c : s) r.push_back((char)std::tolower((unsigned char)c));
    return r;
}

ComputePlan dyn_build_compute_plan(const Tensor&      T,
                                   int                rank_padded,
                                   int                num_threads,
                                   const std::string& layout_cli,
                                   const std::string& ooc_cli,
                                   const std::string& kernel_cli)
{
    ComputePlan p{};
    p.num_modes   = T.num_modes;
    p.rank_padded = rank_padded;
    p.num_threads = num_threads;

    // -------- storage plan (reuses the existing cost model). --------------
    bool any_dense_fiber = false;
    for (int n = 0; n < T.num_modes; ++n)
        if (T.kernel_class[n] == KernelClass::DenseFiber)
            any_dense_fiber = true;
    p.storage = dyn_plan_storage(T.nnz, T.num_modes,
                                 T.mode_size, rank_padded,
                                 any_dense_fiber, layout_cli, ooc_cli);
    // If the tensor is already Morton/OOC in-flight (e.g. main.cpp converted
    // it after build_flycoo), reflect that in the plan so downstream
    // dispatch can trust the plan alone.
    if (T.layout == Layout::Morton || T.layout == Layout::OutOfCore) {
        p.storage.choice = T.layout;
    }

    // -------- kernel-algorithm decision (lifted pick_auto). ---------------
    const int N  = T.num_modes;
    const int Rp = rank_padded;

    // ofibs footprint matches spmttkrp_all_modes_dynasor allocation.
    int nt = num_threads;
    if (nt <= 0) {
#ifdef _OPENMP
        nt = omp_get_max_threads();
#else
        nt = 1;
#endif
    }
    size_t per_thread_bytes = 0;
    for (int n = 0; n < N; ++n) {
        size_t b = (size_t)T.mode_size[n] * (size_t)Rp * sizeof(value_t);
        b = (b + 63ULL) & ~63ULL;
        per_thread_bytes += b;
    }
    const uint64_t ofibs_bytes = (uint64_t)per_thread_bytes * (uint64_t)nt;
    p.ofibs_bytes = (size_t)ofibs_bytes;

    // Factor-matrix bytes (Y + Yhat) -- used for pf_far decision below.
    size_t factor_bytes = 0;
    for (int n = 0; n < N; ++n) {
        factor_bytes += 2 * (size_t)T.mode_size[n]
                          * (size_t)Rp * sizeof(value_t);
    }
    p.factor_bytes = factor_bytes;

    const uint64_t avail  = (uint64_t)dyn_memory_available_bytes();
    const uint64_t cap8   = (uint64_t)8 * 1024ULL * 1024ULL * 1024ULL;
    const uint64_t budget = std::min(avail / 4, cap8);

    // Parse kernel CLI (env override wins).  "auto" / "" are treated as
    // "no explicit user choice" so the banner doesn't mislead into
    // thinking a CLI flag was given.
    std::string kcli = kernel_cli;
    const char* kenv = std::getenv("DYN_CP_KERNEL");
    const char* src  = "auto";
    if (kenv && kenv[0]) { kcli = kenv; src = "env-kernel"; }
    else if (!kernel_cli.empty() && kernel_cli != "auto") {
        src = "cli-kernel";
    }

    const std::string kl = s4_lower(kcli);

    auto pick_auto_kernel = [&]() -> int {
        // Morton / OOC only run under the all-modes Jacobi kernel; GS has
        // no compatible driver for these layouts.
        if (p.storage.choice == Layout::Morton ||
            p.storage.choice == Layout::OutOfCore ||
            T.ooc_enabled) {
            p.kernel_reason = "forced-jacobi-by-layout";
            return 1;  // Jacobi
        }
        // Tiny tensors: linalg dominates total time, use GS for freshest
        // factors and faster convergence per iter.
        if (T.nnz < (uint64_t)1'000'000) {
            p.kernel_reason = "nnz<1M (linalg-dominated)";
            return 0;  // GaussSeidel
        }
        // Match the internal fallback predicates of
        // spmttkrp_all_modes_dynasor -- when the all-modes kernel
        // would itself fall back to per-mode, just pick GS up-front.
        if ((N == 3 && Rp >= 64) || Rp >= 128) {
            p.kernel_reason = "all-modes-fallback-shape";
            return 0;  // GaussSeidel
        }
        // Memory budget for per-thread private output buffers.
        if (ofibs_bytes >= budget) {
            p.kernel_reason = "ofibs>budget";
            return 0;  // GaussSeidel
        }
        p.kernel_reason = "all-modes fits";
        return 1;      // Jacobi
    };

    if (s4_ieq(kl, "jacobi") || s4_ieq(kl, "all-modes") ||
        s4_ieq(kl, "all_modes") || s4_ieq(kl, "js")) {
        p.kernel_enum = 1;
        p.kernel_reason = "user-forced-jacobi";
    } else if (s4_ieq(kl, "gauss-seidel") || s4_ieq(kl, "gs") ||
               s4_ieq(kl, "classical") || s4_ieq(kl, "per-mode") ||
               s4_ieq(kl, "per_mode")) {
        p.kernel_enum = 0;
        p.kernel_reason = "user-forced-gs";
        // If layout forces Jacobi, the user's GS pick is about to be
        // overridden by cpals(); leave the reason honest so the banner
        // shows the conflict.
        if (p.storage.choice == Layout::Morton ||
            p.storage.choice == Layout::OutOfCore || T.ooc_enabled) {
            p.kernel_enum = 1;
            p.kernel_reason = "user-wanted-gs-but-layout-forces-jacobi";
        }
    } else if (kl.empty() || s4_ieq(kl, "auto")) {
        p.kernel_enum = pick_auto_kernel();
    } else {
        std::fprintf(stderr, "warning: dyn_build_compute_plan: unknown "
                             "kernel '%s'; falling back to auto.\n",
                     kernel_cli.c_str());
        p.kernel_enum = pick_auto_kernel();
    }
    p.kernel_source = src;

    // -------- per-mode kernel impl (Gauss-Seidel only). -------------------
    for (int n = 0; n < N; ++n) {
        KernelImpl impl = KernelImpl::Element;
        if (p.storage.choice == Layout::NCopy && T.ncopy_csr[n]) {
            impl = KernelImpl::FiberCSR;
        } else if (T.kernel_class[n] == KernelClass::DenseFiber ||
                   T.avg_fiber_len[n] >= 32.0) {
            impl = KernelImpl::Fiber;
        }
        p.per_mode[n] = impl;
    }

    // -------- pf_far: factors overflow L3 budget? --------------------------
    // 32 MiB is the empirical cutoff below which the extra T2 prefetches
    // contend with demand misses and cost more than they save (see
    // dynasor_kernel.cpp comment on DYN_PF_FAR).  Above that, the random-
    // row access pattern is dominated by DRAM latency and the L2 hint
    // pays for itself.
    {
        const char* pf_env = std::getenv("DYN_PF_FAR");
        if (pf_env && pf_env[0]) {
            p.pf_far = (pf_env[0] != '0');
        } else {
            p.pf_far = (factor_bytes > (size_t)32 * 1024 * 1024);
        }
    }

    // -------- use_jit: plan recommendation (independent of runtime). -----
    //
    //  This field answers "would a JIT specialization pay back on this
    //  shape?" -- not "is JIT enabled now?".  main.cpp consumes it to
    //  auto-enable JIT when the plan says yes AND the user didn't
    //  explicitly opt out.  Reported in the banner as the effective
    //  runtime state after applying the promotion.
    //
    //  Rule 1: Morton layout lets us bake per-mode masks as immediates
    //          so _pext_u64 compiles to a single-cycle constant-mask
    //          BMI2 op.  Worth it on any non-trivial tensor.
    //  Rule 2: out-of-baked-grid shapes (N=2 or 6..8, or Rp not in the
    //          {16,32,64,128,256} set).  The baked runtime-fallback
    //          path is un-unrolled; JIT closes the 3-5x gap.
    p.use_jit = false;
    if (p.storage.choice == Layout::Morton &&
        T.nnz >= (uint64_t)1'000'000) {
        p.use_jit = true;
    }
    const bool Rp_in_grid =
        (Rp == 16 || Rp == 32 || Rp == 64 || Rp == 128 || Rp == 256);
    const bool N_in_grid = (N >= 3 && N <= 5);
    if (!Rp_in_grid || !N_in_grid) p.use_jit = true;

    return p;
}

void dyn_print_compute_plan(const ComputePlan& p, const char* prefix) {
    // Print the storage rows first (same format as dyn_print_plan).
    dyn_print_plan(p.storage, prefix);

    const char* kernel_str = (p.kernel_enum == 1) ? "Jacobi" : "GaussSeidel";

    // Per-mode impl summary (ignored in Jacobi but helpful for debugging).
    // Bounded by p.num_modes -- no sentinel scans.
    char per_mode_buf[128];
    int  off = 0;
    off += std::snprintf(per_mode_buf + off, sizeof(per_mode_buf) - off, "[");
    const int N = p.num_modes > 0 ? p.num_modes : 0;
    for (int n = 0; n < N; ++n) {
        if (off >= (int)sizeof(per_mode_buf) - 16) break;
        const char* s =
            (p.per_mode[n] == KernelImpl::FiberCSR) ? "FiberCSR" :
            (p.per_mode[n] == KernelImpl::Fiber)    ? "Fiber"    : "Element";
        off += std::snprintf(per_mode_buf + off, sizeof(per_mode_buf) - off,
                             "%s%s", n ? "," : "", s);
    }
    std::snprintf(per_mode_buf + off, sizeof(per_mode_buf) - off, "]");

    auto mib = [](size_t b) { return (double)b / (1024.0 * 1024.0); };
    std::printf(
        "%s kernel : %s  (reason=%s; src=%s)\n"
        "%s per-mode : %s  (only used under GS)\n"
        "%s tuning : pf_far=%s  jit=%s  Rp=%d  threads=%d\n"
        "%s budgets : ofibs=%.2f MiB  factors=%.2f MiB\n",
        prefix, kernel_str, p.kernel_reason, p.kernel_source,
        prefix, per_mode_buf,
        prefix, p.pf_far  ? "on" : "off",
                p.use_jit ? "on" : "off",
                p.rank_padded, p.num_threads,
        prefix, mib(p.ofibs_bytes), mib(p.factor_bytes));
}

// Thin back-compat wrapper: post-load caller keeps the old 4-arg signature.
// Forwards to dyn_plan_storage with has_dense_fiber_hint computed from the
// already-classified kernel_class[] array.
static Layout decide_layout(const Tensor& T, size_t& need_bytes,
                            size_t& budget_bytes, const char*& reason)
{
    bool any_dense_fiber = false;
    for (int n = 0; n < T.num_modes; ++n)
        if (T.kernel_class[n] == KernelClass::DenseFiber) any_dense_fiber = true;

    // Post-load plan: no OOC override possible (we're already in RAM);
    // pass ooc="off" to keep the three-tier in-core selection.
    StoragePlan p = dyn_plan_storage(T.nnz, T.num_modes,
                                     T.mode_size, /*rank_padded_est=*/64,
                                     any_dense_fiber,
                                     /*layout_cli=*/std::string(),
                                     /*ooc_cli=*/std::string("off"));

    // Report the NCopy-specific need when NCopy was chosen; otherwise
    // pick the need that matches the layout so the banner is honest.
    switch (p.choice) {
        case Layout::NCopy:     need_bytes = p.ncopy_need;    break;
        case Layout::PingPong:  need_bytes = p.pingpong_need; break;
        case Layout::InPlace:   need_bytes = p.inplace_need;  break;
        default:                need_bytes = p.ncopy_need;    break;
    }
    budget_bytes = p.budget_bytes;
    reason       = p.reason;
    return p.choice;
}

// ---------------------------------------------------------------------------
//  NCopy + CSR slab compaction (Stage 2B).
//
//  On entry:
//    * copy k has been counting-sorted into the transient T.ncopy_sort_*
//      buffer (i.e. ncopy_sort_vals + ncopy_sort_idx[0..N-1] hold the
//      permuted data, sorted by ix[k] within every super-shard).
//    * The "permanent" copy-k slab (whichever slab ncopy_vals[k] currently
//      points into) is still live but about to be freed.
//
//  On exit:
//    * T.buf_ncopy_compact[k] owns a newly allocated, cache-line-aligned
//      buffer sized for (nnz * sizeof(val)) + (N-1) * (nnz * sizeof(idx))
//      plus per-slab padding -- exactly one idx column shorter than the
//      pre-compaction slab.  Layout:
//          [vals | ix[0] | ... | ix[k-1] | ix[k+1] | ... | ix[N-1]]
//    * T.ncopy_vals[k] and T.ncopy_idx[k][m != k] point into the new
//      buffer; T.ncopy_idx[k][k] is nullptr (rowptr replaces it).
//    * T.ncopy_rowptr[k] is built from the sort-temp's sorted ix[k].
//    * The previous backing for copy k is released:
//          k == 0 -> frees T.buf_raw; nulls T.vals / T.idx_buf[*].
//          k == 1 -> frees T.buf_scratch_raw; nulls T.scr_vals / T.scr_idx.
//          k >= 2 -> frees T.buf_ncopy_raw[k - 2].
//      The specific slab is chosen by membership of ncopy_vals[k] inside
//      the primary / scratch byte ranges -- this correctly handles the
//      swap-buffers history from sort_and_count_all, where vals can end
//      up inside buf_scratch_raw (and vice-versa).
//    * T.ncopy_csr[k] = true and T.ncopy_fiber_sorted[k] = true.
//
//  Side effect on memory (per CSR-compacted copy):
//      -1 idx slab (nnz * sizeof(idx_t) bytes)
//      +1 rowptr slab ((I_k+1) * sizeof(uint64_t) bytes, tiny)
//  For 32-bit idx and typical FROSTT shapes this is a ~20-25% per-copy
//  win on 3D and ~15-20% on 4D after accounting for all N modes.
// ---------------------------------------------------------------------------
static void ncopy_csr_compact_copy_k(Tensor& T, int k, int num_threads) {
    constexpr size_t kA = DYN_ALIGN;
    const int      N       = T.num_modes;
    const uint64_t nnz     = T.nnz;
    const size_t   v_bytes = (nnz * sizeof(value_t) + kA - 1) & ~(kA - 1);
    const size_t   i_bytes = (nnz * sizeof(idx_t)   + kA - 1) & ~(kA - 1);
    const size_t   compact_need = v_bytes + (size_t)(N - 1) * i_bytes + kA;

    // 1. Allocate the compact slab.  We don't reuse buf_ncopy_raw slots
    //    because those are indexed (copy_index - 2); buf_ncopy_compact is
    //    indexed directly by copy_index and therefore works for all k.
    if (T.buf_ncopy_compact[k] &&
        T.buf_ncopy_compact_bytes[k] < compact_need) {
        dyn_aligned_free(T.buf_ncopy_compact[k]);
        T.buf_ncopy_compact[k] = nullptr;
    }
    if (!T.buf_ncopy_compact[k]) {
        T.buf_ncopy_compact[k] = dyn_aligned_alloc(compact_need);
        if (!T.buf_ncopy_compact[k]) {
            std::fprintf(stderr,
                "ncopy_csr_compact_copy_k(%d): OOM (requested %.2f MiB).\n"
                "  Consider --ncopy-csr-compact off.\n",
                k, compact_need / (1024.0 * 1024.0));
            std::exit(1);
        }
        T.buf_ncopy_compact_bytes[k] = compact_need;
        dyn_advise_huge(T.buf_ncopy_compact[k],
                        T.buf_ncopy_compact_bytes[k]);
    }

    // 2. Carve the new slab into val + (N-1) idx columns, skipping m == k.
    char* p = (char*) T.buf_ncopy_compact[k];
    value_t* new_vals = reinterpret_cast<value_t*>(p); p += v_bytes;
    idx_t*   new_idx [DYN_MAX_MODES] = {nullptr};
    for (int m = 0; m < N; ++m) {
        if (m == k) continue;
        new_idx[m] = reinterpret_cast<idx_t*>(p); p += i_bytes;
    }

    // 3. Parallel memcpy sort_temp -> compact slab.  Thread stripes share
    //    the per-thread cacheline offset so each thread's writes stay on
    //    its own NUMA page (same approach as the legacy copy-back).
    const value_t*      DYN_RESTRICT tmp_vals = T.ncopy_sort_vals;
    const idx_t* const* DYN_RESTRICT tmp_idx  = T.ncopy_sort_idx;

    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < (int64_t)num_threads; ++t) {
        const uint64_t lo = (nnz * (uint64_t)t)     / (uint64_t)num_threads;
        const uint64_t hi = (nnz * (uint64_t)(t+1)) / (uint64_t)num_threads;
        const uint64_t cnt_nnz = hi - lo;
        if (cnt_nnz == 0) continue;
        std::memcpy((void*)(new_vals + lo),
                    (const void*)(tmp_vals + lo),
                    cnt_nnz * sizeof(value_t));
        for (int m = 0; m < N; ++m) {
            if (m == k) continue;
            std::memcpy((void*)(new_idx[m] + lo),
                        (const void*)(tmp_idx[m] + lo),
                        cnt_nnz * sizeof(idx_t));
        }
    }

    // 4. Build rowptr from sort_temp's sorted ix[k] column.  Same
    //    algorithm as the Stage 2A non-compact path; the only
    //    difference is the source array (sort_temp instead of the
    //    permanent slab, which we're about to free).
    T.ensure_ncopy_rowptr(k);
    uint64_t* DYN_RESTRICT rowptr = T.ncopy_rowptr[k];
    const idx_t* DYN_RESTRICT ixn = tmp_idx[k];
    const idx_t I_k    = T.mode_size[k];
    const idx_t K_sh   = T.shards_per_mode[k];
    const int   shft_k = T.mn_shift[k];
    rowptr[I_k] = T.nnz;

    #pragma omp parallel for schedule(dynamic)
    for (int64_t s64 = 0; s64 < (int64_t)K_sh; ++s64) {
        const idx_t    s      = (idx_t)s64;
        const uint64_t b      = T.shard_begin[k][s];
        const uint64_t e      = T.shard_end  [k][s];
        const idx_t    row_lo = (idx_t)(s << shft_k);
        idx_t          row_hi = (idx_t)((s + 1) << shft_k);
        if (row_hi > I_k) row_hi = I_k;

        uint64_t i = b;
        for (idx_t r = row_lo; r < row_hi; ++r) {
            rowptr[r] = i;
            while (i < e && ixn[i] == r) ++i;
        }
        rowptr[row_hi] = i;
        (void)e;
    }

    // 5. Release the pre-compaction backing for copy k.  Uses pointer-
    //    membership to pick the correct slab: this is robust to the
    //    earlier swap_buffers() in sort_and_count_all which may have
    //    flipped vals/scr_vals across buf_raw/buf_scratch_raw.
    const value_t* stale_vals = T.ncopy_vals[k];
    auto in_range = [](const void* p, const void* base, size_t sz) -> bool {
        return base && (const char*)p >= (const char*)base
                    && (const char*)p <  (const char*)base + sz;
    };

    if (k >= 2) {
        const int slot = k - 2;
        if (T.buf_ncopy_raw[slot]) {
            dyn_aligned_free(T.buf_ncopy_raw[slot]);
            T.buf_ncopy_raw  [slot] = nullptr;
            T.buf_ncopy_bytes[slot] = 0;
        }
    } else {
        // k == 0 or k == 1: ncopy_vals[k] aliases primary OR scratch
        // depending on swap history.  Free whichever slab currently
        // contains it.
        if (in_range(stale_vals, T.buf_raw, T.buf_bytes)) {
            dyn_aligned_free(T.buf_raw);
            T.buf_raw         = nullptr;
            T.buf_bytes       = 0;
            T.buf_capacity    = 0;
            T.vals            = nullptr;
            for (int m = 0; m < DYN_MAX_MODES; ++m) T.idx_buf[m] = nullptr;
        } else if (in_range(stale_vals,
                            T.buf_scratch_raw, T.buf_scratch_bytes)) {
            dyn_aligned_free(T.buf_scratch_raw);
            T.buf_scratch_raw   = nullptr;
            T.buf_scratch_bytes = 0;
            T.scr_vals          = nullptr;
            for (int m = 0; m < DYN_MAX_MODES; ++m) T.scr_idx[m] = nullptr;
        }
        // If neither, something is very wrong; do not double-free.
    }

    // 6. Repoint ncopy_vals[k] / ncopy_idx[k][m] at the compact slab.
    T.ncopy_vals[k] = new_vals;
    for (int m = 0; m < N; ++m) {
        T.ncopy_idx[k][m] = (m == k) ? nullptr : new_idx[m];
    }
    for (int m = N; m < DYN_MAX_MODES; ++m) T.ncopy_idx[k][m] = nullptr;

    T.ncopy_fiber_sorted[k] = true;
    T.ncopy_csr[k]          = true;
}

static void decide_and_populate_layout(Tensor& T, int num_threads) {
    // Unified plan: reuse the same function main.cpp uses pre-load so the
    // banner format and candidate accounting are identical in every case.
    // Post-load we pass ooc="off" (we have already loaded; cannot downgrade)
    // and include the actual DenseFiber info.
    bool any_dense_fiber = false;
    for (int n = 0; n < T.num_modes; ++n)
        if (T.kernel_class[n] == KernelClass::DenseFiber) any_dense_fiber = true;

    StoragePlan plan = dyn_plan_storage(T.nnz, T.num_modes,
                                        T.mode_size, /*rank_padded_est=*/64,
                                        any_dense_fiber,
                                        /*layout_cli=*/std::string(),
                                        /*ooc_cli=*/std::string("off"));
    dyn_print_plan(plan, "[layout]");

    const Layout      choice = plan.choice;
    const size_t      need   = (choice == Layout::NCopy    ) ? plan.ncopy_need    :
                               (choice == Layout::PingPong ) ? plan.pingpong_need :
                                                               plan.inplace_need;
    const size_t      budget = plan.budget_bytes;
    const char* const reason = plan.reason;
    auto gib = [](size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); };

    T.layout = choice;
    if (choice == Layout::PingPong) {
        // Nothing to do: existing PingPong code path uses vals / idx_buf
        // and scr_vals / scr_idx as before.  Leave ncopy_* nulled.
        return;
    }

    if (choice == Layout::InPlace) {
        // Catastrophic OOM guard: "TENSOR DOES NOT FIT IN RAM" means even
        // a single-copy slab + factors + slack exceeds budget.  Refuse.
        if (std::string(reason).find("DOES NOT FIT") != std::string::npos) {
            std::fprintf(stderr,
                "[layout] FATAL: tensor cannot be held in available RAM "
                "even at the 1 x NNZ InPlace footprint (need=%.2f GiB, "
                "budget=%.2f GiB).\n"
                "         Remedies:\n"
                "           * --decompose cpals --ooc on   (stream from disk)\n"
                "           * free more RAM and retry\n"
                "         Aborting in-core path.\n",
                gib(need), gib(budget));
            std::exit(2);
        }
        // InPlace uses one slab only.  sort_and_count_all may have done a
        // swap_buffers, so T.vals might actually point INTO buf_scratch_raw
        // (with buf_raw holding stale pre-sort data).  Free whichever slab
        // does NOT back T.vals, and relabel the survivor as buf_raw /
        // T.vals / T.idx_buf so the kernel's primary-slab dispatch works.
        auto in_range = [](const void* p, const void* base, size_t sz) -> bool {
            return base && p && (const char*)p >= (const char*)base
                               && (const char*)p <  (const char*)base + sz;
        };
        const bool vals_in_primary = in_range(T.vals, T.buf_raw, T.buf_bytes);
        const bool vals_in_scratch =
            in_range(T.vals, T.buf_scratch_raw, T.buf_scratch_bytes);

        if (vals_in_scratch) {
            // Free buf_raw (stale), then relabel the scratch as the primary
            // so downstream code (reference kernel, SpmttkrpSweep) sees
            // T.vals / T.idx_buf backed by T.buf_raw.
            if (T.buf_raw) {
                dyn_aligned_free(T.buf_raw);
            }
            T.buf_raw         = T.buf_scratch_raw;
            T.buf_bytes       = T.buf_scratch_bytes;
            T.buf_capacity    = T.buf_scratch_bytes;
            // T.vals / T.idx_buf[*] already point into the scratch slab.
            // Null out the scratch-side bookkeeping.
            T.buf_scratch_raw   = nullptr;
            T.buf_scratch_bytes = 0;
            T.scr_vals          = nullptr;
            for (int m = 0; m < DYN_MAX_MODES; ++m) T.scr_idx[m] = nullptr;
        } else if (vals_in_primary) {
            // Normal case: T.vals is in buf_raw; just drop the scratch.
            if (T.buf_scratch_raw) {
                dyn_aligned_free(T.buf_scratch_raw);
                T.buf_scratch_raw   = nullptr;
                T.buf_scratch_bytes = 0;
            }
            T.scr_vals = nullptr;
            for (int m = 0; m < DYN_MAX_MODES; ++m) T.scr_idx[m] = nullptr;
        } else {
            // Neither matched -- leave both slabs alone to avoid a
            // use-after-free.  This should never happen in practice.
            std::fprintf(stderr,
                "[layout] InPlace: WARNING T.vals does not lie in either slab; "
                "skipping scratch release.\n");
        }
        // Allocate a single-column scratch for the parallel sort path.
        // sizeof(value_t) * nnz is enough for vals and bigger than any
        // idx column, so we reuse this one buffer for every column.
        // If allocation fails (e.g. the tensor is so huge that even
        // 8 B/nnz is unaffordable), inplace_sort_slab_by_mode will
        // detect inplace_scr_raw == nullptr and fall back to the serial
        // cycle sort.
        const char* parallel_env = std::getenv("DYN_INPLACE_PARALLEL");
        const bool  parallel_off = parallel_env && parallel_env[0] == '0';
        if (!parallel_off) {
            const size_t scr_need = sizeof(value_t) * (size_t)T.nnz;
            T.inplace_scr_raw = dyn_aligned_alloc(scr_need);
            if (T.inplace_scr_raw) {
                T.inplace_scr_bytes = scr_need;
            } else {
                std::fprintf(stderr,
                    "[layout] InPlace: WARNING unable to allocate "
                    "%.2f GiB sort scratch; falling back to serial "
                    "in-place cycle sort.\n", gib(scr_need));
                T.inplace_scr_bytes = 0;
            }
        }

        // Primary slab is already sorted by mode-0 shard after the fused
        // sort_and_count_all pass.  The first process(0) call can read it
        // directly; subsequent modes call inplace_sort_slab_by_mode().
        // Primary slab bytes = sizeof(value_t) + N * sizeof(idx_t) per nnz.
        const size_t slab_bytes =
            (sizeof(value_t) + (size_t)T.num_modes * sizeof(idx_t))
            * (size_t)T.nnz;
        std::printf(
            "[layout] InPlace: freed scratch slab; primary=%.2f GiB "
            "(1 x NNZ) + sort-scratch=%.2f GiB -> steady-state %.2f GiB. "
            " Each mode re-sorts the primary slab in-place before MTTKRP.\n",
            gib(slab_bytes),
            gib(T.inplace_scr_bytes),
            gib(slab_bytes + T.inplace_scr_bytes));
        return;
    }

    // -------------------------- NCopy: populate copies 0 .. N-1 --------
    const int N = T.num_modes;
    const double t0 = omp_get_wtime();

    // copy 0 = primary slab (already sorted by mode-0 shard after the
    // fused sort_and_count_all pass above).
    T.ensure_ncopy_slab(0);

    // copies 1..N-1: remap from copy (k-1) to copy k using mode-(k-1)'s plan.
    for (int k = 1; k < N; ++k) {
        T.ensure_ncopy_slab(k);
        dyn_remap_shard_only_all(
            T, /*from_n=*/k - 1,
            T.ncopy_vals[k - 1], T.ncopy_idx[k - 1],
            T.ncopy_vals[k],     T.ncopy_idx[k],
            num_threads);
    }

    const double t1 = omp_get_wtime();

    // ----------------------------------------------------------------
    //  NCopy + fiber combo: within-shard sort by ix[k] for every mode
    //  that the kernel classifier marked DenseFiber.  This is what the
    //  runtime fiber_sort_shards would do in PingPong mode, but here
    //  we do it ONCE at preprocess time so every ALS iteration can just
    //  dispatch the fiber kernel directly on the pre-sorted copy and
    //  skip both the per-iter sort AND the per-iter remap.
    //
    //  counting_sort_one_shard can't run in place (scatter would
    //  overwrite its own source), so we carve one transient "sort temp"
    //  slab, counting-sort the copy into it, then memcpy the sorted
    //  data back into the copy's permanent home.  The temp is freed
    //  before build_flycoo returns, so the steady-state footprint is
    //  still N x (vals + N*idx) * nnz -- exactly what the memory
    //  heuristic charged us for.
    //
    //  Opt-out: DYN_NCOPY_FIBER_SORT=0 disables the pass (useful for
    //  isolating its contribution or for debug).  Auto-skipped when no
    //  mode is DenseFiber.
    // ----------------------------------------------------------------
    int n_fiber_sort_modes = 0;
    for (int k = 0; k < N; ++k)
        if (T.kernel_class[k] == KernelClass::DenseFiber) ++n_fiber_sort_modes;

    const char* fs_env     = std::getenv("DYN_NCOPY_FIBER_SORT");
    const bool  fs_off     = fs_env && *fs_env && *fs_env == '0';
    const bool  do_fs_pass = (n_fiber_sort_modes > 0) && !fs_off;

    // CSR row-pointer construction runs piggyback on the fiber-sort pass:
    // it needs the fully-sorted ix[k] column.  Opt-in via DYN_NCOPY_CSR
    // (default = 1 when fiber-sort itself is on).
    const char* csr_env    = std::getenv("DYN_NCOPY_CSR");
    const bool  csr_off    = csr_env && *csr_env && *csr_env == '0';
    const bool  do_csr     = do_fs_pass && !csr_off;

    // Stage 2B: actually drop ix[k] from the slab by repacking copy k
    // into a new compact buffer.  Requires do_csr.  Defaults ON; the
    // CLI wrapper forces it off when --verify is active (so the
    // reference kernel can still read T.vals / T.idx_buf).
    const char* cmp_env    = std::getenv("DYN_NCOPY_CSR_COMPACT");
    const bool  cmp_off    = cmp_env && *cmp_env && *cmp_env == '0';
    const bool  do_compact = do_csr && !cmp_off;

    double t2 = t1;
    double sort_total_ms = 0.0, copy_total_ms = 0.0;
    double csr_total_ms  = 0.0;
    int    n_csr_modes   = 0;
    int    n_compact_modes = 0;
    size_t bytes_freed_by_compact = 0;
    if (do_fs_pass) {
        T.ensure_ncopy_sort_temp();

        for (int k = 0; k < N; ++k) {
            if (T.kernel_class[k] != KernelClass::DenseFiber) continue;
            const idx_t K    = T.shards_per_mode[k];
            const idx_t m_n  = T.mn[k];
            const int   shft = T.mn_shift[k];
            if (K == 0) continue;

            const double ts_k0 = omp_get_wtime();
            const value_t*      src_vals = T.ncopy_vals[k];
            const idx_t* const* src_idx  = T.ncopy_idx [k];
            value_t*            tmp_vals = T.ncopy_sort_vals;
            idx_t* const*       tmp_idx  = T.ncopy_sort_idx;
            const idx_t* DYN_RESTRICT ixn = src_idx[k];

            // Counting-sort every mode-k shard of copy k into the temp slab.
            // Safe to parallelize across shards: distinct shards write to
            // disjoint [b, e) ranges of the temp.
            #pragma omp parallel
            {
                std::vector<uint32_t> cnt;
                cnt.reserve((size_t)m_n + 1);

                #pragma omp for schedule(dynamic)
                for (int64_t s64 = 0; s64 < (int64_t)K; ++s64) {
                    const idx_t    s        = (idx_t)s64;
                    const uint64_t b        = T.shard_begin[k][s];
                    const uint64_t e        = T.shard_end  [k][s];
                    const idx_t    row_base = (idx_t)(s << shft);

                    counting_sort_one_shard(
                        src_vals, src_idx, tmp_vals, tmp_idx, N,
                        b, e, (uint32_t)m_n, cnt,
                        [ixn, row_base](uint64_t i) -> uint32_t {
                            return (uint32_t)(ixn[i] - row_base);
                        });
                }
                dyn_sfence_stores();
            }
            dyn_sfence_stores();

            const double ts_k1 = omp_get_wtime();
            sort_total_ms += (ts_k1 - ts_k0) * 1e3;

            if (do_compact) {
                // --------------------- Stage 2B: COMPACT PATH -----------
                //
                //  Skip the copy-back to the original slab entirely.
                //  Instead allocate a smaller slab (N-1 idx columns),
                //  copy sort_temp into it, build rowptr directly from
                //  sort_temp's sorted ix[k], then free the old backing
                //  (primary for k=0, scratch for k=1, private for k>=2).
                //  Net preprocess cost is one memcpy (same as legacy
                //  path) + one small alloc; the slab we'd have memcpy'd
                //  into is now released instead.
                // --------------------------------------------------------
                const double tc_k0 = omp_get_wtime();
                // Compute freed bytes before mutating state.  The "old"
                // slab size is a full copy worth: v + N*i + padding.
                // The "new" slab is v + (N-1)*i + padding.  Net win per
                // mode is roughly i_bytes (one idx column), minus the
                // rowptr slab overhead (I_k+1) * 8.
                const size_t kA = DYN_ALIGN;
                const size_t i_b = (T.nnz * sizeof(idx_t) + kA - 1) & ~(kA - 1);
                const size_t rp_b = (((size_t)T.mode_size[k] + 1)
                                     * sizeof(uint64_t) + kA - 1) & ~(kA - 1);
                bytes_freed_by_compact +=
                    (i_b > rp_b) ? (i_b - rp_b) : 0;

                ncopy_csr_compact_copy_k(T, k, num_threads);
                ++n_csr_modes;
                ++n_compact_modes;

                const double tc_k1 = omp_get_wtime();
                copy_total_ms += (tc_k1 - tc_k0) * 1e3;
            } else {
                // --------------------- Legacy COPY-BACK PATH ------------
                //
                //  Copy sorted data back into copy k's permanent slab.
                //  Parallel memcpy: distribute nnz across threads in
                //  stripes aligned to idx_t for simplicity.
                // --------------------------------------------------------
                const uint64_t nnz = T.nnz;
                #pragma omp parallel for schedule(static)
                for (int64_t t = 0; t < (int64_t)num_threads; ++t) {
                    const uint64_t lo = (nnz * (uint64_t)t)     / (uint64_t)num_threads;
                    const uint64_t hi = (nnz * (uint64_t)(t+1)) / (uint64_t)num_threads;
                    const uint64_t cnt_nnz = hi - lo;
                    if (cnt_nnz == 0) continue;
                    std::memcpy((void*)(const_cast<value_t*>(T.ncopy_vals[k]) + lo),
                                (const void*)(tmp_vals + lo),
                                cnt_nnz * sizeof(value_t));
                    for (int m = 0; m < N; ++m) {
                        std::memcpy((void*)(T.ncopy_idx[k][m] + lo),
                                    (const void*)(tmp_idx[m] + lo),
                                    cnt_nnz * sizeof(idx_t));
                    }
                }

                const double ts_k2 = omp_get_wtime();
                copy_total_ms += (ts_k2 - ts_k1) * 1e3;

                T.ncopy_fiber_sorted[k] = true;

                // Stage 2A rowptr build (non-compact path): ixn_sorted
                // is now in ncopy_idx[k][k], so we build rowptr from
                // there.  The kernel dispatch flips on ncopy_csr[k].
                if (do_csr) {
                    const double tc_k0 = omp_get_wtime();
                    T.ensure_ncopy_rowptr(k);
                    uint64_t* DYN_RESTRICT rowptr =
                        T.ncopy_rowptr[k];
                    const idx_t* DYN_RESTRICT ixn_sorted =
                        T.ncopy_idx[k][k];
                    const idx_t I_k    = T.mode_size[k];
                    const idx_t K_sh   = T.shards_per_mode[k];
                    const int   shft_k = T.mn_shift[k];
                    rowptr[I_k] = T.nnz;

                    #pragma omp parallel for schedule(dynamic)
                    for (int64_t s64 = 0; s64 < (int64_t)K_sh; ++s64) {
                        const idx_t    s      = (idx_t)s64;
                        const uint64_t b      = T.shard_begin[k][s];
                        const uint64_t e      = T.shard_end  [k][s];
                        const idx_t    row_lo = (idx_t)(s << shft_k);
                        idx_t          row_hi = (idx_t)((s + 1) << shft_k);
                        if (row_hi > I_k) row_hi = I_k;

                        uint64_t i = b;
                        for (idx_t r = row_lo; r < row_hi; ++r) {
                            rowptr[r] = i;
                            while (i < e && ixn_sorted[i] == r) ++i;
                        }
                        rowptr[row_hi] = i;
                        (void)e;
                    }
                    T.ncopy_csr[k] = true;
                    ++n_csr_modes;
                    const double tc_k1 = omp_get_wtime();
                    csr_total_ms += (tc_k1 - tc_k0) * 1e3;
                }
            }
        }

        T.free_ncopy_sort_temp();
        t2 = omp_get_wtime();
    }

    std::printf("[layout] NCopy populated in %.3f ms  "
                "(extra slabs: %d, total tensor storage: %.2f GiB)\n",
                (t1 - t0) * 1e3,
                std::max(0, N - 2),
                gib(dyn_tensor_ncopy_bytes(T.nnz, T.num_modes)));
    if (do_fs_pass) {
        std::printf("[layout] NCopy fiber-sort pass: %.3f ms across %d "
                    "DenseFiber mode(s)  (sort=%.1f ms  copy-back=%.1f ms  "
                    "transient +1 slab peak during sort)\n",
                    (t2 - t1) * 1e3, n_fiber_sort_modes,
                    sort_total_ms, copy_total_ms);
        if (n_csr_modes > 0) {
            double rowptr_mib = 0.0;
            for (int k = 0; k < N; ++k)
                if (T.ncopy_csr[k])
                    rowptr_mib +=
                        (double)T.ncopy_rowptr_bytes[k] / (1024.0 * 1024.0);
            std::printf("[layout] NCopy CSR rowptr built on %d mode(s)  "
                        "(build=%.1f ms  rowptr slab=%.2f MiB)\n",
                        n_csr_modes, csr_total_ms, rowptr_mib);
            if (n_compact_modes > 0) {
                const double freed_mib =
                    (double)bytes_freed_by_compact / (1024.0 * 1024.0);
                const double freed_gib = freed_mib / 1024.0;
                const double original_gib =
                    (double)dyn_tensor_ncopy_bytes(T.nnz, T.num_modes)
                    / (1024.0 * 1024.0 * 1024.0);
                const double savings_pct =
                    100.0 * freed_gib / std::max(original_gib, 1e-12);
                std::printf("[layout] NCopy CSR compact: %d copy(s) repacked "
                            "(dropped ix[k] + rowptr slab).  Freed %.2f MiB "
                            "(%.1f%% of uncompacted NCopy footprint).\n",
                            n_compact_modes, freed_mib, savings_pct);
            } else if (cmp_off) {
                std::printf("[layout] NCopy CSR compact SKIPPED "
                            "(DYN_NCOPY_CSR_COMPACT=0 or --verify active)  "
                            "-> copies keep full ix[k] column.\n");
            }
        } else if (csr_off) {
            std::printf("[layout] NCopy CSR rowptr SKIPPED "
                        "(DYN_NCOPY_CSR=0)  -> CSR fiber kernel "
                        "disabled; falls back to COO fiber kernel.\n");
        }
    } else if (n_fiber_sort_modes > 0) {
        std::printf("[layout] NCopy fiber-sort pass SKIPPED "
                    "(DYN_NCOPY_FIBER_SORT=0)  -> fiber kernel will fall "
                    "back to element kernel for %d mode(s).\n",
                    n_fiber_sort_modes);
    }
}

// ---------------------------------------------------------------------------
//  Public entry: build_flycoo
// ---------------------------------------------------------------------------
void build_flycoo(Tensor& T,
                  int num_threads,
                  const idx_t* mn_target,
                  idx_t shard_size)
{
    if (T.num_modes <= 0 || T.nnz == 0) {
        std::fprintf(stderr, "build_flycoo: empty tensor\n");
        std::exit(1);
    }

    if (shard_size == 0) shard_size = 16;
    T.shard_size = shard_size;

    // Commit the scratch SoA slab now -- the remap/sort passes below need
    // it.  Until this call the parse path kept peak RSS at ~half of the
    // final tensor footprint.
    T.ensure_scratch();

    // Pick power-of-2 m_n per mode and derive shift / shards_per_mode.
    for (int n = 0; n < T.num_modes; ++n) {
        idx_t mn;
        if (mn_target && mn_target[n] > 0) {
            mn = round_up_pow2(mn_target[n]);
            if (mn != mn_target[n]) {
                std::printf(
                    "[flycoo] mode %d: rounded mn %u -> %u (must be power of 2)\n",
                    n, (unsigned)mn_target[n], (unsigned)mn);
            }
        } else {
            mn = pick_mn(T.mode_size[n], num_threads);
        }
        T.mn[n]              = mn;
        T.mn_shift[n]        = log2_pow2(mn);
        T.shards_per_mode[n] = (T.mode_size[n] + mn - 1) / mn;
        if (T.shards_per_mode[n] == 0) T.shards_per_mode[n] = 1;
    }

    // ----------------------------------------------------------------
    //  Fused sort-and-count over the whole tensor.  Populates count_flat
    //  for every mode while physically sorting into mode-0 shard order.
    //  Down-stream steps read these counts and never re-scan the tensor.
    // ----------------------------------------------------------------
    std::vector<uint64_t> count_flat;
    std::vector<size_t>   cnt_off;

    const double t_sort0 = omp_get_wtime();
    sort_and_count_all(T, count_flat, cnt_off);
    const double t_sort1 = omp_get_wtime();

    // Scheduler + remap plan -- scan-free (apart from the histogram pass
    // inside build_remap_plan_from_counts).
    build_schedule_from_counts(T, count_flat, cnt_off, num_threads);
    const double t_sched = omp_get_wtime();

    build_remap_plan_from_counts(T, count_flat, cnt_off, num_threads);
    const double t_plan = omp_get_wtime();

    // --------------------------------------------------------------------
    //  Per-mode kernel classification.
    //
    //  The Yhat[n] row at index i is written by every element whose mode-n
    //  index equals i (its "fiber").  If the average fiber is long, a
    //  fiber-unrolled kernel can load each Yhat row once into SIMD regs
    //  and amortize the store across many nnz -- a big win.  If the fiber
    //  is short the element-at-a-time kernel that we ship today is
    //  already optimal and a fiber kernel would just add row-pointer
    //  overhead for no benefit.
    //
    //  The *exact* metric would count distinct fibers (requires hashing
    //  tuples of N-1 indices), but `nnz / |I_n|` is a safe upper bound
    //  on the true average and is essentially free.  It's also the
    //  metric that matters for within-shard Yhat reuse: it caps the
    //  number of distinct outRow values a single shard can touch.
    // --------------------------------------------------------------------
    int  n_dense = 0, n_medium = 0, n_sparse = 0;
    for (int n = 0; n < T.num_modes; ++n) {
        const double denom = (double)std::max<idx_t>(1u, T.mode_size[n]);
        const double afl   = (double)T.nnz / denom;
        T.avg_fiber_len[n] = afl;
        if      (afl >= DYN_FIBER_LEN_DENSE)  { T.kernel_class[n] = KernelClass::DenseFiber; ++n_dense; }
        else if (afl >= DYN_FIBER_LEN_MEDIUM) { T.kernel_class[n] = KernelClass::MediumCOO;  ++n_medium; }
        else                                  { T.kernel_class[n] = KernelClass::SparseCOO;  ++n_sparse; }
    }

    std::printf("[flycoo] threads=%d shard_size=%u (plan precomputed)\n",
                num_threads, (unsigned)shard_size);
    std::printf("[flycoo] preprocess timing: "
                "sort+count=%.3f ms  schedule=%.3f ms  remap-plan=%.3f ms  "
                "total=%.3f ms\n",
                (t_sort1 - t_sort0) * 1e3,
                (t_sched  - t_sort1) * 1e3,
                (t_plan   - t_sched) * 1e3,
                (t_plan   - t_sort0) * 1e3);
    for (int n = 0; n < T.num_modes; ++n) {
        const char* tag =
            (T.kernel_class[n] == KernelClass::DenseFiber) ? "DENSE fibers"  :
            (T.kernel_class[n] == KernelClass::MediumCOO ) ? "medium fibers" :
                                                             "sparse fibers";
        std::printf("         mode %d : I=%u  m_n=%u (shift=%d)  "
                    "super-shards=%u  avg_nnz/row=%.2f  [%s]\n",
                    n,
                    (unsigned)T.mode_size[n],
                    (unsigned)T.mn[n],
                    T.mn_shift[n],
                    (unsigned)T.shards_per_mode[n],
                    T.avg_fiber_len[n],
                    tag);
    }

    // --------------------------------------------------------------------
    //  Actionable hint for the user: tell them whether a different build
    //  would be faster for this specific tensor.  This is the feedback
    //  loop you asked for -- preprocessing says "rebuild with X" if X
    //  would win, and stays quiet otherwise.
    // --------------------------------------------------------------------
    if (n_dense > 0) {
        std::printf(
            "[hint] %d mode(s) have DENSE fibers (>= %.0f nnz/row).  "
            "A fiber-unrolled kernel can reuse Yhat rows in SIMD registers "
            "for large gains on those modes.  If/when built, rebuild with "
            "DYN_FIBER_KERNEL=1 -- the runtime dispatches per-mode using "
            "Tensor::kernel_class[].\n",
            n_dense, (double)DYN_FIBER_LEN_DENSE);
    } else if (n_medium > 0) {
        std::printf(
            "[hint] %d mode(s) have medium-length fibers "
            "(%.0f - %.0f nnz/row).  The element kernel is safe here; a "
            "fiber kernel might win 5-15%% on those modes on a "
            "bandwidth-bound target.\n",
            n_medium,
            (double)DYN_FIBER_LEN_MEDIUM,
            (double)DYN_FIBER_LEN_DENSE);
    } else {
        std::printf(
            "[hint] all modes have short fibers (< %.0f nnz/row).  "
            "Current element-at-a-time kernel is optimal for this tensor; "
            "a fiber kernel would add overhead for no benefit.  "
            "Keep the default build.\n",
            (double)DYN_FIBER_LEN_MEDIUM);
    }

    // --------------------------------------------------------------------
    //  Hybrid Layout decision and NCopy population.  At this point the
    //  primary slab holds the tensor sorted by mode-0 shard order and
    //  every precomp_thr_off[n] / shard_begin/end[n] / ss_list[n] entry
    //  is ready.  We choose between PingPong (current behavior) and
    //  NCopy (ALTO-style N pre-sorted copies, no per-iter remap) based on
    //  the tensor's memory footprint vs. available RAM.
    // --------------------------------------------------------------------
    decide_and_populate_layout(T, num_threads);
}

// ============================================================================
//  fiber_sort_shards  -- two-pass LSD-then-MSD radix sort
//
//  After this call, within each mode-n shard the nonzeros are sorted by the
//  composite key   (ix[n], ix[w_sort])   -- lexicographic, primary = mode n,
//  secondary = the "input" factor mode that the kernel gathers first
//  (w_sort = (n == 0) ? 1 : 0, matching the kernel's w0 pick).
//
//  Why two keys?
//    * Primary (ix[n]): groups every fiber into one contiguous run, so the
//      fiber kernel can amortize a single Yhat row load/store across the
//      whole fiber (unchanged from the single-pass version).
//    * Secondary (ix[w_sort]): makes the Y[w_sort] gathers monotone WITHIN
//      each fiber.  The inner loop does
//         rowY = Y[w0] + ix[w0][k] * Rp;  prod = vval * rowY;
//      repeatedly for k in [fiber_begin, fiber_end).  With the secondary
//      sort, consecutive k's touch the same or adjacent Y[w0] rows, so
//      each 64 B factor line is loaded once from L2/L3 and reused by every
//      nnz in the run -- dropping Y[w0] misses from ~100% to roughly
//      (distinct w0 values in the fiber) / (fiber length), which for
//      typical dense fibers is an order of magnitude lower.
//
//  Implementation: two back-to-back stable counting sorts per shard.
//    Pass 1 (LSD, secondary):  primary  -> scratch,   key = ix[w_sort]
//    Pass 2 (MSD, primary):    scratch -> primary,    key = ix[n] - row_base
//  Because Pass 2 is stable, the LSD order inside each MSD bucket is
//  preserved, giving the desired lex(ix[n], ix[w_sort]) ordering.  Pass 2
//  lands the result directly in primary, so no swap_buffers() is needed.
//
//  Cost model: 2 x (one scan + one scatter) per shard.  Roughly 2x the
//  legacy single-pass cost, but typically <3% of an ALS iteration; the
//  kernel-side gather speedup pays for it many times over on dense-fiber
//  modes.
//
//  Opt-in: the two-pass path runs only when DYN_SECONDARY_SORT=1 is set
//  in the environment (or I[w_sort] is small enough to keep cnt[] cache-
//  resident).  Default is the legacy 1-pass sort with a trailing
//  swap_buffers(), which matches the pre-secondary-sort behavior exactly
//  and is the safe choice for uniform / well-prefetched workloads.
//
//  Scratch usage:
//    Pass 1 writes scr_*.  Pass 2 writes primary.  After return, scr_* is
//    dirty intermediate data that the kernel's own remap pass overwrites
//    on its next invocation -- no conflict.
// ============================================================================

// Maximum |I_{w_sort}| for which the per-thread cnt[] histogram stays
// comfortably in L2 (uint32 array, 4 B * domain).  Above this the secondary
// pass can hurt more than it helps, so we fall back to the legacy 1-pass
// sort.  Tunable via env at startup; default chosen so cnt <= 8 MiB.
#ifndef DYN_SECONDARY_SORT_MAX_DOMAIN
#define DYN_SECONDARY_SORT_MAX_DOMAIN ((idx_t)(2u * 1024u * 1024u))
#endif

// See counting_sort_one_shard defined higher up in this TU (hoisted so
// the NCopy populate pass can use it before fiber_sort_shards is defined).

// Legacy single-pass fiber sort -- used as fallback when the secondary
// pass is disabled or infeasible.  Identical behavior to the previous
// fiber_sort_shards (primary sort only, ends with swap_buffers()).
static void fiber_sort_shards_single_pass(Tensor& T, int n) {
    const idx_t  K    = T.shards_per_mode[n];
    const int    NM   = T.num_modes;
    const idx_t  m_n  = T.mn[n];
    const int    shft = T.mn_shift[n];
    const idx_t* DYN_RESTRICT ixn = T.idx_buf[n];

    #pragma omp parallel
    {
        std::vector<uint32_t> cnt;
        cnt.reserve((size_t)m_n + 1);

        #pragma omp for schedule(dynamic)
        for (int64_t s64 = 0; s64 < (int64_t)K; ++s64) {
            const idx_t    s        = (idx_t)s64;
            const uint64_t b        = T.shard_begin[n][s];
            const uint64_t e        = T.shard_end  [n][s];
            const idx_t    row_base = (idx_t)(s << shft);

            counting_sort_one_shard(
                T.vals, T.idx_buf, T.scr_vals, T.scr_idx, NM,
                b, e, (uint32_t)m_n, cnt,
                [ixn, row_base](uint64_t i) -> uint32_t {
                    return (uint32_t)(ixn[i] - row_base);
                });
        }
        dyn_sfence_stores();
    }

    dyn_sfence_stores();
    T.swap_buffers();
}

void fiber_sort_shards(Tensor& T, int n) {
    const idx_t K     = T.shards_per_mode[n];
    const int   NM    = T.num_modes;
    const idx_t m_n   = T.mn[n];
    const int   shft  = T.mn_shift[n];

    if (K == 0 || T.nnz == 0) return;

    // --------------------------------------------------------------- dispatch
    // The two-pass (primary, secondary) radix sort is OPT-IN.  Rationale:
    //
    //   * Cost of the secondary pass is deterministic -- roughly 2x the
    //     legacy single-pass sort.  On a 335M-nnz 3-mode uniform tensor at
    //     R=16 / 2 threads, that adds ~2-4 s per mode-MTTKRP call, or
    //     ~60-120 s across a 10-iter ALS run.
    //
    //   * Benefit of the secondary pass is highly tensor-dependent.  It
    //     pays off when Y[w_sort] is too big to fit in L2 AND hardware +
    //     software prefetch cannot hide the miss latency, i.e. on:
    //        - large-I real FROSTT tensors (amazon-reviews, patents, etc.)
    //        - high rank (R >= 64)
    //        - power-law index distributions where a small set of w_sort
    //          rows is reused heavily within each fiber
    //     On uniform / well-prefetched workloads the software prefetch
    //     already hides the L2 latency, so the secondary sort is a net
    //     loss.
    //
    // Default = OFF.  Enable with DYN_SECONDARY_SORT=1 when benchmarking
    // on tensors where factor-row locality is the bottleneck.  The legacy
    // fast path is always a safe, known-good fallback.
    const char* on_env     = std::getenv("DYN_SECONDARY_SORT");
    const bool  force_on   = on_env && *on_env && *on_env != '0';

    if (!force_on || NM < 2) {
        fiber_sort_shards_single_pass(T, n);
        return;
    }

    const int   w_sort = (n == 0) ? 1 : 0;    // matches kernel's w0
    const idx_t I_w    = T.mode_size[w_sort];

    if (I_w == 0 || I_w > DYN_SECONDARY_SORT_MAX_DOMAIN) {
        // Hard guard: cnt-array too large to stay cache-resident.
        fiber_sort_shards_single_pass(T, n);
        return;
    }

    const idx_t* DYN_RESTRICT ixs_prim = T.idx_buf[w_sort];

    // ------------------------------------------------------------ Pass 1 (LSD)
    // Sort each shard by ix[w_sort].  Reads primary, writes scratch.  After
    // this pass, within each shard the nonzeros are in ascending ix[w_sort]
    // order.  Shard boundaries (shard_begin/end[n][s]) are preserved because
    // the scatter stays inside each [b, e).
    #pragma omp parallel
    {
        std::vector<uint32_t> cnt;
        cnt.reserve((size_t)I_w + 1);

        #pragma omp for schedule(dynamic)
        for (int64_t s64 = 0; s64 < (int64_t)K; ++s64) {
            const idx_t    s = (idx_t)s64;
            const uint64_t b = T.shard_begin[n][s];
            const uint64_t e = T.shard_end  [n][s];

            counting_sort_one_shard(
                T.vals, T.idx_buf, T.scr_vals, T.scr_idx, NM,
                b, e, (uint32_t)I_w, cnt,
                [ixs_prim](uint64_t i) -> uint32_t {
                    return (uint32_t)ixs_prim[i];
                });
        }
        dyn_sfence_stores();
    }
    dyn_sfence_stores();

    // ------------------------------------------------------------ Pass 2 (MSD)
    // Stable sort by (ix[n] - row_base).  Reads scratch (Pass 1 output),
    // writes primary.  Stability preserves the Pass 1 secondary order within
    // each primary bucket, so the final layout in T.vals / T.idx_buf is
    // sorted by lex(ix[n], ix[w_sort]) within each shard.  Because Pass 2
    // lands in primary, no swap_buffers() is required.
    const idx_t* DYN_RESTRICT ixn_scr = T.scr_idx[n];

    #pragma omp parallel
    {
        std::vector<uint32_t> cnt;
        cnt.reserve((size_t)m_n + 1);

        #pragma omp for schedule(dynamic)
        for (int64_t s64 = 0; s64 < (int64_t)K; ++s64) {
            const idx_t    s        = (idx_t)s64;
            const uint64_t b        = T.shard_begin[n][s];
            const uint64_t e        = T.shard_end  [n][s];
            const idx_t    row_base = (idx_t)(s << shft);

            counting_sort_one_shard(
                T.scr_vals, T.scr_idx, T.vals, T.idx_buf, NM,
                b, e, (uint32_t)m_n, cnt,
                [ixn_scr, row_base](uint64_t i) -> uint32_t {
                    return (uint32_t)(ixn_scr[i] - row_base);
                });
        }
        dyn_sfence_stores();
    }
    dyn_sfence_stores();

    // NOTE: no swap_buffers().  Primary already holds the final sorted data;
    // scratch holds Pass 1's intermediate, which the kernel's remap will
    // overwrite on its very next write.
}

} // namespace dynasor
