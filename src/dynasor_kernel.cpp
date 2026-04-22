// ============================================================================
//  dynasor_kernel.cpp
//
//  Vectorized elementwise spMTTKRP kernel (Equation 4 of the paper) plus
//  dynamic tensor remapping (Section III-B), operating on the SoA tensor
//  layout.
//
//  Tier-2 optimizations:
//    * Template specialization on (num_modes, rank_padded, target_mode)
//      so the inner per-mode and per-SIMD-chunk loops fully unroll and
//      the `w == n`, `w == w0`, `w == w_last` branches fold away at
//      compile time.  Dispatch grid is N in {3,4,5} x Rp in {16,32,64,
//      128,256} x tgt in [0,N-1], plus a runtime fallback for shapes
//      outside the grid.
//    * The FINAL factor-matrix multiply is fused with the Yhat
//      accumulation via `vfma` (one fewer vmul per SIMD chunk per
//      element relative to the old vmul-chain + vadd pattern).
//    * Software prefetch of the factor rows for element i + PF_DIST
//      while the current FMA chain executes.
//    * No `val == 0` early-out branch: the compute is unconditional and
//      zero-valued entries propagate through the FMA chain harmlessly.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_simd.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dynasor {

const char* simd_backend_name() { return DYN_SIMD_NAME; }

// Two-tier software-prefetch distances (in nnz units).
//
//   DYN_PF_DIST      : short.  Brings the factor row into L1 just in
//                      time for the FMA chain that will consume it.
//                      4-8 is right for the typical 64-512 B row size
//                      and the ~40-80 cycle FMA chain per nnz on Zen
//                      4/5 and Neoverse V2.
//   DYN_PF_DIST_FAR  : long.  Issues a T2/PLDL2KEEP hint far enough
//                      ahead that the DRAM->L2 fill completes before
//                      the corresponding T0 lands, which makes the T0
//                      a pure L2->L1 transfer instead of a cold miss.
//                      16 covers ~150-cycle DRAM latency at ~10 cy/nnz
//                      FMA throughput; users can raise it via
//                      DYN_PF_DIST_FAR if their DRAM is slower.
static constexpr uint64_t DYN_PF_DIST     = 4;
static constexpr uint64_t DYN_PF_DIST_FAR = 16;

// Env toggle for the L2-tier far prefetch.  DEFAULT OFF because on
// moderate-scale tensors (10M-ish nnz, Rp=128) the extra T2 hints
// flood the Line-Fill Buffers / miss-status handlers and contend with
// the near T0 stream, producing a net regression.  The trick is only
// profitable when factor matrices overflow L3 AND the random-index
// access pattern forces cold DRAM misses on every nnz.  For those
// workloads set DYN_PF_FAR=1 at the command line.  A/B tested on
// bench_3d_10M / bench_4d_10M at rank 128: default-off is +20..40%
// faster than default-on.
static inline bool dyn_pf_far_on() {
    static const bool on = [] {
        const char* s = std::getenv("DYN_PF_FAR");
        return s && s[0] && s[0] != '0';
    }();
    return on;
}

// Maximum vector-chunks per row the register-resident product can hold.
// With cap = 64 we cover rank_padded up to 64 * DYN_SIMD_WIDTH, i.e.
// 1024 on AVX-512 / 512 on AVX2 / 256 on NEON.  The compiler is free to
// spill `regs[]` to stack for large chunk counts; the template variants
// actually used by the dispatcher (<= 256) hold everything in registers
// on Zen 4/5 (32 ZMM / 32 YMM) and spill only the tail on 16-reg NEON.
static constexpr int DYN_MAX_VEC_CHUNKS = 64;

// Lines per factor-matrix row -- used to decide how many prefetches to
// issue per row when rank_padded spans multiple cache lines.
static inline int dyn_pf_lines_per_row(int rank_padded) {
    return (rank_padded * (int)sizeof(value_t) + 63) / 64;
}

// ---------------------------------------------------------------------------
//  Core templated kernel.  N_TMPL==0 / RP_TMPL==0 means "use the runtime
//  argument"; otherwise the loop bounds are compile-time constants and
//  the per-mode / per-chunk loops unroll cleanly.  TGT_TMPL>=0 pins the
//  target mode `n` at compile time, allowing the `w == n` and `w == w0`
//  branches in the per-mode loops to be folded away.  TGT_TMPL==-1
//  leaves `n` runtime.
// ---------------------------------------------------------------------------
template<int N_TMPL, int RP_TMPL, int TGT_TMPL>
static inline void process_range_T(
    const value_t* DYN_RESTRICT vals,
    const idx_t*   const*       idx,
    uint64_t b, uint64_t e,
    int n_rt,
    int num_modes_rt,
    int rank_padded_rt,
    const value_t* const* DYN_RESTRICT Y,
    value_t* DYN_RESTRICT Yhat_n,
    int next_mode, int shift_next,
    value_t* DYN_RESTRICT scr_vals,
    idx_t*   const*       scr_idx,
    uint64_t* DYN_RESTRICT offsets_local)
{
    constexpr bool kHasN   = (N_TMPL   > 0);
    constexpr bool kHasRP  = (RP_TMPL  > 0);
    constexpr bool kHasTgt = (TGT_TMPL >= 0);
    const int N  = kHasN  ? N_TMPL  : num_modes_rt;
    const int Rp = kHasRP ? RP_TMPL : rank_padded_rt;
    const int VC = Rp / DYN_SIMD_WIDTH;

    // Target mode.  When TGT_TMPL >= 0 both `n` and `w0` become compile-
    // time integer constants, so the per-mode `if (w == n || w == w0)`
    // checks inside the (already-unrolled) w-loop collapse away.
    const int n  = kHasTgt ? TGT_TMPL                          : n_rt;
    const int w0 = kHasTgt ? ((TGT_TMPL == 0) ? 1 : 0)         : ((n_rt == 0) ? 1 : 0);

    // Last non-n, non-w0 mode, used to fuse into the Yhat accumulate.
    // -1 iff N == 2 (in which case there are no middle factors).  When
    // N and n/w0 are all compile-time, GCC folds this loop to a
    // constant.
    int w_last = -1;
    for (int w = N - 1; w >= 0; --w) {
        if (w != n && w != w0) { w_last = w; break; }
    }

    const idx_t* const DYN_RESTRICT ix_n    = idx[n];
    const idx_t* const DYN_RESTRICT ix_next = idx[next_mode];

    const int pf_lines = dyn_pf_lines_per_row(Rp);
    // Loop-invariant toggle; LICM + branch-prediction fold this out of
    // the steady-state path at negligible cost.
    const bool pf_far_on = dyn_pf_far_on();

    for (uint64_t i = b; i < e; ++i) {
        // ----- software prefetch (tiered: T0 near + T2 far) ---------------
        // T0 brings the factor row into L1 just before the FMA chain
        // consumes it.  T2 at a longer distance kicks off the DRAM->L2
        // fill so the T0 is a near-L2 hit instead of a cold miss.
        if (DYN_LIKELY(i + DYN_PF_DIST < e)) {
            const uint64_t pi = i + DYN_PF_DIST;
            for (int w = 0; w < N; ++w) {
                const value_t* pf_row = Y[w] + (size_t)idx[w][pi] * Rp;
                for (int k = 0; k < pf_lines; ++k)
                    dyn_prefetch(pf_row + k * (64 / (int)sizeof(value_t)));
            }
        }
        if (pf_far_on && DYN_LIKELY(i + DYN_PF_DIST_FAR < e)) {
            const uint64_t pi = i + DYN_PF_DIST_FAR;
            for (int w = 0; w < N; ++w) {
                const value_t* pf_row = Y[w] + (size_t)idx[w][pi] * Rp;
                for (int k = 0; k < pf_lines; ++k)
                    dyn_prefetch_l2(pf_row + k * (64 / (int)sizeof(value_t)));
            }
        }

        const value_t   val  = vals[i];
        const dyn_vec_t vval = dyn_vset1(val);

        // Running product regs[c] = val * prod_{w in {w0}} Y[w][idx_w, c]
        // (step 0), then multiplied through by the middle factors, and
        // finally FMA'd with the last factor into Yhat.
        dyn_vec_t regs[DYN_MAX_VEC_CHUNKS];

        // Step 0: init with val * Y[w0]
        {
            const value_t* rowY = Y[w0] + (size_t)idx[w0][i] * Rp;
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t y = dyn_vload(rowY + c * DYN_SIMD_WIDTH);
                regs[c] = dyn_vmul(vval, y);
            }
        }

        // Step 1: middle mul chain (all w except n, w0, w_last).
        for (int w = 0; w < N; ++w) {
            if (w == n || w == w0 || w == w_last) continue;
            const value_t* rY = Y[w] + (size_t)idx[w][i] * Rp;
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                regs[c] = dyn_vmul(regs[c], y);
            }
        }

        // Step 2: fused final FMA into Yhat_n[idx_n, :].
        value_t* const outRow = Yhat_n + (size_t)ix_n[i] * Rp;
        if (w_last >= 0) {
            const value_t* rY = Y[w_last] + (size_t)idx[w_last][i] * Rp;
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t y = dyn_vload(rY     + c * DYN_SIMD_WIDTH);
                dyn_vec_t o = dyn_vload(outRow + c * DYN_SIMD_WIDTH);
                o = dyn_vfma(regs[c], y, o);
                dyn_vstore(outRow + c * DYN_SIMD_WIDTH, o);
            }
        } else {
            // N == 2: no middle factors.
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t o = dyn_vload(outRow + c * DYN_SIMD_WIDTH);
                dyn_vstore(outRow + c * DYN_SIMD_WIDTH,
                           dyn_vadd(o, regs[c]));
            }
        }

        // ---------------------------------- dynamic tensor remapping (SoA)
        //
        // Write-once stream: these slots are not read again until the next
        // mode iteration begins, by which time every cache line we touch
        // here has been evicted by the FMA pipeline above.  NT scalar
        // stores skip the Read-For-Ownership on each line and halve the
        // effective write bandwidth we pay.  A global sfence is issued
        // after the OpenMP parallel region in the driver, before the
        // next consumer reads scr_*.
        //
        // NCopy layout short-circuit: when scr_vals == nullptr the driver
        // has pre-sorted N physical copies of the tensor and does NOT
        // need a per-iter remap.  The branch is loop-invariant (identical
        // for every nnz in the shard), so GCC peels it out via LICM and
        // the hot path sees zero overhead.  We ALSO skip offsets_local[]
        // bumping since the caller passes nullptr cursors in that mode.
        if (scr_vals) {
            const sid_t    sid_next = (sid_t)(ix_next[i] >> shift_next);
            const uint64_t d        = offsets_local[sid_next]++;
            dyn_stream_f32(scr_vals + d, val);
            #pragma GCC unroll 8
            for (int w = 0; w < N; ++w)
                dyn_stream_u32((uint32_t*)&scr_idx[w][d], (uint32_t)idx[w][i]);
        }
    }
}

// ---------------------------------------------------------------------------
//  Public entry.  Dispatches on (num_modes, rank_padded, target_mode) to
//  a tuned specialization or a fully-runtime fallback.
//
//  Specializations:  N  in {3, 4, 5}
//                    Rp in {16, 32, 64, 128, 256}
//                    n  in [0, N-1]
//  Fallback:         N=0, Rp=0, TGT=-1 (fully runtime-parameterized)
//
//  With N_TMPL, RP_TMPL, and TGT_TMPL all known, GCC unrolls the per-
//  mode loop (3, 4, or 5 trips -- each predicated on a compile-time
//  `w == n` / `w == w0` comparison that collapses) and the per-chunk
//  loop (1..16 trips).  The resulting .text includes a dedicated
//  function body for every supported (N, Rp, n) triple; tensors outside
//  the grid (e.g. 2D, 6D, Rp=48) fall back to the runtime variant.
// ---------------------------------------------------------------------------
#define DYN_CALL(N_T, RP_T, TGT_T) \
    process_range_T<N_T, RP_T, TGT_T>(vals, idx, b, e, n,                   \
                                      num_modes, rank_padded,               \
                                      Y, Yhat_n, next_mode, shift_next,     \
                                      scr_vals, scr_idx, offsets_local)

#define DYN_DISPATCH_TGT(N_T, RP_T)                                          \
    do {                                                                     \
        if (n == 0) { DYN_CALL(N_T, RP_T, 0); return; }                      \
        if (n == 1) { DYN_CALL(N_T, RP_T, 1); return; }                      \
        if (n == 2) { DYN_CALL(N_T, RP_T, 2); return; }                      \
        if constexpr ((N_T) >= 4) {                                          \
            if (n == 3) { DYN_CALL(N_T, RP_T, 3); return; }                  \
        }                                                                    \
        if constexpr ((N_T) >= 5) {                                          \
            if (n == 4) { DYN_CALL(N_T, RP_T, 4); return; }                  \
        }                                                                    \
        DYN_CALL(N_T, RP_T, -1); return;                                     \
    } while (0)

#define DYN_DISPATCH_RP(N_T)                                      \
    do {                                                          \
        if (rank_padded == 16)  DYN_DISPATCH_TGT(N_T, 16);        \
        if (rank_padded == 32)  DYN_DISPATCH_TGT(N_T, 32);        \
        if (rank_padded == 64)  DYN_DISPATCH_TGT(N_T, 64);        \
        if (rank_padded == 128) DYN_DISPATCH_TGT(N_T, 128);       \
        if (rank_padded == 256) DYN_DISPATCH_TGT(N_T, 256);       \
        DYN_CALL(N_T, 0, -1); return;                             \
    } while (0)

// Env toggle to disable target-mode template specialization at runtime.
// Set DYN_TGT_SPECIALIZE=0 to force the (N, Rp) template only, runtime n.
// Primarily for A/B measurement of the Stage-1 delta; leave unset (=1) in
// normal runs.
static inline bool dyn_tgt_specialize_on() {
    static const bool on = [] {
        const char* s = std::getenv("DYN_TGT_SPECIALIZE");
        return !(s && s[0] == '0');
    }();
    return on;
}

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
    uint64_t* offsets_local)
{
    (void)rank;

    if (DYN_UNLIKELY(!dyn_tgt_specialize_on())) {
        // A/B fallback: runtime n, no TGT specialization.
        if (num_modes == 3) {
            if (rank_padded == 16)  { DYN_CALL(3, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL(3, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL(3, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL(3, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL(3, 256, -1); return; }
            DYN_CALL(3, 0, -1); return;
        }
        if (num_modes == 4) {
            if (rank_padded == 16)  { DYN_CALL(4, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL(4, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL(4, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL(4, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL(4, 256, -1); return; }
            DYN_CALL(4, 0, -1); return;
        }
        if (num_modes == 5) {
            if (rank_padded == 16)  { DYN_CALL(5, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL(5, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL(5, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL(5, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL(5, 256, -1); return; }
            DYN_CALL(5, 0, -1); return;
        }
        DYN_CALL(0, 0, -1);
        return;
    }

    if (num_modes == 3) DYN_DISPATCH_RP(3);
    if (num_modes == 4) DYN_DISPATCH_RP(4);
    if (num_modes == 5) DYN_DISPATCH_RP(5);
    DYN_CALL(0, 0, -1);
}

#undef DYN_CALL
#undef DYN_DISPATCH_RP
#undef DYN_DISPATCH_TGT

// ============================================================================
//  Fiber kernel variant  (DenseFiber modes)
//
//  Preconditions:
//    * The caller has already invoked fiber_sort_shards(T, n) so that
//      nonzeros within each shard are sorted by idx[n].  Consequently,
//      every fiber -- the set of nonzeros sharing a Yhat row -- forms a
//      contiguous run in [b, e).
//
//  Why it wins over the element kernel:
//    * Yhat[row] is loaded ONCE at fiber start (not once per nonzero).
//    * Yhat[row] is stored ONCE at fiber end       (not once per nonzero).
//    * Across a fiber of length F:
//         element kernel: F loads + F stores + F FMAs on Yhat
//         fiber  kernel: 1 load  + 1 store  + F FMAs on Yhat
//      For dense modes (F >= 32) this is a ~30x reduction in Yhat
//      memory traffic.  FMA throughput itself is unchanged.
//
//  The remap scatter at the tail of the inner loop is unchanged from the
//  element kernel (NT scalar stores into scr_*).
// ============================================================================
template<int N_TMPL, int RP_TMPL, int TGT_TMPL>
static inline void process_range_fiber_T(
    const value_t* DYN_RESTRICT vals,
    const idx_t*   const*       idx,
    uint64_t b, uint64_t e,
    int n_rt,
    int num_modes_rt,
    int rank_padded_rt,
    const value_t* const* DYN_RESTRICT Y,
    value_t* DYN_RESTRICT Yhat_n,
    int next_mode, int shift_next,
    value_t* DYN_RESTRICT scr_vals,
    idx_t*   const*       scr_idx,
    uint64_t* DYN_RESTRICT offsets_local)
{
    constexpr bool kHasN   = (N_TMPL   > 0);
    constexpr bool kHasRP  = (RP_TMPL  > 0);
    constexpr bool kHasTgt = (TGT_TMPL >= 0);
    const int N  = kHasN  ? N_TMPL  : num_modes_rt;
    const int Rp = kHasRP ? RP_TMPL : rank_padded_rt;
    const int VC = Rp / DYN_SIMD_WIDTH;

    const int n  = kHasTgt ? TGT_TMPL                          : n_rt;
    const int w0 = kHasTgt ? ((TGT_TMPL == 0) ? 1 : 0)         : ((n_rt == 0) ? 1 : 0);
    int w_last = -1;
    for (int w = N - 1; w >= 0; --w) {
        if (w != n && w != w0) { w_last = w; break; }
    }

    const idx_t* const DYN_RESTRICT ix_n    = idx[n];
    const idx_t* const DYN_RESTRICT ix_next = idx[next_mode];
    const int pf_lines = dyn_pf_lines_per_row(Rp);
    const bool pf_far_on = dyn_pf_far_on();

    uint64_t i = b;
    while (i < e) {
        const idx_t row = ix_n[i];

        // Find fiber end: first index whose idx[n] differs from `row`.
        // All nonzeros within [i, j) contribute to Yhat[row].
        uint64_t j = i + 1;
        while (j < e && ix_n[j] == row) ++j;

        // Load Yhat row into SIMD register file once per fiber.
        value_t* const outRow = Yhat_n + (size_t)row * Rp;
        dyn_vec_t acc[DYN_MAX_VEC_CHUNKS];
        for (int c = 0; c < VC; ++c)
            acc[c] = dyn_vload(outRow + c * DYN_SIMD_WIDTH);

        // Accumulate every nonzero in the fiber into `acc`.
        for (uint64_t k = i; k < j; ++k) {
            // Tiered software prefetch (T0 near + T2 far) to hide both
            // L2->L1 and DRAM->L2 latencies on the random factor-row
            // access pattern that HW prefetchers can't catch.
            if (DYN_LIKELY(k + DYN_PF_DIST < e)) {
                const uint64_t pk = k + DYN_PF_DIST;
                for (int w = 0; w < N; ++w) {
                    const value_t* pf = Y[w] + (size_t)idx[w][pk] * Rp;
                    for (int l = 0; l < pf_lines; ++l)
                        dyn_prefetch(pf + l * (64 / (int)sizeof(value_t)));
                }
            }
            if (pf_far_on && DYN_LIKELY(k + DYN_PF_DIST_FAR < e)) {
                const uint64_t pk = k + DYN_PF_DIST_FAR;
                for (int w = 0; w < N; ++w) {
                    const value_t* pf = Y[w] + (size_t)idx[w][pk] * Rp;
                    for (int l = 0; l < pf_lines; ++l)
                        dyn_prefetch_l2(pf + l * (64 / (int)sizeof(value_t)));
                }
            }

            const value_t   val  = vals[k];
            const dyn_vec_t vval = dyn_vset1(val);

            // Running product regs[c] = val * Y[w0][..][c] ; then chain.
            dyn_vec_t prod[DYN_MAX_VEC_CHUNKS];
            {
                const value_t* rowY = Y[w0] + (size_t)idx[w0][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rowY + c * DYN_SIMD_WIDTH);
                    prod[c] = dyn_vmul(vval, y);
                }
            }
            for (int w = 0; w < N; ++w) {
                if (w == n || w == w0 || w == w_last) continue;
                const value_t* rY = Y[w] + (size_t)idx[w][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                    prod[c] = dyn_vmul(prod[c], y);
                }
            }

            // FMA into the fiber accumulator -- NO Yhat load/store inside
            // the inner fiber loop; that's the whole point of this kernel.
            if (w_last >= 0) {
                const value_t* rY = Y[w_last] + (size_t)idx[w_last][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                    acc[c] = dyn_vfma(prod[c], y, acc[c]);
                }
            } else {
                // N == 2: no middle factors.
                for (int c = 0; c < VC; ++c)
                    acc[c] = dyn_vadd(acc[c], prod[c]);
            }

            // Remap scatter (NT stores, identical to element kernel).
            // Skipped in NCopy layout -- see element-kernel note above.
            if (scr_vals) {
                const sid_t    sid_next = (sid_t)(ix_next[k] >> shift_next);
                const uint64_t d        = offsets_local[sid_next]++;
                dyn_stream_f32(scr_vals + d, val);
                #pragma GCC unroll 8
                for (int w = 0; w < N; ++w)
                    dyn_stream_u32((uint32_t*)&scr_idx[w][d], (uint32_t)idx[w][k]);
            }
        }

        // One-shot store of the fiber accumulator back into Yhat[row].
        for (int c = 0; c < VC; ++c)
            dyn_vstore(outRow + c * DYN_SIMD_WIDTH, acc[c]);

        i = j;
    }
}

#define DYN_CALL_FIBER(N_T, RP_T, TGT_T) \
    process_range_fiber_T<N_T, RP_T, TGT_T>(vals, idx, b, e, n,             \
                                            num_modes, rank_padded,         \
                                            Y, Yhat_n, next_mode, shift_next, \
                                            scr_vals, scr_idx, offsets_local)

#define DYN_DISPATCH_TGT_FIBER(N_T, RP_T)                                    \
    do {                                                                     \
        if (n == 0) { DYN_CALL_FIBER(N_T, RP_T, 0); return; }                \
        if (n == 1) { DYN_CALL_FIBER(N_T, RP_T, 1); return; }                \
        if (n == 2) { DYN_CALL_FIBER(N_T, RP_T, 2); return; }                \
        if constexpr ((N_T) >= 4) {                                          \
            if (n == 3) { DYN_CALL_FIBER(N_T, RP_T, 3); return; }            \
        }                                                                    \
        if constexpr ((N_T) >= 5) {                                          \
            if (n == 4) { DYN_CALL_FIBER(N_T, RP_T, 4); return; }            \
        }                                                                    \
        DYN_CALL_FIBER(N_T, RP_T, -1); return;                               \
    } while (0)

#define DYN_DISPATCH_RP_FIBER(N_T)                                \
    do {                                                          \
        if (rank_padded == 16)  DYN_DISPATCH_TGT_FIBER(N_T, 16);  \
        if (rank_padded == 32)  DYN_DISPATCH_TGT_FIBER(N_T, 32);  \
        if (rank_padded == 64)  DYN_DISPATCH_TGT_FIBER(N_T, 64);  \
        if (rank_padded == 128) DYN_DISPATCH_TGT_FIBER(N_T, 128); \
        if (rank_padded == 256) DYN_DISPATCH_TGT_FIBER(N_T, 256); \
        DYN_CALL_FIBER(N_T, 0, -1); return;                       \
    } while (0)

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
    uint64_t* offsets_local)
{
    (void)rank;

    if (DYN_UNLIKELY(!dyn_tgt_specialize_on())) {
        if (num_modes == 3) {
            if (rank_padded == 16)  { DYN_CALL_FIBER(3, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL_FIBER(3, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL_FIBER(3, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL_FIBER(3, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL_FIBER(3, 256, -1); return; }
            DYN_CALL_FIBER(3, 0, -1); return;
        }
        if (num_modes == 4) {
            if (rank_padded == 16)  { DYN_CALL_FIBER(4, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL_FIBER(4, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL_FIBER(4, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL_FIBER(4, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL_FIBER(4, 256, -1); return; }
            DYN_CALL_FIBER(4, 0, -1); return;
        }
        if (num_modes == 5) {
            if (rank_padded == 16)  { DYN_CALL_FIBER(5, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL_FIBER(5, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL_FIBER(5, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL_FIBER(5, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL_FIBER(5, 256, -1); return; }
            DYN_CALL_FIBER(5, 0, -1); return;
        }
        DYN_CALL_FIBER(0, 0, -1);
        return;
    }

    if (num_modes == 3) DYN_DISPATCH_RP_FIBER(3);
    if (num_modes == 4) DYN_DISPATCH_RP_FIBER(4);
    if (num_modes == 5) DYN_DISPATCH_RP_FIBER(5);
    DYN_CALL_FIBER(0, 0, -1);
}

#undef DYN_CALL_FIBER
#undef DYN_DISPATCH_RP_FIBER
#undef DYN_DISPATCH_TGT_FIBER

// ============================================================================
//  Fiber kernel, CSR variant  (DenseFiber modes, NCopy layout only)
//
//  Preconditions (enforced by decide_and_populate_layout + SpmttkrpSweep):
//    * layout == NCopy and ncopy_csr[n] == true.
//    * ncopy_rowptr[n] is a valid (mode_size[n] + 1)-length uint64_t array.
//    * The copy is sorted by ix[n] within every super-shard (the invariant
//      that gives `rowptr[row .. row+1)` covering exactly fiber row).
//    * No remap scatter: NCopy never writes to scr_*.
//
//  Differences vs process_range_fiber_T:
//    * Outer loop is a direct row enumeration over [row_b, row_e); we do
//      NOT scan ix_n to locate fiber boundaries.
//    * The prefetch loop skips w == n -- the Yhat row is already loaded
//      into `acc[]` at fiber start, so Y[n][row] is cache-resident.  This
//      also lets a Stage-2B compaction drop ix[n] entirely without the
//      kernel needing to know.
//    * No `scr_vals` branch: runtime-guaranteed nullptr in NCopy, and the
//      remap is intentionally not supported by this variant.
//    * The "find fiber end" while loop is gone; the work per fiber drops
//      by one branch and one cacheline read (ix_n[i]) per nnz.
// ============================================================================
template<int N_TMPL, int RP_TMPL, int TGT_TMPL>
static inline void process_range_fiber_csr_T(
    const value_t* DYN_RESTRICT vals,
    const idx_t*   const*       idx,
    const uint64_t* DYN_RESTRICT rowptr,
    idx_t row_b, idx_t row_e,
    int n_rt,
    int num_modes_rt,
    int rank_padded_rt,
    const value_t* const* DYN_RESTRICT Y,
    value_t* DYN_RESTRICT Yhat_n)
{
    constexpr bool kHasN   = (N_TMPL   > 0);
    constexpr bool kHasRP  = (RP_TMPL  > 0);
    constexpr bool kHasTgt = (TGT_TMPL >= 0);
    const int N  = kHasN  ? N_TMPL  : num_modes_rt;
    const int Rp = kHasRP ? RP_TMPL : rank_padded_rt;
    const int VC = Rp / DYN_SIMD_WIDTH;

    const int n  = kHasTgt ? TGT_TMPL                          : n_rt;
    const int w0 = kHasTgt ? ((TGT_TMPL == 0) ? 1 : 0)         : ((n_rt == 0) ? 1 : 0);
    int w_last = -1;
    for (int w = N - 1; w >= 0; --w) {
        if (w != n && w != w0) { w_last = w; break; }
    }

    const int pf_lines = dyn_pf_lines_per_row(Rp);
    const uint64_t e = rowptr[row_e];   // global end of the shard's nnz
    const bool pf_far_on = dyn_pf_far_on();

    for (idx_t row = row_b; row < row_e; ++row) {
        const uint64_t i_beg = rowptr[row];
        const uint64_t i_end = rowptr[row + 1];
        if (i_beg == i_end) continue;      // empty row -- skip entirely

        // Prefetch the NEXT fiber's Yhat row now, while the current
        // fiber's acc[] load is in flight.  The CSR format gives us
        // row+1 trivially (PingPong/COO would have to scan ix_n for
        // it), so this is a CSR-specific win.  Gated on the same
        // DYN_PF_FAR switch as the per-nnz T2 prefetch -- both target
        // DRAM-resident outputs and both hurt when outputs fit in L3.
        if (pf_far_on && row + 1 < row_e) {
            value_t* const pf_next =
                Yhat_n + (size_t)(row + 1) * (size_t)Rp;
            for (int k = 0; k < pf_lines; ++k)
                dyn_prefetch_l2(pf_next + k * (64 / (int)sizeof(value_t)));
        }

        // Load the Yhat fiber row into SIMD registers exactly once.
        value_t* const outRow = Yhat_n + (size_t)row * Rp;
        dyn_vec_t acc[DYN_MAX_VEC_CHUNKS];
        for (int c = 0; c < VC; ++c)
            acc[c] = dyn_vload(outRow + c * DYN_SIMD_WIDTH);

        for (uint64_t k = i_beg; k < i_end; ++k) {
            // Tiered software prefetch (T0 near + T2 far).  Skip w==n
            // because Yhat[n][row] is already pinned in `acc[]`;
            // prefetching it again would waste a cache-line fetch per
            // fiber element.
            if (DYN_LIKELY(k + DYN_PF_DIST < e)) {
                const uint64_t pk = k + DYN_PF_DIST;
                for (int w = 0; w < N; ++w) {
                    if (w == n) continue;
                    const value_t* pf = Y[w] + (size_t)idx[w][pk] * Rp;
                    for (int l = 0; l < pf_lines; ++l)
                        dyn_prefetch(pf + l * (64 / (int)sizeof(value_t)));
                }
            }
            if (pf_far_on && DYN_LIKELY(k + DYN_PF_DIST_FAR < e)) {
                const uint64_t pk = k + DYN_PF_DIST_FAR;
                for (int w = 0; w < N; ++w) {
                    if (w == n) continue;
                    const value_t* pf = Y[w] + (size_t)idx[w][pk] * Rp;
                    for (int l = 0; l < pf_lines; ++l)
                        dyn_prefetch_l2(pf + l * (64 / (int)sizeof(value_t)));
                }
            }

            const value_t   val  = vals[k];
            const dyn_vec_t vval = dyn_vset1(val);

            dyn_vec_t prod[DYN_MAX_VEC_CHUNKS];
            {
                const value_t* rowY = Y[w0] + (size_t)idx[w0][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rowY + c * DYN_SIMD_WIDTH);
                    prod[c] = dyn_vmul(vval, y);
                }
            }
            for (int w = 0; w < N; ++w) {
                if (w == n || w == w0 || w == w_last) continue;
                const value_t* rY = Y[w] + (size_t)idx[w][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                    prod[c] = dyn_vmul(prod[c], y);
                }
            }

            if (w_last >= 0) {
                const value_t* rY = Y[w_last] + (size_t)idx[w_last][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                    acc[c] = dyn_vfma(prod[c], y, acc[c]);
                }
            } else {
                for (int c = 0; c < VC; ++c)
                    acc[c] = dyn_vadd(acc[c], prod[c]);
            }
        }

        for (int c = 0; c < VC; ++c)
            dyn_vstore(outRow + c * DYN_SIMD_WIDTH, acc[c]);
    }
}

#define DYN_CALL_FIBER_CSR(N_T, RP_T, TGT_T)                                \
    process_range_fiber_csr_T<N_T, RP_T, TGT_T>(vals, idx, rowptr,          \
                                                row_b, row_e, n,            \
                                                num_modes, rank_padded,     \
                                                Y, Yhat_n)

#define DYN_DISPATCH_TGT_FIBER_CSR(N_T, RP_T)                                \
    do {                                                                     \
        if (n == 0) { DYN_CALL_FIBER_CSR(N_T, RP_T, 0); return; }            \
        if (n == 1) { DYN_CALL_FIBER_CSR(N_T, RP_T, 1); return; }            \
        if (n == 2) { DYN_CALL_FIBER_CSR(N_T, RP_T, 2); return; }            \
        if constexpr ((N_T) >= 4) {                                          \
            if (n == 3) { DYN_CALL_FIBER_CSR(N_T, RP_T, 3); return; }        \
        }                                                                    \
        if constexpr ((N_T) >= 5) {                                          \
            if (n == 4) { DYN_CALL_FIBER_CSR(N_T, RP_T, 4); return; }        \
        }                                                                    \
        DYN_CALL_FIBER_CSR(N_T, RP_T, -1); return;                           \
    } while (0)

#define DYN_DISPATCH_RP_FIBER_CSR(N_T)                                \
    do {                                                              \
        if (rank_padded == 16)  DYN_DISPATCH_TGT_FIBER_CSR(N_T, 16);  \
        if (rank_padded == 32)  DYN_DISPATCH_TGT_FIBER_CSR(N_T, 32);  \
        if (rank_padded == 64)  DYN_DISPATCH_TGT_FIBER_CSR(N_T, 64);  \
        if (rank_padded == 128) DYN_DISPATCH_TGT_FIBER_CSR(N_T, 128); \
        if (rank_padded == 256) DYN_DISPATCH_TGT_FIBER_CSR(N_T, 256); \
        DYN_CALL_FIBER_CSR(N_T, 0, -1); return;                       \
    } while (0)

void dynasor_process_shard_fiber_csr(
    const value_t* vals,
    const idx_t*   const* idx,
    const uint64_t* rowptr,
    idx_t row_b, idx_t row_e,
    int n, int num_modes,
    int rank, int rank_padded,
    const value_t* const* Y,
    value_t* Yhat_n)
{
    (void)rank;

    if (DYN_UNLIKELY(!dyn_tgt_specialize_on())) {
        if (num_modes == 3) {
            if (rank_padded == 16)  { DYN_CALL_FIBER_CSR(3, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL_FIBER_CSR(3, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL_FIBER_CSR(3, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL_FIBER_CSR(3, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL_FIBER_CSR(3, 256, -1); return; }
            DYN_CALL_FIBER_CSR(3, 0, -1); return;
        }
        if (num_modes == 4) {
            if (rank_padded == 16)  { DYN_CALL_FIBER_CSR(4, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL_FIBER_CSR(4, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL_FIBER_CSR(4, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL_FIBER_CSR(4, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL_FIBER_CSR(4, 256, -1); return; }
            DYN_CALL_FIBER_CSR(4, 0, -1); return;
        }
        if (num_modes == 5) {
            if (rank_padded == 16)  { DYN_CALL_FIBER_CSR(5, 16,  -1); return; }
            if (rank_padded == 32)  { DYN_CALL_FIBER_CSR(5, 32,  -1); return; }
            if (rank_padded == 64)  { DYN_CALL_FIBER_CSR(5, 64,  -1); return; }
            if (rank_padded == 128) { DYN_CALL_FIBER_CSR(5, 128, -1); return; }
            if (rank_padded == 256) { DYN_CALL_FIBER_CSR(5, 256, -1); return; }
            DYN_CALL_FIBER_CSR(5, 0, -1); return;
        }
        DYN_CALL_FIBER_CSR(0, 0, -1);
        return;
    }

    if (num_modes == 3) DYN_DISPATCH_RP_FIBER_CSR(3);
    if (num_modes == 4) DYN_DISPATCH_RP_FIBER_CSR(4);
    if (num_modes == 5) DYN_DISPATCH_RP_FIBER_CSR(5);
    DYN_CALL_FIBER_CSR(0, 0, -1);
}

#undef DYN_CALL_FIBER_CSR
#undef DYN_DISPATCH_RP_FIBER_CSR
#undef DYN_DISPATCH_TGT_FIBER_CSR

// ============================================================================
//  All-modes single-pass kernel
//
//  Given one shard-range [b, e) of the SoA slab, produce partial Yhat updates
//  for EVERY target mode 0..N-1 in a single pass over the tensor.  Writes
//  land in the caller-supplied per-thread private output buffers
//  Yout[0..N-1], each sized mode_size[n] * rank_padded.  A separate reduce
//  step (see spmttkrp_all_modes_dynasor in dynasor.cpp) fuses these into
//  the global Yhat after all threads finish.
//
//  Per-nnz compute (N=3):
//      vY0  = val * Y[0][idx[0][i]]          (1 vmul)
//      vY1  = val * Y[1][idx[1][i]]          (1 vmul)
//      Yout[0][idx[0][i]] += vY1 * Y[2][..]  (1 vfma)   val*Y1*Y2 for mode 0
//      Yout[1][idx[1][i]] += vY0 * Y[2][..]  (1 vfma)   val*Y0*Y2 for mode 1
//      Yout[2][idx[2][i]] += vY0 * Y[1][..]  (1 vfma)   val*Y0*Y1 for mode 2
//  Total: 2 vmul + 3 vfma per nnz per SIMD chunk.  Compare to the per-mode
//  kernel at N=3: each mode does 1 vmul + 1 vfma (2 ops), times N=3 sweeps
//  = 6 ops per nnz per chunk total.  All-modes is 5 ops -- 17% fewer FLOPs
//  AND the tensor slab is streamed once instead of 3x.
//
//  Per-nnz compute (N=4, prefix-suffix):
//      prefix:  p1 = val*Y0 ; p2 = p1*Y1 ; p3 = p2*Y2        (3 vmul)
//      suffix:  s1 = Y2*Y3  ; s0 = Y1*s1                     (2 vmul)
//      Yout[0] += val * s0        == val * (Y1*Y2*Y3)        (1 vfma)
//      Yout[1] += p1 * s1         == (val*Y0) * (Y2*Y3)      (1 vfma)
//      Yout[2] += p2 * Y3         == (val*Y0*Y1) * Y3        (1 vfma)
//      Yout[3] += p3              == (val*Y0*Y1*Y2)          (1 vadd)
//  Total: 5 vmul + 3 vfma + 1 vadd = 9 ops per nnz per chunk.  Compare to
//  per-mode at N=4: each mode does 2 vmul + 1 vfma = 3 ops, times N=4 = 12.
//  All-modes is 25% fewer FLOPs AND the tensor slab is streamed once
//  instead of 4x.
//
//  Correctness note: because each thread writes into its own private
//  Yout[] copy, no atomics or locks are needed.  The caller reduces the
//  per-thread copies into the global Yhat after the parallel region.
// ============================================================================
template<int N_TMPL, int RP_TMPL>
static inline void process_range_all_modes_T(
    const value_t* DYN_RESTRICT vals,
    const idx_t*   const*       idx,
    uint64_t b, uint64_t e,
    int num_modes_rt,
    int rank_padded_rt,
    const value_t* const* DYN_RESTRICT Y,
    value_t* const* DYN_RESTRICT Yout)
{
    constexpr bool kHasN  = (N_TMPL  > 0);
    constexpr bool kHasRP = (RP_TMPL > 0);
    const int N  = kHasN  ? N_TMPL  : num_modes_rt;
    const int Rp = kHasRP ? RP_TMPL : rank_padded_rt;
    const int VC = Rp / DYN_SIMD_WIDTH;

    const int pf_lines   = dyn_pf_lines_per_row(Rp);
    const bool pf_far_on = dyn_pf_far_on();

    for (uint64_t i = b; i < e; ++i) {
        // Tiered software prefetch.  For all-modes we have N factor-row
        // loads AND N output-row loads per nnz; prefetching both helps a
        // lot on random-index workloads.
        if (DYN_LIKELY(i + DYN_PF_DIST < e)) {
            const uint64_t pi = i + DYN_PF_DIST;
            for (int w = 0; w < N; ++w) {
                const idx_t r = idx[w][pi];
                const value_t* pf_in = Y[w]    + (size_t)r * Rp;
                const value_t* pf_ot = Yout[w] + (size_t)r * Rp;
                for (int k = 0; k < pf_lines; ++k) {
                    dyn_prefetch(pf_in + k * (64 / (int)sizeof(value_t)));
                    dyn_prefetch(pf_ot + k * (64 / (int)sizeof(value_t)));
                }
            }
        }
        if (pf_far_on && DYN_LIKELY(i + DYN_PF_DIST_FAR < e)) {
            const uint64_t pi = i + DYN_PF_DIST_FAR;
            for (int w = 0; w < N; ++w) {
                const idx_t r = idx[w][pi];
                const value_t* pf_in = Y[w]    + (size_t)r * Rp;
                const value_t* pf_ot = Yout[w] + (size_t)r * Rp;
                for (int k = 0; k < pf_lines; ++k) {
                    dyn_prefetch_l2(pf_in + k * (64 / (int)sizeof(value_t)));
                    dyn_prefetch_l2(pf_ot + k * (64 / (int)sizeof(value_t)));
                }
            }
        }

        const value_t   val  = vals[i];
        const dyn_vec_t vval = dyn_vset1(val);

        // -----------------------------------------------------------------
        //  N == 3 hot path: 2 vmul + 3 vfma per chunk.
        // -----------------------------------------------------------------
        if constexpr (N_TMPL == 3) {
            const value_t* DYN_RESTRICT rY0 = Y[0] + (size_t)idx[0][i] * Rp;
            const value_t* DYN_RESTRICT rY1 = Y[1] + (size_t)idx[1][i] * Rp;
            const value_t* DYN_RESTRICT rY2 = Y[2] + (size_t)idx[2][i] * Rp;
            value_t* DYN_RESTRICT oY0 = Yout[0] + (size_t)idx[0][i] * Rp;
            value_t* DYN_RESTRICT oY1 = Yout[1] + (size_t)idx[1][i] * Rp;
            value_t* DYN_RESTRICT oY2 = Yout[2] + (size_t)idx[2][i] * Rp;

            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t y0  = dyn_vload(rY0 + off);
                dyn_vec_t y1  = dyn_vload(rY1 + off);
                dyn_vec_t y2  = dyn_vload(rY2 + off);
                dyn_vec_t vy0 = dyn_vmul(vval, y0);   // val * Y0
                dyn_vec_t vy1 = dyn_vmul(vval, y1);   // val * Y1
                // Yhat[0] += val * Y1 * Y2
                dyn_vec_t o0 = dyn_vload(oY0 + off);
                o0 = dyn_vfma(vy1, y2, o0);
                dyn_vstore(oY0 + off, o0);
                // Yhat[1] += val * Y0 * Y2
                dyn_vec_t o1 = dyn_vload(oY1 + off);
                o1 = dyn_vfma(vy0, y2, o1);
                dyn_vstore(oY1 + off, o1);
                // Yhat[2] += val * Y0 * Y1
                dyn_vec_t o2 = dyn_vload(oY2 + off);
                o2 = dyn_vfma(vy0, y1, o2);
                dyn_vstore(oY2 + off, o2);
            }
            continue;
        }

        // -----------------------------------------------------------------
        //  N == 4 prefix-suffix path: 5 vmul + 3 vfma + 1 vadd per chunk.
        // -----------------------------------------------------------------
        if constexpr (N_TMPL == 4) {
            const value_t* DYN_RESTRICT rY0 = Y[0] + (size_t)idx[0][i] * Rp;
            const value_t* DYN_RESTRICT rY1 = Y[1] + (size_t)idx[1][i] * Rp;
            const value_t* DYN_RESTRICT rY2 = Y[2] + (size_t)idx[2][i] * Rp;
            const value_t* DYN_RESTRICT rY3 = Y[3] + (size_t)idx[3][i] * Rp;
            value_t* DYN_RESTRICT oY0 = Yout[0] + (size_t)idx[0][i] * Rp;
            value_t* DYN_RESTRICT oY1 = Yout[1] + (size_t)idx[1][i] * Rp;
            value_t* DYN_RESTRICT oY2 = Yout[2] + (size_t)idx[2][i] * Rp;
            value_t* DYN_RESTRICT oY3 = Yout[3] + (size_t)idx[3][i] * Rp;

            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t y0 = dyn_vload(rY0 + off);
                dyn_vec_t y1 = dyn_vload(rY1 + off);
                dyn_vec_t y2 = dyn_vload(rY2 + off);
                dyn_vec_t y3 = dyn_vload(rY3 + off);

                // prefix
                dyn_vec_t p1 = dyn_vmul(vval, y0);     // val*Y0
                dyn_vec_t p2 = dyn_vmul(p1,   y1);     // val*Y0*Y1
                dyn_vec_t p3 = dyn_vmul(p2,   y2);     // val*Y0*Y1*Y2

                // suffix
                dyn_vec_t s1 = dyn_vmul(y2,   y3);     // Y2*Y3
                dyn_vec_t s0 = dyn_vmul(y1,   s1);     // Y1*Y2*Y3

                // Yhat[0] += val * (Y1*Y2*Y3) = vval * s0
                dyn_vec_t o0 = dyn_vload(oY0 + off);
                o0 = dyn_vfma(vval, s0, o0);
                dyn_vstore(oY0 + off, o0);
                // Yhat[1] += (val*Y0) * (Y2*Y3) = p1 * s1
                dyn_vec_t o1 = dyn_vload(oY1 + off);
                o1 = dyn_vfma(p1, s1, o1);
                dyn_vstore(oY1 + off, o1);
                // Yhat[2] += (val*Y0*Y1) * Y3 = p2 * y3
                dyn_vec_t o2 = dyn_vload(oY2 + off);
                o2 = dyn_vfma(p2, y3, o2);
                dyn_vstore(oY2 + off, o2);
                // Yhat[3] += val*Y0*Y1*Y2 = p3  (suffix[3] = 1 -> just add)
                dyn_vec_t o3 = dyn_vload(oY3 + off);
                o3 = dyn_vadd(o3, p3);
                dyn_vstore(oY3 + off, o3);
            }
            continue;
        }

        // -----------------------------------------------------------------
        //  General-N fallback (prefix-suffix, works for any N >= 2).
        //
        //  One-pass strategy that avoids buffering all N factor rows into
        //  vector registers (pressure on 16-reg NEON at large VC):
        //
        //    1. Sweep forward with a running product `run`, initializing
        //       run = val and multiplying in Y[0..N-2].  Stash the row
        //       pointers in `rPtr[]`.
        //    2. Sweep backward rebuilding the suffix product `suf`, and
        //       at each step k combine `prefix[k] * suf[k+1]` into
        //       Yout[k].  prefix[k] = run-at-that-moment which we can
        //       derive in reverse by dividing `run` by Y[k] -- but we
        //       want to avoid division, so instead we stash prefixes as
        //       we go and re-use them.
        //
        //  Concretely we use two scratch arrays:
        //
        //      pref[k][c] = val * Y[0] * ... * Y[k-1]    (prefix[k])
        //      pref[0]    = val (implicit)
        //
        //  built forward over k = 0..N-1 (1 vmul per k * VC chunks).
        //  Then we sweep k = N-1 downto 0 keeping a running suffix:
        //
        //      suf      = 1 initially
        //      excl_k   = pref[k] * suf            -> Yout[k] += excl_k
        //      suf      = suf * Y[k]               (prepare next iter)
        //
        //  Total: (N-1) vmul to build pref, (N-1) vmul to update suf,
        //  N combine ops (vfma or vadd).  Slightly more ops than the
        //  specialized N=3/4 paths but identical FLOP count as the
        //  hand-tuned prefix-suffix above for N >= 4.
        //
        //  Memory footprint: pref[] holds N * VC vectors -- up to
        //  8 * 16 = 128 SIMD regs which spills to stack for N >= 5 on
        //  AVX-512 (32 archreg).  Still fine: the stack slots are L1-
        //  resident for the duration of the nnz iteration.
        // -----------------------------------------------------------------
        dyn_vec_t pref[DYN_MAX_MODES][DYN_MAX_VEC_CHUNKS];
        const value_t* rPtr[DYN_MAX_MODES];
        value_t*       oPtr[DYN_MAX_MODES];
        for (int w = 0; w < N; ++w) {
            rPtr[w] = Y[w]    + (size_t)idx[w][i] * Rp;
            oPtr[w] = Yout[w] + (size_t)idx[w][i] * Rp;
        }

        // Build prefix products: pref[0] = val, pref[k] = pref[k-1] * Y[k-1].
        for (int c = 0; c < VC; ++c) pref[0][c] = vval;
        for (int k = 1; k < N; ++k) {
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t y = dyn_vload(rPtr[k - 1] + c * DYN_SIMD_WIDTH);
                pref[k][c] = dyn_vmul(pref[k - 1][c], y);
            }
        }

        // Reverse sweep: maintain suf (init = 1), combine with pref[k]
        // into Yout[k], then fold Y[k] into suf.
        dyn_vec_t suf[DYN_MAX_VEC_CHUNKS];
        for (int c = 0; c < VC; ++c) suf[c] = dyn_vset1(1.0f);

        for (int k = N - 1; k >= 0; --k) {
            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t o = dyn_vload(oPtr[k] + off);
                o = dyn_vfma(pref[k][c], suf[c], o);
                dyn_vstore(oPtr[k] + off, o);
            }
            if (k > 0) {
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rPtr[k] + c * DYN_SIMD_WIDTH);
                    suf[c] = dyn_vmul(suf[c], y);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Public entry + dispatch grid for the all-modes kernel.
//  Specializations:   N in {3, 4, 5}  x  Rp in {16, 32, 64, 128, 256}
//  Fallback:          N=0, Rp=0 (fully runtime-parameterized)
// ---------------------------------------------------------------------------
#define DYN_CALL_AM(N_T, RP_T) \
    process_range_all_modes_T<N_T, RP_T>(vals, idx, b, e,                 \
                                         num_modes, rank_padded,          \
                                         Y, Yout)

#define DYN_DISPATCH_RP_AM(N_T)                                  \
    do {                                                          \
        if (rank_padded == 16)  { DYN_CALL_AM(N_T, 16 ); return; } \
        if (rank_padded == 32)  { DYN_CALL_AM(N_T, 32 ); return; } \
        if (rank_padded == 64)  { DYN_CALL_AM(N_T, 64 ); return; } \
        if (rank_padded == 128) { DYN_CALL_AM(N_T, 128); return; } \
        if (rank_padded == 256) { DYN_CALL_AM(N_T, 256); return; } \
        DYN_CALL_AM(N_T, 0); return;                               \
    } while (0)

void dynasor_process_shard_all_modes(
    const value_t* vals,
    const idx_t*   const* idx,
    uint64_t b, uint64_t e,
    int num_modes, int rank, int rank_padded,
    const value_t* const* Y,
    value_t* const* Yout)
{
    (void)rank;

    if (num_modes == 3) DYN_DISPATCH_RP_AM(3);
    if (num_modes == 4) DYN_DISPATCH_RP_AM(4);
    if (num_modes == 5) DYN_DISPATCH_RP_AM(5);
    DYN_CALL_AM(0, 0);
}

#undef DYN_CALL_AM
#undef DYN_DISPATCH_RP_AM

} // namespace dynasor
