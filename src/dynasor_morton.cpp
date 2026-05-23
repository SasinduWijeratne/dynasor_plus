// ============================================================================
//  dynasor_morton.cpp  --  Morton / Z-curve linearized storage + all-modes
//  MTTKRP driver.  See include/dynasor_morton.h for the design overview.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_jit.h"
#include "dynasor_morton.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#include <ctime>
static inline double omp_get_wtime() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

// x86 BMI2 pdep/pext.  On -march=native for Zen 3+ / Haswell+ the builtins
// become single-cycle throughput instructions; without BMI2 we fall back
// to a portable scalar loop (only used during the one-shot build; the
// kernel is specialized to BMI2 paths only -- see MORTON_USE_BMI2).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #if defined(__BMI2__)
    #include <immintrin.h>
    #define MORTON_USE_BMI2 1
  #endif
#endif

namespace dynasor {

// ---------------------------------------------------------------------------
//  Portable pdep/pext shims.  Inline for the hot kernel; the build path
//  uses them too and only runs once per tensor, so the non-BMI2 fallback
//  cost is negligible there.
// ---------------------------------------------------------------------------
static inline uint64_t morton_pdep(uint64_t x, uint64_t mask) {
#if defined(MORTON_USE_BMI2)
    return _pdep_u64(x, mask);
#else
    uint64_t result = 0;
    uint64_t m = mask;
    uint64_t b = 1;
    while (m) {
        const uint64_t low = m & (uint64_t)(-(int64_t)m);   // lowest set bit
        if (x & b) result |= low;
        m ^= low;
        b <<= 1;
    }
    return result;
#endif
}

static inline uint64_t morton_pext(uint64_t x, uint64_t mask) {
#if defined(MORTON_USE_BMI2)
    return _pext_u64(x, mask);
#else
    uint64_t result = 0;
    uint64_t m = mask;
    uint64_t b = 1;
    while (m) {
        const uint64_t low = m & (uint64_t)(-(int64_t)m);   // lowest set bit
        if (x & low) result |= b;
        m ^= low;
        b <<= 1;
    }
    return result;
#endif
}

static inline int ceil_log2_idx(idx_t v) {
    if (v <= 1) return 1;                  // always reserve >=1 bit per mode
    int b = 0;
    idx_t m = v - 1;                       // ceil(log2(v)) = floor(log2(v-1))+1
    while (m) { b++; m >>= 1; }
    return b;
}

// ---------------------------------------------------------------------------
//  Build the per-mode bit plan.  Bit positions are assigned LSB-first
//  round-robin across modes with remaining bits: at global bit k, we
//  pick the next mode n that still has bits left, assign bit k to that
//  mode, and advance.
//
//  Example (N=3, I = (8192, 8192, 8192)): b = (13, 13, 13), total=39.
//  Bit positions 0,3,6,... -> mode 0; 1,4,7,... -> mode 1;
//  2,5,8,... -> mode 2.  The key looks like
//      [ z_12 y_12 x_12  z_11 y_11 x_11  ...  z_0 y_0 x_0 ]_MSB..LSB
//  Sorting ascending gives Z-curve order.
//
//  For unequal mode sizes, a mode drops out of the round-robin once
//  it has received all its bits; remaining positions are filled by the
//  still-active modes.  Example (N=4, I=(16, 16, 4096, 16)):
//  b = (4, 4, 12, 4), total = 24.
// ---------------------------------------------------------------------------
bool morton_build_plan(int             num_modes,
                       const idx_t*    mode_size,
                       uint64_t*       mask_out,
                       uint8_t*        bits_out)
{
    int b[DYN_MAX_MODES] = {0};
    int total = 0;
    for (int n = 0; n < num_modes; ++n) {
        b[n] = ceil_log2_idx(mode_size[n]);
        bits_out[n] = (uint8_t)b[n];
        total += b[n];
    }
    if (total > 64) return false;           // does not fit in a 64-bit key

    int remaining[DYN_MAX_MODES];
    for (int n = 0; n < num_modes; ++n) {
        remaining[n] = b[n];
        mask_out[n]  = 0;
    }

    int pos = 0;
    bool progressed = true;
    while (pos < 64 && progressed) {
        progressed = false;
        for (int n = 0; n < num_modes; ++n) {
            if (remaining[n] > 0) {
                mask_out[n] |= (uint64_t)1 << pos;
                remaining[n]--;
                pos++;
                progressed = true;
                if (pos >= 64) break;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
//  Convert a loaded SoA tensor into Morton storage.
// ---------------------------------------------------------------------------
bool build_morton_layout(Tensor& T, int num_threads) {
    const int      N   = T.num_modes;
    const uint64_t nnz = T.nnz;

    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    uint64_t mask[DYN_MAX_MODES];
    uint8_t  bits[DYN_MAX_MODES];
    if (!morton_build_plan(N, T.mode_size, mask, bits)) {
        std::fprintf(stderr,
            "[morton] cannot build plan: sum(ceil(log2(I_n))) > 64\n");
        return false;
    }

    if (!T.vals || !T.idx_buf[0]) {
        std::fprintf(stderr,
            "[morton] source SoA slab not available (vals=%p idx0=%p); "
            "Morton build requires the primary slab.\n",
            (void*)T.vals, (void*)T.idx_buf[0]);
        return false;
    }

    const double t0 = omp_get_wtime();

    // Allocate the Morton backing buffer: [keys (u64 * nnz)][vals (f32 * nnz)].
    constexpr size_t kA = DYN_ALIGN;
    const size_t keys_bytes = (nnz * sizeof(uint64_t) + kA - 1) & ~(kA - 1);
    const size_t vals_bytes = (nnz * sizeof(value_t)  + kA - 1) & ~(kA - 1);
    const size_t total_bytes = keys_bytes + vals_bytes + kA;

    void* raw = dyn_aligned_alloc(total_bytes);
    if (!raw) {
        std::fprintf(stderr,
            "[morton] OOM allocating %.2f GiB for morton keys+vals\n",
            (double)total_bytes / (double)(1ULL << 30));
        return false;
    }

    uint64_t* keys = (uint64_t*)raw;
    value_t*  vals = (value_t*)((char*)raw + keys_bytes);

    // ---- Pass 1: pack keys in parallel, memcpy vals ----
    const int      N_loc = N;
    uint64_t       mask_loc[DYN_MAX_MODES];
    for (int n = 0; n < N_loc; ++n) mask_loc[n] = mask[n];

    const idx_t* const* idx_in = (const idx_t* const*)T.idx_buf;
    const value_t*      val_in = T.vals;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t i64 = 0; i64 < (int64_t)nnz; ++i64) {
        const uint64_t i = (uint64_t)i64;
        uint64_t k = 0;
        for (int n = 0; n < N_loc; ++n) {
            k |= morton_pdep((uint64_t)idx_in[n][i], mask_loc[n]);
        }
        keys[i] = k;
    }

    std::memcpy(vals, val_in, (size_t)nnz * sizeof(value_t));

    // ---- Pass 2: parallel radix-partition + per-bucket std::sort. ----
    //
    //   (a) Histogram top-8-bits of each key across all threads.
    //   (b) Prefix-sum bucket counts and per-thread offsets so each
    //       thread scatters its slice of nnz into a uniquely-owned
    //       sub-range of the AoS temp buffer.
    //   (c) std::sort each of 256 buckets in parallel (embarrassingly
    //       parallel across buckets; dynamic schedule since bucket
    //       sizes may vary for skewed tensors).
    //   (d) Split the sorted AoS buffer back into (keys, vals) SoA.
    //
    // Peak extra memory: sizeof(MortonPair) * nnz = 16 B * nnz.  For
    // 335 M nnz that's 5.36 GiB which fits alongside the 3.74 GiB
    // permanent Morton buffer in the existing budget.
    constexpr int kBucketBits = 8;
    constexpr int kNumBuckets = 1 << kBucketBits;                    // 256
    const     int shift       = 64 - kBucketBits;                    // 56

    struct MortonPair { uint64_t k; value_t v; };   // 16 B (with 4 B pad)

    MortonPair* tbuf = (MortonPair*)dyn_aligned_alloc(
        (size_t)nnz * sizeof(MortonPair));
    if (!tbuf) {
        dyn_aligned_free(raw);
        std::fprintf(stderr, "[morton] OOM for sort temp (%.2f GiB)\n",
            (double)((size_t)nnz * sizeof(MortonPair)) / (double)(1ULL << 30));
        return false;
    }

    // Per-thread histograms laid out as [num_threads][kNumBuckets].
    std::vector<uint64_t> thist((size_t)num_threads * kNumBuckets, 0);

    // (a) Histogram.
    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads) {
            uint64_t* h = thist.data() + (size_t)tid * kNumBuckets;
            const uint64_t b = (uint64_t)tid * ((nnz + num_threads - 1) / num_threads);
            const uint64_t e = std::min(nnz, b + (nnz + num_threads - 1) / num_threads);
            for (uint64_t i = b; i < e; ++i) {
                h[keys[i] >> shift]++;
            }
        }
    }

    // (b) Prefix sums: per-bucket base, then per-(tid, bucket) cursor.
    std::vector<uint64_t> bucket_base((size_t)kNumBuckets + 1, 0);
    for (int bkt = 0; bkt < kNumBuckets; ++bkt) {
        uint64_t sum = 0;
        for (int t = 0; t < num_threads; ++t) {
            sum += thist[(size_t)t * kNumBuckets + bkt];
        }
        bucket_base[bkt + 1] = bucket_base[bkt] + sum;
    }
    // Convert per-(tid,bucket) counts into write cursors: for thread t and
    // bucket bkt, start = bucket_base[bkt] + sum_{t' < t} thist[t'][bkt].
    std::vector<uint64_t> cursor((size_t)num_threads * kNumBuckets, 0);
    for (int bkt = 0; bkt < kNumBuckets; ++bkt) {
        uint64_t running = bucket_base[bkt];
        for (int t = 0; t < num_threads; ++t) {
            cursor[(size_t)t * kNumBuckets + bkt] = running;
            running += thist[(size_t)t * kNumBuckets + bkt];
        }
    }

    // (c) Parallel scatter into the AoS temp buffer.
    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads) {
            uint64_t* cur = cursor.data() + (size_t)tid * kNumBuckets;
            const uint64_t b = (uint64_t)tid * ((nnz + num_threads - 1) / num_threads);
            const uint64_t e = std::min(nnz, b + (nnz + num_threads - 1) / num_threads);
            for (uint64_t i = b; i < e; ++i) {
                const uint64_t k = keys[i];
                const uint64_t bkt = k >> shift;
                const uint64_t p = cur[bkt]++;
                tbuf[p].k = k;
                tbuf[p].v = vals[i];
            }
        }
    }

    // (d) Parallel per-bucket sort.  Dynamic schedule handles skew.
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 4)
    for (int bkt = 0; bkt < kNumBuckets; ++bkt) {
        const uint64_t lo = bucket_base[(size_t)bkt];
        const uint64_t hi = bucket_base[(size_t)bkt + 1];
        if (hi > lo) {
            std::sort(tbuf + lo, tbuf + hi,
                      [](const MortonPair& a, const MortonPair& b) {
                          return a.k < b.k;
                      });
        }
    }

    // (e) Split the sorted AoS pairs back into (keys, vals) SoA.
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t i = 0; i < (int64_t)nnz; ++i) {
        keys[(size_t)i] = tbuf[(size_t)i].k;
        vals[(size_t)i] = tbuf[(size_t)i].v;
    }

    dyn_aligned_free(tbuf);

    // Commit the new storage to the Tensor, then free the SoA slabs.
    T.morton_buf_raw   = raw;
    T.morton_buf_bytes = total_bytes;
    T.morton_keys      = keys;
    T.morton_vals      = vals;
    for (int n = 0; n < N_loc; ++n) {
        T.morton_mask[n] = mask[n];
        T.morton_bits[n] = bits[n];
    }

    // Release the SoA buffers.  The Morton layout owns everything it
    // needs now.  Keep buf_scratch_raw around if it exists (caller may
    // still need it for NCopy diagnostics; we only free it if it is
    // currently unused), but free the primary.
    if (T.buf_raw) {
        dyn_aligned_free(T.buf_raw);
        T.buf_raw     = nullptr;
        T.buf_bytes   = 0;
        T.buf_capacity = 0;
    }
    if (T.buf_scratch_raw) {
        dyn_aligned_free(T.buf_scratch_raw);
        T.buf_scratch_raw   = nullptr;
        T.buf_scratch_bytes = 0;
    }
    T.vals     = nullptr;
    T.scr_vals = nullptr;
    for (int n = 0; n < DYN_MAX_MODES; ++n) {
        T.idx_buf[n] = nullptr;
        T.scr_idx[n] = nullptr;
    }

    T.layout = Layout::Morton;

    const double secs = omp_get_wtime() - t0;
    std::printf("[morton] build: N=%d nnz=%llu bits=[", N, (unsigned long long)nnz);
    for (int n = 0; n < N; ++n) std::printf("%s%d", n ? "," : "", (int)bits[n]);
    int total_bits = 0;
    for (int n = 0; n < N; ++n) total_bits += (int)bits[n];
    std::printf("] total=%d  layout-bytes=%.3f GiB  build=%.3f s\n",
                total_bits,
                (double)total_bytes / (double)(1ULL << 30),
                secs);
    return true;
}

// ---------------------------------------------------------------------------
//  Stage 3 -- fused Morton all-modes kernel.
//
//  The Stage 2 driver called `dynasor_process_shard_all_modes` per BATCH
//  of unpacked indices.  That approach wrote N * BATCH * 4 B of SoA idx
//  scratch to L1 and read it back on each pass -- ~96 KiB of L1 traffic
//  per 4096 nnz for N=3.  It also paid a template-dispatch + function-
//  call cost per batch, and the inner kernel's tiered prefetch saw only
//  BATCH-bounded lookahead (instead of the whole shard).
//
//  The fused kernel below extracts indices via _pext_u64 INLINE, so:
//    * no L1 scratch traffic for the idx stream;
//    * software-prefetch reaches DYN_PF_DIST / DYN_PF_DIST_FAR ahead
//      across the full shard range, not just within a BATCH window;
//    * the templated (N, Rp) dispatch happens exactly ONCE per thread-
//      shard, letting GCC fully unroll the per-mode and per-SIMD-chunk
//      loops inside the `for (i = b; i < e; ++i)` steady state.
//
//  Per-nnz compute matches the Stage 1/2 all-modes kernel exactly:
//    N==3 : 2 vmul + 3 vfma per SIMD chunk.
//    N==4 : 5 vmul + 3 vfma + 1 vadd per SIMD chunk (prefix-suffix).
//    N>=5 : prefix-suffix fallback with pref[][] + suf[] scratch.
//
//  Rank specializations {16, 32, 64, 128, 256} preserve the register-
//  resident running products across full rank_padded rows on Zen 4/5
//  (32 ZMM) and Neoverse V2 (32 NEON).
// ---------------------------------------------------------------------------
template<int N_TMPL, int RP_TMPL>
static inline void process_range_all_modes_morton_T(
    const value_t*  DYN_RESTRICT mvals,
    const uint64_t* DYN_RESTRICT mkeys,
    const uint64_t* DYN_RESTRICT mask_arr,
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

    // Per-mode masks held in a compact local array.  For N<=4 the loop
    // unroller keeps these in registers across the whole steady state.
    uint64_t m[DYN_MAX_MODES];
    for (int w = 0; w < N; ++w) m[w] = mask_arr[w];

    const int  pf_lines   = (Rp * (int)sizeof(value_t) + 63) / 64;

    // DYN_PF_DIST / DYN_PF_DIST_FAR mirror dynasor_kernel.cpp.  We do
    // not depend on dyn_pf_far_on() here (local copy to avoid the
    // cross-TU dependency; default off matches the SoA driver).
    constexpr uint64_t DYN_PF_DIST     = 4;
    constexpr uint64_t DYN_PF_DIST_FAR = 16;
    const bool pf_far_on = [] {
        const char* s = std::getenv("DYN_PF_FAR");
        return s && s[0] && s[0] != '0';
    }();

    for (uint64_t i = b; i < e; ++i) {
        // ---- tiered software prefetch (inline pext) ---------------------
        if (DYN_LIKELY(i + DYN_PF_DIST < e)) {
            const uint64_t kpf = mkeys[i + DYN_PF_DIST];
            for (int w = 0; w < N; ++w) {
                const idx_t r = (idx_t)morton_pext(kpf, m[w]);
                const value_t* pf_in = Y[w]    + (size_t)r * Rp;
                const value_t* pf_ot = Yout[w] + (size_t)r * Rp;
                for (int k = 0; k < pf_lines; ++k) {
                    dyn_prefetch(pf_in + k * (64 / (int)sizeof(value_t)));
                    dyn_prefetch(pf_ot + k * (64 / (int)sizeof(value_t)));
                }
            }
        }
        if (pf_far_on && DYN_LIKELY(i + DYN_PF_DIST_FAR < e)) {
            const uint64_t kpf = mkeys[i + DYN_PF_DIST_FAR];
            for (int w = 0; w < N; ++w) {
                const idx_t r = (idx_t)morton_pext(kpf, m[w]);
                const value_t* pf_in = Y[w]    + (size_t)r * Rp;
                const value_t* pf_ot = Yout[w] + (size_t)r * Rp;
                for (int k = 0; k < pf_lines; ++k) {
                    dyn_prefetch_l2(pf_in + k * (64 / (int)sizeof(value_t)));
                    dyn_prefetch_l2(pf_ot + k * (64 / (int)sizeof(value_t)));
                }
            }
        }

        // ---- extract current-nnz indices from the Morton key -----------
        const uint64_t key = mkeys[i];
        idx_t idx[DYN_MAX_MODES];
        for (int w = 0; w < N; ++w) idx[w] = (idx_t)morton_pext(key, m[w]);

        const value_t   val  = mvals[i];
        const dyn_vec_t vval = dyn_vset1(val);

        // -----------------------------------------------------------------
        //  N == 3 hot path: 2 vmul + 3 vfma per SIMD chunk.
        // -----------------------------------------------------------------
        if constexpr (N_TMPL == 3) {
            const value_t* DYN_RESTRICT rY0 = Y[0] + (size_t)idx[0] * Rp;
            const value_t* DYN_RESTRICT rY1 = Y[1] + (size_t)idx[1] * Rp;
            const value_t* DYN_RESTRICT rY2 = Y[2] + (size_t)idx[2] * Rp;
            value_t* DYN_RESTRICT oY0 = Yout[0] + (size_t)idx[0] * Rp;
            value_t* DYN_RESTRICT oY1 = Yout[1] + (size_t)idx[1] * Rp;
            value_t* DYN_RESTRICT oY2 = Yout[2] + (size_t)idx[2] * Rp;

            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t y0  = dyn_vload(rY0 + off);
                dyn_vec_t y1  = dyn_vload(rY1 + off);
                dyn_vec_t y2  = dyn_vload(rY2 + off);
                dyn_vec_t vy0 = dyn_vmul(vval, y0);
                dyn_vec_t vy1 = dyn_vmul(vval, y1);
                dyn_vec_t o0 = dyn_vload(oY0 + off);
                o0 = dyn_vfma(vy1, y2, o0);
                dyn_vstore(oY0 + off, o0);
                dyn_vec_t o1 = dyn_vload(oY1 + off);
                o1 = dyn_vfma(vy0, y2, o1);
                dyn_vstore(oY1 + off, o1);
                dyn_vec_t o2 = dyn_vload(oY2 + off);
                o2 = dyn_vfma(vy0, y1, o2);
                dyn_vstore(oY2 + off, o2);
            }
            continue;
        }

        // -----------------------------------------------------------------
        //  N == 4 prefix-suffix: 5 vmul + 3 vfma + 1 vadd per SIMD chunk.
        // -----------------------------------------------------------------
        if constexpr (N_TMPL == 4) {
            const value_t* DYN_RESTRICT rY0 = Y[0] + (size_t)idx[0] * Rp;
            const value_t* DYN_RESTRICT rY1 = Y[1] + (size_t)idx[1] * Rp;
            const value_t* DYN_RESTRICT rY2 = Y[2] + (size_t)idx[2] * Rp;
            const value_t* DYN_RESTRICT rY3 = Y[3] + (size_t)idx[3] * Rp;
            value_t* DYN_RESTRICT oY0 = Yout[0] + (size_t)idx[0] * Rp;
            value_t* DYN_RESTRICT oY1 = Yout[1] + (size_t)idx[1] * Rp;
            value_t* DYN_RESTRICT oY2 = Yout[2] + (size_t)idx[2] * Rp;
            value_t* DYN_RESTRICT oY3 = Yout[3] + (size_t)idx[3] * Rp;

            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t y0 = dyn_vload(rY0 + off);
                dyn_vec_t y1 = dyn_vload(rY1 + off);
                dyn_vec_t y2 = dyn_vload(rY2 + off);
                dyn_vec_t y3 = dyn_vload(rY3 + off);

                dyn_vec_t p1 = dyn_vmul(vval, y0);
                dyn_vec_t p2 = dyn_vmul(p1,   y1);
                dyn_vec_t p3 = dyn_vmul(p2,   y2);

                dyn_vec_t s1 = dyn_vmul(y2,   y3);
                dyn_vec_t s0 = dyn_vmul(y1,   s1);

                dyn_vec_t o0 = dyn_vload(oY0 + off);
                o0 = dyn_vfma(vval, s0, o0);
                dyn_vstore(oY0 + off, o0);
                dyn_vec_t o1 = dyn_vload(oY1 + off);
                o1 = dyn_vfma(p1, s1, o1);
                dyn_vstore(oY1 + off, o1);
                dyn_vec_t o2 = dyn_vload(oY2 + off);
                o2 = dyn_vfma(p2, y3, o2);
                dyn_vstore(oY2 + off, o2);
                dyn_vec_t o3 = dyn_vload(oY3 + off);
                o3 = dyn_vadd(o3, p3);
                dyn_vstore(oY3 + off, o3);
            }
            continue;
        }

        // -----------------------------------------------------------------
        //  General-N fallback (prefix-suffix, works for any N >= 2).
        // -----------------------------------------------------------------
        constexpr int DYN_MAX_VEC_CHUNKS = 64;
        dyn_vec_t pref[DYN_MAX_MODES][DYN_MAX_VEC_CHUNKS];
        const value_t* rPtr[DYN_MAX_MODES];
        value_t*       oPtr[DYN_MAX_MODES];
        for (int w = 0; w < N; ++w) {
            rPtr[w] = Y[w]    + (size_t)idx[w] * Rp;
            oPtr[w] = Yout[w] + (size_t)idx[w] * Rp;
        }

        for (int c = 0; c < VC; ++c) pref[0][c] = vval;
        for (int k = 1; k < N; ++k) {
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t y = dyn_vload(rPtr[k - 1] + c * DYN_SIMD_WIDTH);
                pref[k][c] = dyn_vmul(pref[k - 1][c], y);
            }
        }

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

#define DYN_CALL_AM_MORTON(N_T, RP_T) \
    process_range_all_modes_morton_T<N_T, RP_T>(mvals, mkeys, mask_arr,     \
                                                b, e,                        \
                                                num_modes, rank_padded,      \
                                                Y, Yout)

#define DYN_DISPATCH_RP_AM_MORTON(N_T)                                       \
    do {                                                                      \
        if (rank_padded == 16)  { DYN_CALL_AM_MORTON(N_T, 16 ); return; }     \
        if (rank_padded == 32)  { DYN_CALL_AM_MORTON(N_T, 32 ); return; }     \
        if (rank_padded == 64)  { DYN_CALL_AM_MORTON(N_T, 64 ); return; }     \
        if (rank_padded == 128) { DYN_CALL_AM_MORTON(N_T, 128); return; }     \
        if (rank_padded == 256) { DYN_CALL_AM_MORTON(N_T, 256); return; }     \
        DYN_CALL_AM_MORTON(N_T, 0); return;                                   \
    } while (0)

// Public entry.  Specializations: N in {3,4,5} x Rp in {16,32,64,128,256};
// runtime fallback for shapes outside the grid (N=2 / 6+, Rp=48 etc.).
static inline void dynasor_process_shard_all_modes_morton(
    const value_t*  mvals,
    const uint64_t* mkeys,
    const uint64_t* mask_arr,
    uint64_t b, uint64_t e,
    int num_modes, int rank, int rank_padded,
    const value_t* const* Y,
    value_t* const* Yout)
{
    (void)rank;
    if (num_modes == 3) DYN_DISPATCH_RP_AM_MORTON(3);
    if (num_modes == 4) DYN_DISPATCH_RP_AM_MORTON(4);
    if (num_modes == 5) DYN_DISPATCH_RP_AM_MORTON(5);
    DYN_CALL_AM_MORTON(0, 0);
}

#undef DYN_CALL_AM_MORTON
#undef DYN_DISPATCH_RP_AM_MORTON

double spmttkrp_all_modes_morton(Tensor& T, FactorMatrices& F, int num_threads) {
    const int      N   = T.num_modes;
    const int      R   = F.rank;
    const int      Rp  = F.rank_padded;
    const uint64_t nnz = T.nnz;

    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }

    if (T.layout != Layout::Morton || !T.morton_keys || !T.morton_vals) {
        std::fprintf(stderr,
            "[morton] MTTKRP called without valid Morton layout; falling "
            "back to SoA all-modes driver.\n");
        return spmttkrp_all_modes_dynasor(T, F, num_threads);
    }

    // ---- per-mode output buffer layout (same as SoA driver) ----
    std::vector<size_t> ofib_off((size_t)N + 1, 0);
    for (int n = 0; n < N; ++n) {
        size_t rows  = (size_t)T.mode_size[n];
        size_t bytes = rows * (size_t)Rp * sizeof(value_t);
        bytes = (bytes + 63ULL) & ~63ULL;
        ofib_off[n + 1] = ofib_off[n] + bytes / sizeof(value_t);
    }
    const size_t stride_values = ofib_off[N];
    const size_t stride_bytes  = stride_values * sizeof(value_t);
    const size_t total_bytes   = stride_bytes * (size_t)num_threads;

    value_t* ofibs_raw = (value_t*)dyn_aligned_alloc(total_bytes);
    if (!ofibs_raw) {
        std::fprintf(stderr,
            "[morton] unable to allocate %.2f GiB ofibs; falling back.\n",
            (double)total_bytes / (double)(1ULL << 30));
        return spmttkrp_all_modes_dynasor(T, F, num_threads);
    }
    std::printf("[morton] ALL-MODES driver: threads=%d rank=%d (padded=%d) "
                "SIMD=%s  ofibs=%.3f MiB\n",
                num_threads, R, Rp, DYN_SIMD_NAME,
                (double)total_bytes / (double)(1ULL << 20));

    std::vector<std::vector<value_t*>> Yout_by_thread(
        (size_t)num_threads, std::vector<value_t*>((size_t)N, nullptr));
    for (int tid = 0; tid < num_threads; ++tid) {
        value_t* base = ofibs_raw + (size_t)tid * stride_values;
        for (int n = 0; n < N; ++n)
            Yout_by_thread[(size_t)tid][(size_t)n] = base + ofib_off[(size_t)n];
    }

    const double t0 = omp_get_wtime();

    // ---- A. zero per-thread ofibs ----
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

    // ---- B. parallel kernel pass (one Morton slab read) ----
    const uint64_t chunk =
        (nnz + (uint64_t)num_threads - 1) / (uint64_t)num_threads;

    const value_t* const* Y_in =
        const_cast<const value_t* const*>(F.Y.data());

    const uint64_t* DYN_RESTRICT mkeys = T.morton_keys;
    const value_t*  DYN_RESTRICT mvals = T.morton_vals;
    uint64_t        mask_loc[DYN_MAX_MODES];
    for (int n = 0; n < N; ++n) mask_loc[n] = T.morton_mask[n];

    // Stage 4: ask the JIT for an (N, Rp, masks)-baked specialization.
    // Falls back to nullptr when JIT is disabled or the compile failed;
    // in that case we keep using the Stage 3 baked templated kernel.
    jit_all_modes_morton_fn jit_fn =
        dyn_jit_get_all_modes_morton(N, Rp, mask_loc);
    if (jit_fn) {
        std::printf("[morton] using JIT'd am_morton (baked masks, "
                    "N=%d Rp=%d)\n", N, Rp);
    }

    // Stage 3: fused Morton kernel -- pext per-nnz, no batch scratch.
    // One templated (N, Rp) dispatch per thread-shard lets GCC fully
    // unroll the steady-state loop and keep the rank-running products
    // register-resident.
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
                value_t* Yout_local[DYN_MAX_MODES];
                for (int n = 0; n < N; ++n)
                    Yout_local[n] = Yout_by_thread[(size_t)tid][(size_t)n];

                if (jit_fn) {
                    jit_fn(mvals, mkeys, b, e, Y_in, Yout_local);
                } else {
                    dynasor_process_shard_all_modes_morton(
                        mvals,
                        mkeys,
                        mask_loc,
                        b, e,
                        N, R, Rp,
                        Y_in,
                        Yout_local);
                }
            }
        }
    }

    // ---- C. parallel reduction ofibs -> Yhat ----
    for (int n = 0; n < N; ++n) {
        const idx_t rows = T.mode_size[n];
        value_t* DYN_RESTRICT dst = F.Yhat[n];
        const size_t row_stride = (size_t)Rp;

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int64_t r64 = 0; r64 < (int64_t)rows; ++r64) {
            const size_t r = (size_t)r64;
            value_t* DYN_RESTRICT out = dst + r * row_stride;

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

    std::printf("[morton] ALL-MODES spMTTKRP (1 slab pass, all %d modes) "
                "= %.6f s (%.3f Mnnz/s)\n",
                N, secs, (double)nnz / 1e6 / secs);
    return secs;
}

} // namespace dynasor
