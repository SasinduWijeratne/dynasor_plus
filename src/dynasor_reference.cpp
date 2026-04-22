// ============================================================================
//  dynasor_reference.cpp
//
//  Plain single-threaded spMTTKRP (one pass over the tensor per mode) used
//  as the ground-truth oracle for correctness tests.  No SIMD, no FLYCOO,
//  no dynamic remapping -- just the textbook definition.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_simd.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace dynasor {

void spmttkrp_reference(const Tensor& T, FactorMatrices& F) {
    const int      N   = T.num_modes;
    const int      R   = F.rank;
    const int      Rp  = F.rank_padded;
    const uint64_t nnz = T.nnz;

    std::printf("[reference] rank=%d  modes=%d  nnz=%llu\n",
                R, N, (unsigned long long)nnz);

    std::vector<value_t> tmp(Rp);

    for (int n = 0; n < N; ++n) {
        std::memset(F.Yhat[n],
                    0,
                    (size_t)T.mode_size[n] * Rp * sizeof(value_t));

        for (uint64_t i = 0; i < nnz; ++i) {
            const value_t val = T.vals[i];
            if (val == 0.0f) continue;

            for (int r = 0; r < R; ++r) tmp[r] = val;

            for (int w = 0; w < N; ++w) {
                if (w == n) continue;
                const value_t* row = F.Y[w] + (size_t)T.idx_buf[w][i] * Rp;
                for (int r = 0; r < R; ++r) tmp[r] *= row[r];
            }

            value_t* out = F.Yhat[n] + (size_t)T.idx_buf[n][i] * Rp;
            for (int r = 0; r < R; ++r) out[r] += tmp[r];
        }
    }
}

// ---------------------------------------------------------------------------
//  Max relative error between two factor-matrix sets (typically dynasor
//  output vs reference).  Uses the same tolerance convention as the
//  original compare_2files.cpp but returns a single numeric summary.
// ---------------------------------------------------------------------------
// Minimal shape check; real numeric comparison lives in
// compare_factors_detail (which also needs the per-mode row counts).
double compare_factors(const FactorMatrices& A, const FactorMatrices& B) {
    if (A.num_modes != B.num_modes || A.rank != B.rank) {
        std::fprintf(stderr, "compare_factors: shape mismatch\n");
        return 1.0;
    }
    return 0.0;
}

// Actually-used comparison: caller passes mode sizes explicitly.
double compare_factors_detail(const FactorMatrices& A,
                              const FactorMatrices& B,
                              const idx_t* mode_sizes)
{
    // We compare both (i) element-wise max absolute / relative error and
    // (ii) the Frobenius-norm relative error per mode,
    //           ||A_n - B_n||_F / ||B_n||_F .
    //
    // The Frobenius metric is the one used in most CP/Tucker test suites --
    // it tolerates the unavoidable float32 summation-reorder noise that
    // differs between the single-thread reference and the parallel
    // implementation but still catches logic bugs.
    double max_abs   = 0.0;
    double max_rel   = 0.0;
    double worst_fro = 0.0;

    for (int n = 0; n < A.num_modes; ++n) {
        const value_t* a = A.Yhat[n];
        const value_t* b = B.Yhat[n];
        const int      R = A.rank;
        const int      Rp = A.rank_padded;
        const idx_t    rows = mode_sizes[n];

        double ss_diff = 0.0;
        double ss_ref  = 0.0;
        for (idx_t i = 0; i < rows; ++i) {
            for (int r = 0; r < R; ++r) {
                float va = a[(size_t)i * Rp + r];
                float vb = b[(size_t)i * Rp + r];
                float diff = std::fabs(va - vb);
                float scale = std::fmax(std::fabs(va), std::fabs(vb));
                double rel = (scale > 1e-8) ? diff / scale : diff;
                if (rel  > max_rel) max_rel = rel;
                if (diff > max_abs) max_abs = diff;
                ss_diff += (double)diff * diff;
                ss_ref  += (double)vb   * vb;
            }
        }
        double fro = (ss_ref > 0.0) ? std::sqrt(ss_diff / ss_ref)
                                    : std::sqrt(ss_diff);
        if (fro > worst_fro) worst_fro = fro;
        std::printf("[compare] mode %d : frob_rel = %.3e\n", n, fro);
    }

    std::printf("[compare] max |abs|           = %.6g\n", max_abs);
    std::printf("[compare] max per-entry |rel| = %.6g (noisy near cancellations)\n", max_rel);
    std::printf("[compare] worst Frobenius rel = %.6g\n", worst_fro);
    return worst_fro;
}

} // namespace dynasor
