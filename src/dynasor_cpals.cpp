// ============================================================================
//  dynasor_cpals.cpp  -- CP-ALS driver on top of SpmttkrpSweep.
//
//  The hot kernel (spMTTKRP) is Dynasor's; everything else in here is
//  R-by-R linear algebra or an O(R * N) reduction.  For any realistic
//  tensor (nnz * N * R >> R^3 * N), CP-ALS throughput is bounded by the
//  Dynasor kernel.
// ============================================================================
#include "dynasor_cpals.h"
#include "dynasor_linalg.h"
#include "dynasor_morton.h"
#include "dynasor_ooc.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
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

namespace {

// Factor-matrix initialisation: i.i.d. standard normal with a deterministic
// per-(mode, row, col) seed, so the decomposition is bitwise reproducible
// across runs and thread counts.  Padded columns are kept zero (the MTTKRP
// kernel relies on those slots being harmless for stride-Rp loads).
void init_random_factors(FactorMatrices& F,
                         const idx_t*    mode_sizes,
                         uint64_t        seed)
{
    const int R  = F.rank;
    const int Rp = F.rank_padded;

    for (int n = 0; n < F.num_modes; ++n) {
        // Hash (seed, n) -> independent sub-seed per mode.  Mixing with a
        // 64-bit splitmix step keeps adjacent modes statistically
        // independent without needing a global PRNG state.
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(n + 1);
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 27; z *= 0x94D049BB133111EBULL;
        z ^= z >> 31;

        std::mt19937_64              rng(z);
        std::normal_distribution<float> N01(0.0f, 1.0f);

        const idx_t rows = mode_sizes[n];
        value_t* Y = F.Y[n];
        for (idx_t i = 0; i < rows; ++i) {
            value_t* yi = Y + (size_t)i * (size_t)Rp;
            for (int r = 0; r < R;  ++r) yi[r] = N01(rng);
            for (int r = R; r < Rp; ++r) yi[r] = (value_t)0;
        }
    }
}

// Compute the CP model norm squared:
//    || [[lambda ; Y]] ||_F^2 = sum_{r, s} lambda[r] lambda[s] * prod_n G[n][r, s]
// with G[n] = Y[n]^T Y[n].
//
// Returned in double so the outer fit calculation can keep full precision
// (||T||^2, <T, M>, and ||M||^2 are three near-equal quantities whose
// difference is the residual; subtracting them in float32 throws away
// exactly the digits that encode the residual).
double model_norm_sq_d(const std::vector<std::vector<value_t>>& G,
                       const std::vector<value_t>&              lambda,
                       int R, int Rp)
{
    const int N = (int)G.size();
    // H := elementwise product of all G[n], accumulated in double so the
    // product of N ~|I_n|-sized reductions does not lose bits.  Seed with
    // the all-ones matrix (Hadamard's neutral element): an identity seed
    // would zero every off-diagonal entry and collapse ||M||^2 to
    // sum_r lambda_r^2 * prod_n G[n][r,r], producing the classic
    // "mnrm2 == <T, M>" symptom on L2-normalised factors.
    std::vector<double> H((size_t)R * (size_t)Rp, 0.0);
    for (int r = 0; r < R; ++r) {
        double* Hr = H.data() + (size_t)r * (size_t)Rp;
        for (int c = 0; c < R; ++c) Hr[c] = 1.0;
    }
    for (int n = 0; n < N; ++n) {
        const value_t* Gn = G[n].data();
        for (int r = 0; r < R; ++r) {
            double*       Hr  = H.data() + (size_t)r * (size_t)Rp;
            const value_t* Gr = Gn       + (size_t)r * (size_t)Rp;
            for (int c = 0; c < R; ++c) Hr[c] *= (double)Gr[c];
        }
    }

    double s = 0;
    for (int r = 0; r < R; ++r) {
        const double lr = (double)lambda[r];
        const double* Hr = H.data() + (size_t)r * (size_t)Rp;
        double rowsum = 0;
        for (int c = 0; c < R; ++c) rowsum += (double)lambda[c] * Hr[c];
        s += lr * rowsum;
    }
    return std::max(0.0, s);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Public CP-ALS entry point.
// ---------------------------------------------------------------------------
CpResult cpals(Tensor& T, FactorMatrices& F, const CpOptions& opt) {
    CpResult result;

    const int N  = T.num_modes;
    const int R  = F.rank;
    const int Rp = F.rank_padded;

    if (N < 2) {
        std::fprintf(stderr, "[cpals] need num_modes >= 2, got %d\n", N);
        return result;
    }
    if (R <= 0) {
        std::fprintf(stderr, "[cpals] need rank >= 1, got %d\n", R);
        return result;
    }

    // --- Initial Y's and Gram matrices --------------------------------------
    init_random_factors(F, T.mode_size, opt.seed);

    // Optional: dump the random initial factors (before any ALS updates)
    // so an external reference implementation can start from the same
    // point.  Used by tools/compare_vs_pyttb.py.
    if (!opt.init_dump_prefix.empty()) {
        for (int n = 0; n < N; ++n) {
            const std::string p = opt.init_dump_prefix
                                + ".cp_init_fm" + std::to_string(n) + ".bin";
            if (!dump_factor_matrix_bin(p, F.Y[n], T.mode_size[n], R, Rp)) {
                std::fprintf(stderr,
                             "[cpals] WARN: failed to dump init factor %s\n",
                             p.c_str());
            }
        }
    }

    std::vector<std::vector<value_t>> G(N, std::vector<value_t>(
                                              (size_t)R * (size_t)Rp, 0));
    for (int n = 0; n < N; ++n)
        la::gram(F.Y[n], T.mode_size[n], R, Rp, G[n].data());

    std::vector<value_t> lambda(R, (value_t)1);

    // || T ||^2 once, up front.  Kept in double: fit = 1 - ||T-M||/||T||
    // subtracts three near-equal quantities; float32 cancellation there
    // is observable for rank-R recovery tests (residual flips to 0).
    //
    // Source-of-vals priority:
    //   1. OOC: use the cached T.ooc_tnorm_sq (computed once at OOC
    //      setup by a streaming pass over the vals column).  No in-RAM
    //      slab exists so we cannot recompute here.
    //   2. T.vals when present (PingPong, or uncompacted NCopy copy 0 --
    //      which aliases primary).
    //   3. T.ncopy_vals[0] as fallback when copy 0 has been CSR-compacted
    //      and the primary slab has been freed.  The compact copy still
    //      owns a contiguous val[] array holding the same nnz scalars,
    //      just in a differently-sized slab.
    double Tnorm2;
    if (T.ooc_enabled) {
        if (T.ooc_tnorm_sq <= 0.0) {
            std::fprintf(stderr,
                "[cpals] fatal: OOC tensor has no cached ||T||^2; "
                "ooc_precompute_norm was not called during setup.\n");
            std::exit(1);
        }
        Tnorm2 = T.ooc_tnorm_sq;
    } else {
        const value_t* vals_for_norm =
            T.vals ? T.vals :
            (T.layout == Layout::NCopy  && T.ncopy_vals[0]) ? T.ncopy_vals[0] :
            (T.layout == Layout::Morton && T.morton_vals  ) ? T.morton_vals   :
            nullptr;
        if (!vals_for_norm) {
            std::fprintf(stderr,
                "[cpals] fatal: no vals pointer available for ||T||^2.\n");
            std::exit(1);
        }
        Tnorm2 = la::tensor_frob_norm_sq(vals_for_norm, T.nnz);
    }
    result.tensor_norm = std::sqrt(Tnorm2);

    // Reusable R-by-R scratch:  V = Hadamard of G[m] for m != n, then
    // Cholesky-factored in place into L = V^{1/2} (lower triangular).
    std::vector<value_t> V((size_t)R * (size_t)Rp, 0);

    if (opt.verbose) {
        std::printf("[cpals] modes=%d  rank=%d  padded=%d  nnz=%llu  "
                    "||T|| = %.6e  max_iters=%d  tol=%.1e  seed=%llu\n",
                    N, R, Rp, (unsigned long long)T.nnz,
                    result.tensor_norm, opt.max_iters, opt.tol,
                    (unsigned long long)opt.seed);
    }

    const double t_start = omp_get_wtime();
    double fit_prev = 0.0;

    // MTTKRP backend selection.  See CpKernel in dynasor_cpals.h.
    //
    //  Three cases:
    //    (1) OOC tensor: must use Jacobi (the OOC driver is all-modes
    //        only; GS would need N disk passes per iter and has no OOC
    //        implementation).  If the caller asked for GS, warn and
    //        promote to Jacobi.  MTTKRP goes through spmttkrp_all_modes_ooc.
    //    (2) In-core + Jacobi: spmttkrp_all_modes_dynasor.
    //    (3) In-core + GS: classic per-mode sweep.
    bool use_jacobi = (opt.kernel == CpKernel::Jacobi);
    const bool use_ooc = T.ooc_enabled;
    if (use_ooc && !use_jacobi) {
        std::fprintf(stderr,
            "[cpals] WARN: tensor is out-of-core; Gauss-Seidel has no OOC "
            "driver.\n"
            "        Promoting --cp-kernel to jacobi (all-modes single-pass).\n");
        use_jacobi = true;
    }
    if (T.layout == Layout::Morton && !use_jacobi) {
        std::fprintf(stderr,
            "[cpals] WARN: tensor is Morton-linearized; Gauss-Seidel has no "
            "Morton driver\n"
            "        (would require a mode-sorted SoA that Morton storage "
            "does not keep).\n"
            "        Promoting --cp-kernel to jacobi (all-modes single-pass).\n");
        use_jacobi = true;
    }

    std::unique_ptr<SpmttkrpSweep> sweep_gs;
    if (!use_jacobi) {
        sweep_gs.reset(new SpmttkrpSweep(T, F, opt.num_threads, opt.verbose));
    }

    if (opt.verbose) {
        const char* label =
            use_ooc    ? "jacobi (out-of-core streaming)" :
            use_jacobi ? "jacobi (all-modes single-pass)" :
                         "gauss-seidel (classical per-mode)";
        std::printf("[cpals] kernel = %s\n", label);
    }

    // Accumulator for all-modes MTTKRP wall time when running Jacobi.
    // SpmttkrpSweep owns the analog for Gauss-Seidel.
    double mttkrp_accum_s = 0.0;

    for (int iter = 0; iter < opt.max_iters; ++iter) {
        const double t_iter0 = omp_get_wtime();

        // --- MTTKRP: either all-modes (Jacobi) or deferred to per-mode ----
        //
        //  Jacobi path: one tensor pass produces every Yhat[n] from the
        //  factors at the START of the iteration.  The per-mode linalg
        //  below then writes the new Y[n], which is NOT re-read by the
        //  remaining MTTKRPs in this iter -- that's the Jacobi property
        //  that decouples the sub-problems but halves the per-iter
        //  tensor bandwidth.
        if (use_jacobi) {
            if (use_ooc) {
                mttkrp_accum_s += spmttkrp_all_modes_ooc(T, F, opt.num_threads);
            } else if (T.layout == Layout::Morton) {
                mttkrp_accum_s += spmttkrp_all_modes_morton(T, F, opt.num_threads);
            } else {
                mttkrp_accum_s += spmttkrp_all_modes_dynasor(T, F, opt.num_threads);
            }
        }

        // --- ALS pass over modes ------------------------------------------
        for (int n = 0; n < N; ++n) {
            // 1. MTTKRP (Gauss-Seidel path only):  Yhat[n] <- T_(n) *
            //    KR(Y_m : m != n).  After process(n), Yhat[n] is valid and
            //    the tensor is remapped for mode (n+1) % N.  On the Jacobi
            //    path we already have Yhat[n] from the single-pass above.
            if (!use_jacobi) {
                sweep_gs->process(n);
            }

            const double t_la0 = omp_get_wtime();

            // 2. V_n = Hadamard of G[m] for m != n.  Start from the
            //    all-ones matrix (Hadamard's neutral element): starting
            //    from an identity matrix would zero every off-diagonal
            //    entry on the first multiply and collapse the LS solve
            //    to a column-wise rescale.
            la::ones(V.data(), R, Rp);
            for (int m = 0; m < N; ++m) {
                if (m == n) continue;
                la::hadamard_inplace(V.data(), G[m].data(), R, Rp);
            }

            // 3. Cholesky factor V_n = L L^T (with a tiny relative ridge
            //    for numerical robustness), then Y[n] = Yhat[n] * (L L^T)^{-1}.
            la::cholesky_lower(V.data(), R, Rp, opt.ridge);
            la::right_solve_chol(V.data(), R, Rp,
                                 F.Yhat[n], T.mode_size[n], F.Y[n]);

            // 4. Normalise columns, recording their L2 norms in lambda.
            //    (Tensor Toolbox switches to max-abs after the first
            //    iteration as a scale-stabilisation trick, but it is not
            //    required for the fit identity <T, M> = ||M||^2 to hold,
            //    and the L2 convention keeps G[n] = Y[n]^T Y[n] with unit
            //    diagonal entries, which dramatically improves the
            //    conditioning of V for later modes.)
            la::col_norms(F.Y[n], T.mode_size[n], R, Rp, lambda.data());
            la::col_scale_inv(F.Y[n], T.mode_size[n], R, Rp, lambda.data());

            // 5. Refresh G[n] = Y[n]^T Y[n] for the remaining modes.
            la::gram(F.Y[n], T.mode_size[n], R, Rp, G[n].data());

            result.linalg_time_s += omp_get_wtime() - t_la0;
        }

        // --- Fit metric using the last mode's Yhat and updated Y ----------
        //
        //  <T, [[lambda; Y]]> = sum_r lambda[r] * <Y[N-1][:, r], Yhat[N-1][:, r]>
        //
        //  ||[[lambda; Y]]||^2 = sum_{r, s} lambda[r] lambda[s] * prod_n G[n][r, s]
        //
        //  || T - M ||^2 = ||T||^2 - 2 <T, M> + ||M||^2
        const double t_la1 = omp_get_wtime();
        const double inner = (double)la::weighted_col_dot(
            F.Y[N - 1], F.Yhat[N - 1], T.mode_size[N - 1], R, Rp, lambda.data());
        const double mnrm2 = model_norm_sq_d(G, lambda, R, Rp);
        result.linalg_time_s += omp_get_wtime() - t_la1;

        double resid2 = Tnorm2 - 2.0 * inner + mnrm2;
        if (resid2 < 0.0) resid2 = 0.0;
        const double residual = std::sqrt(resid2);
        const double fit = 1.0 - residual / std::max(result.tensor_norm, 1e-30);

        // Uncomment for precision diagnostics on ||T||^2, <T,M>, ||M||^2:
        // if (opt.verbose && iter < 3) {
        //     std::printf("[cpals-dbg] iter %d  Tnorm2=%.6e  inner=%.6e  "
        //                 "mnrm2=%.6e  resid2=%.6e\n",
        //                 iter, Tnorm2, inner, mnrm2, resid2);
        // }

        const double iter_time = omp_get_wtime() - t_iter0;
        result.fit_history.push_back(fit);
        result.iter_time_s.push_back(iter_time);
        result.final_fit       = fit;
        result.final_residual  = residual;
        result.iters_done      = iter + 1;

        if (opt.verbose) {
            std::printf("[cpals] iter %3d  fit = %.6f  dfit = %.2e  "
                        "resid = %.3e  iter = %.3f s\n",
                        iter, fit,
                        iter == 0 ? 0.0 : std::abs(fit - fit_prev),
                        residual, iter_time);
        }

        if (iter > 0 && std::abs(fit - fit_prev) < opt.tol) {
            if (opt.verbose) {
                std::printf("[cpals] converged after %d iterations "
                            "(|dfit|=%.2e < %.1e)\n",
                            iter + 1, std::abs(fit - fit_prev), opt.tol);
            }
            break;
        }
        fit_prev = fit;
    }

    result.total_time_s  = omp_get_wtime() - t_start;
    result.mttkrp_time_s = use_jacobi ? mttkrp_accum_s : sweep_gs->elapsed_s();
    result.lambda.assign(lambda.begin(), lambda.end());

    if (opt.verbose) {
        std::printf("[cpals] done  iters=%d  final_fit=%.6f  resid=%.3e\n"
                    "[cpals] total = %.3f s   "
                    "spMTTKRP = %.3f s (%.1f%%)   "
                    "linalg  = %.3f s (%.1f%%)\n",
                    result.iters_done, result.final_fit, result.final_residual,
                    result.total_time_s,
                    result.mttkrp_time_s,
                    100.0 * result.mttkrp_time_s /
                        std::max(result.total_time_s, 1e-30),
                    result.linalg_time_s,
                    100.0 * result.linalg_time_s /
                        std::max(result.total_time_s, 1e-30));
    }

    return result;
}

} // namespace dynasor
