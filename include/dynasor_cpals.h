// ============================================================================
//  dynasor_cpals.h  -- CP (canonical polyadic) decomposition via the ALS
//  (Alternating Least Squares) algorithm, built on top of Dynasor's
//  per-mode spMTTKRP sweep.
//
//  For a tensor T with N modes of sizes I_0..I_{N-1} and target rank R,
//  CP-ALS fits the model
//
//      T  ~=  [[ lambda ; Y[0], Y[1], ..., Y[N-1] ]]
//
//                    R
//      T[i] ~= sum  lambda[r] * Y[0][i_0, r] * ... * Y[N-1][i_{N-1}, r]
//                   r=1
//
//  by cycling through modes and, for each mode n, solving the normal
//  equations
//
//      Y[n] = Yhat[n] * (V_n)^{-1}
//
//  where Yhat[n] is the MTTKRP output and
//
//      V_n = ( Y[0]^T Y[0] ) .* ... .* ( Y[N-1]^T Y[N-1] )   excluding mode n
//
//  is the R-by-R Khatri-Rao Gram product.  Each column of the new Y[n] is
//  then normalised into the lambda vector.
//
//  The fit is reported as
//
//      fit = 1 - || T - [[lambda ; Y]] || / || T ||
//
//  with ||T||^2 cached once and the cross term <T, [[lambda ; Y]]>
//  evaluated via the identity
//
//      <T, [[lambda ; Y]]> = sum_r lambda[r] * sum_k Y[n][k, r] * Yhat[n][k, r]
//
//  which lets us reuse the last mode's MTTKRP output for free -- no extra
//  tensor pass per iteration.
// ============================================================================
#ifndef DYNASOR_CPALS_H
#define DYNASOR_CPALS_H

#include "dynasor_common.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dynasor {

// ----------------------------------------------------------------------------
//  MTTKRP kernel strategy used inside each ALS iteration.
//
//  GaussSeidel (default, classical CP-ALS):
//    For each mode n = 0..N-1:
//        Yhat[n] <- MTTKRP(T, Y[*] excluding n)    // 1 tensor pass per mode
//        Y[n]    <- solve/normalize/update         // updated factor is
//                                                   // visible to the next
//                                                   // mode's MTTKRP in the
//                                                   // same iteration.
//    Total tensor bandwidth per iter: N passes.
//    Convergence: fast (each sub-problem uses freshly updated factors).
//
//  Jacobi (all-modes single-pass):
//    At iter start, snapshot factors Y_old[*] = Y[*].
//    In ONE tensor pass compute every Yhat[n] from Y_old[*] using the
//    all-modes MTTKRP kernel.  Then independently solve/normalize/update
//    each Y[n] from its Yhat[n].
//    Total tensor bandwidth per iter: 1 pass.
//    Convergence: slower per iter (decoupled sub-problems), but per-iter
//    MTTKRP cost is ~Nx cheaper, so wall-clock to a fixed fit target is
//    often the better number to report for large tensors where MTTKRP
//    dominates.
//
//  Auto: pick GaussSeidel unless --all-modes is explicitly on, matching
//  the standalone MTTKRP benchmark's convention.
// ----------------------------------------------------------------------------
enum class CpKernel {
    GaussSeidel,    // classical per-mode CP-ALS; one tensor pass per mode
    Jacobi,         // all-modes single-pass MTTKRP per iter
};

struct CpOptions {
    int         max_iters    = 50;
    double      tol          = 1e-4;    // stop when |fit_i - fit_{i-1}| < tol
    uint64_t    seed         = 42;
    int         num_threads  = 0;       // 0 means: use the OMP default.
    value_t     ridge        = 1e-10f;  // relative ridge for the Cholesky solve
    bool        verbose      = true;    // print per-iteration summary

    // MTTKRP kernel strategy (see CpKernel above).  Default is the
    // classical Gauss-Seidel CP-ALS; set to Jacobi (or pass
    // --cp-kernel jacobi) to use the all-modes single-pass kernel.
    CpKernel    kernel       = CpKernel::GaussSeidel;

    // Optional: dump the random initial factors right after seeding,
    // before the first ALS sweep.  Used by external validators (e.g.
    // tools/compare_vs_pyttb.py) that want to run a reference CP-ALS
    // implementation starting from the exact same initial guess, so
    // the per-iteration trajectory can be compared directly.
    std::string init_dump_prefix;   // writes <prefix>.cp_init_fm<n>.bin
};

struct CpResult {
    int                 iters_done     = 0;
    double              final_fit      = 0.0;  // last fit value seen
    double              final_residual = 0.0;  // || T - M ||
    double              tensor_norm    = 0.0;  // || T ||
    double              total_time_s   = 0.0;
    double              mttkrp_time_s  = 0.0;
    double              linalg_time_s  = 0.0;
    std::vector<double> fit_history;           // one entry per completed iter
    std::vector<double> iter_time_s;
    std::vector<value_t> lambda;               // length = rank
};

// Run CP-ALS on (T, F).  F must already be allocated (via FactorMatrices::
// allocate) with the desired rank; the Y[n] matrices will be overwritten
// with Gaussian-random initial values seeded by `opt.seed` and then
// iteratively refined.  On exit, F.Y[n] holds the (column-normalised)
// factor matrices and result.lambda holds the R column weights.
CpResult cpals(Tensor& T, FactorMatrices& F, const CpOptions& opt);

} // namespace dynasor

#endif // DYNASOR_CPALS_H
