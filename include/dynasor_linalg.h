// ============================================================================
//  dynasor_linalg.h  -- small-R dense linear algebra used by the CP-ALS
//  driver.  Everything in here is sized by the factor rank R (typically
//  16-256), *not* by nnz or mode_size, so the implementations are kept
//  deliberately simple: parallelise over the I=mode_size dimension when it
//  helps, keep the R-by-R core single-threaded.
//
//  Conventions:
//     * Factor matrices are row-major, I rows of `rank_padded` columns.
//       Only the leading R columns carry useful data; the trailing
//       `rank_padded - R` slots are kept zero (set at init time) and are
//       there so every row is SIMD-friendly.
//     * R-by-R matrices are row-major with leading dimension == rank_padded
//       so the same SIMD loads/stores used for factors apply.
//     * Lower-triangular Cholesky factors are stored in the lower triangle
//       of the same R-by-R layout; the upper triangle is ignored by the
//       solver but not explicitly zeroed.
// ============================================================================
#ifndef DYNASOR_LINALG_H
#define DYNASOR_LINALG_H

#include "dynasor_common.h"

#include <cstddef>
#include <cstdint>

namespace dynasor {
namespace la {

// G (R x R, ld = rank_padded) := Y^T Y, where Y is I x R with ld =
// rank_padded.  Parallelised over I with thread-local R-by-R accumulators,
// then reduced once at the end.  Cost O(I * R^2), dominated by the I-loop.
void gram(const value_t* Y,
          std::size_t    rows,
          int            rank,
          int            rank_padded,
          value_t*       G);

// In-place Hadamard product: A[i,j] *= B[i,j] for i,j in [0, R).  Ignores
// the padding columns (caller keeps them zero; multiplying would leave
// zeros anyway).
void hadamard_inplace(value_t*       A,
                      const value_t* B,
                      int            rank,
                      int            rank_padded);

// Initialise A as the R x R identity (padding columns zeroed).
void identity(value_t* A, int rank, int rank_padded);

// Initialise A to the R x R all-ones matrix (padding columns zeroed).
// This is the Hadamard-product neutral element -- using identity() here
// silently zeroes every off-diagonal entry on the first Hadamard multiply,
// which collapses V into a diagonal matrix and turns the CP-ALS update
// into a meaningless per-column rescale.  Always start Hadamard chains
// with ones().
void ones(value_t* A, int rank, int rank_padded);

// Return A[i,i] * B[i,i] summed over i -- cheap helper used by the CP
// model-norm expression.
value_t sum_diag_product(const value_t* A,
                         const value_t* B,
                         int            rank,
                         int            rank_padded);

// In-place lower Cholesky V = L * L^T (V on entry, L overwritten in place
// in the lower triangle).  Adds `ridge * max(diag(V))` to each diagonal
// BEFORE factoring, guaranteeing SPD even when V is numerically rank-
// deficient (common for CP-ALS updates when two factor columns are nearly
// colinear).  Returns the effective ridge used, so the caller can log it.
value_t cholesky_lower(value_t* V,
                       int      rank,
                       int      rank_padded,
                       value_t  ridge);

// Solve X * (L L^T) = B for X, where L is lower-triangular R x R, B is
// I x R (= Yhat), and X is I x R (= updated Y).  Works row-by-row:
// solve L * z = b^T for z, then L^T * x = z for x, writing x into X's row.
// Parallelised over I.  B and X may alias safely (row-by-row).
void right_solve_chol(const value_t* L,
                      int            rank,
                      int            rank_padded,
                      const value_t* B,
                      std::size_t    rows,
                      value_t*       X);

// Column L2 norms of Y (I x R, ld = rank_padded).  Writes R values into
// norms[0..R).  Padded columns are ignored.
void col_norms(const value_t* Y,
               std::size_t    rows,
               int            rank,
               int            rank_padded,
               value_t*       norms);

// Y[:, r] /= max(norms[r], eps)  for r in [0, R).  Parallelised over I.
// The clamp keeps zero columns (which happen when rank > tensor rank)
// from producing NaNs.
void col_scale_inv(value_t*       Y,
                   std::size_t    rows,
                   int            rank,
                   int            rank_padded,
                   const value_t* norms,
                   value_t        eps = 1e-20f);

// trace(diag(w) * A^T * B), i.e. sum_{r} w[r] * <A[:,r], B[:,r]>, with
// A and B both I x R (ld = rank_padded).  Parallelised over I with
// thread-local R-vectors reduced at the end.  Used for the <T, [[lambda;Y]]>
// term of the CP fit when A = Yhat[N-1] and B = Y[N-1].
value_t weighted_col_dot(const value_t* A,
                         const value_t* B,
                         std::size_t    rows,
                         int            rank,
                         int            rank_padded,
                         const value_t* w);

// Frobenius norm squared of a factor matrix (I x R, ld = rank_padded),
// parallel sum over rows.
value_t frob_norm_sq(const value_t* Y,
                     std::size_t    rows,
                     int            rank,
                     int            rank_padded);

// Frobenius norm squared of the tensor values (T.vals[0..nnz)),
// parallel reduction.  Cached by the CP-ALS driver since T is read-only.
// Returned in double: the value is used in the fit metric where it is
// subtracted from two other near-equal quantities, so float precision
// would be catastrophic.
double tensor_frob_norm_sq(const value_t* vals, std::uint64_t nnz);

} // namespace la
} // namespace dynasor

#endif // DYNASOR_LINALG_H
