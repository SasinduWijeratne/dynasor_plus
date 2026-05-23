// ============================================================================
//  dynasor_linalg.cpp  -- small-R dense linear algebra kernels used by the
//  CP-ALS driver.  See dynasor_linalg.h for the API contract.
//
//  Performance notes:
//    * All kernels whose outer-loop dimension is the factor I=mode_size
//      are OpenMP-parallelised (gram, solve, col_scale_inv, norms, ...).
//      The inner dimension R is typically <= 256, so SIMD over R via the
//      portable dyn_v* intrinsics is almost always a win.
//    * The R-by-R Cholesky itself is tiny (R^3/6 flops, ~2.8 Mflop at
//      R=256) so we leave it serial to avoid parallel-launch overhead.
//    * Gram uses a per-thread RxR accumulator followed by a single
//      reduction, same pattern as the MTTKRP scatter plan -- avoids
//      atomic updates on the small matrix and keeps the inner loop
//      cache-friendly.
// ============================================================================
#include "dynasor_linalg.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dynasor {
namespace la {

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------
static inline value_t* alloc_mat(int rank, int rank_padded, int copies = 1) {
    const size_t bytes =
        (size_t)copies * (size_t)rank * (size_t)rank_padded * sizeof(value_t);
    void* p = dyn_aligned_alloc(bytes);
    std::memset(p, 0, bytes);
    return (value_t*)p;
}
static inline void free_mat(value_t* p) { dyn_aligned_free(p); }

// ---------------------------------------------------------------------------
//  gram:  G(R x R, ld=Rp) := Y^T Y, where Y is (rows x R, ld=Rp).
// ---------------------------------------------------------------------------
void gram(const value_t* Y,
          std::size_t    rows,
          int            rank,
          int            rank_padded,
          value_t*       G)
{
    const int R  = rank;
    const int Rp = rank_padded;

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    // One RxR accumulator per thread to avoid contention on G's cache lines.
    std::vector<value_t*> locals(nt, nullptr);
    for (int t = 0; t < nt; ++t) locals[t] = alloc_mat(R, Rp);

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        value_t* Gl = locals[tid];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)rows; ++i) {
            const value_t* yi = Y + (size_t)i * (size_t)Rp;
            // Outer over row r; inner over column c uses SIMD add via the
            // compiler (simple, small R, no aliasing).
            for (int r = 0; r < R; ++r) {
                const value_t yir = yi[r];
                if (yir == (value_t)0) continue;
                value_t* Gr = Gl + (size_t)r * (size_t)Rp;
                for (int c = 0; c < R; ++c)
                    Gr[c] += yir * yi[c];
            }
        }
    }

    // Reduce locals into G.
    std::memset(G, 0, (size_t)R * (size_t)Rp * sizeof(value_t));
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < R; ++r) {
            value_t*       Gr  = G           + (size_t)r * (size_t)Rp;
            const value_t* Glr = locals[t]   + (size_t)r * (size_t)Rp;
            for (int c = 0; c < R; ++c) Gr[c] += Glr[c];
        }
        free_mat(locals[t]);
    }
}

// ---------------------------------------------------------------------------
//  hadamard_inplace:  A := A .* B.
// ---------------------------------------------------------------------------
void hadamard_inplace(value_t*       A,
                      const value_t* B,
                      int            rank,
                      int            rank_padded)
{
    const int R  = rank;
    const int Rp = rank_padded;
    for (int r = 0; r < R; ++r) {
        value_t*       Ar = A + (size_t)r * (size_t)Rp;
        const value_t* Br = B + (size_t)r * (size_t)Rp;
        for (int c = 0; c < R; ++c) Ar[c] *= Br[c];
    }
}

// ---------------------------------------------------------------------------
//  identity:  A := I_R  (padding columns/rows stay zero).
// ---------------------------------------------------------------------------
void identity(value_t* A, int rank, int rank_padded) {
    std::memset(A, 0, (size_t)rank * (size_t)rank_padded * sizeof(value_t));
    for (int r = 0; r < rank; ++r)
        A[(size_t)r * (size_t)rank_padded + r] = (value_t)1;
}

void ones(value_t* A, int rank, int rank_padded) {
    const int R  = rank;
    const int Rp = rank_padded;
    // Padded columns kept zero so SIMD loads through them stay well-defined.
    for (int r = 0; r < R; ++r) {
        value_t* Ar = A + (size_t)r * (size_t)Rp;
        for (int c = 0; c < R;  ++c) Ar[c] = (value_t)1;
        for (int c = R; c < Rp; ++c) Ar[c] = (value_t)0;
    }
}

// ---------------------------------------------------------------------------
//  sum_diag_product:  sum_i A[i,i] * B[i,i].
// ---------------------------------------------------------------------------
value_t sum_diag_product(const value_t* A,
                         const value_t* B,
                         int            rank,
                         int            rank_padded)
{
    value_t s = 0;
    for (int r = 0; r < rank; ++r) {
        const size_t k = (size_t)r * (size_t)rank_padded + r;
        s += A[k] * B[k];
    }
    return s;
}

// ---------------------------------------------------------------------------
//  cholesky_lower:  V := L  (lower-triangular), V = L L^T on entry.
//  Adds ridge * max(diag(V)) to the diagonal before factoring.  Pivoting
//  is unnecessary for SPD matrices; the ridge guarantees SPD even when V
//  is rank-deficient in floating point.
// ---------------------------------------------------------------------------
value_t cholesky_lower(value_t* V,
                       int      rank,
                       int      rank_padded,
                       value_t  ridge)
{
    const int R  = rank;
    const int Rp = rank_padded;

    // Determine the ridge to apply from the Frobenius / diagonal scale.
    value_t dmax = 0;
    for (int r = 0; r < R; ++r) {
        const value_t d = V[(size_t)r * (size_t)Rp + r];
        if (d > dmax) dmax = d;
    }
    const value_t eps = std::max(ridge * dmax, (value_t)0);
    for (int r = 0; r < R; ++r)
        V[(size_t)r * (size_t)Rp + r] += eps;

    // Standard right-looking serial Cholesky.  R is small (<= 256), so
    // the O(R^3/6) work is negligible; we favour readability over clever
    // blocking.  Writes only the lower triangle.
    for (int j = 0; j < R; ++j) {
        value_t* Vj = V + (size_t)j * (size_t)Rp;
        // Diagonal: a[j,j] = sqrt( a[j,j] - sum_{k<j} a[j,k]^2 ).
        value_t s = Vj[j];
        for (int k = 0; k < j; ++k) s -= Vj[k] * Vj[k];
        if (s < (value_t)0) s = (value_t)0;  // defensive
        const value_t djj = std::sqrt(s);
        Vj[j] = djj;
        const value_t inv = (djj > (value_t)0) ? ((value_t)1 / djj) : (value_t)0;

        // Below-diagonal in column j: a[i,j] = (a[i,j] - sum_{k<j} a[i,k] a[j,k]) / a[j,j]
        for (int i = j + 1; i < R; ++i) {
            value_t* Vi = V + (size_t)i * (size_t)Rp;
            value_t t = Vi[j];
            for (int k = 0; k < j; ++k) t -= Vi[k] * Vj[k];
            Vi[j] = t * inv;
        }
    }
    return eps;
}

// ---------------------------------------------------------------------------
//  right_solve_chol:  X (rows x R) := B (rows x R) * (L L^T)^{-1}.
//  For each row b of B:
//      solve  L   z = b    (forward  substitution, size R)
//      solve  L^T x = z    (backward substitution, size R)
//  x is written into the corresponding row of X.  O(rows * R^2), the
//  rows dimension is the one worth parallelising.
// ---------------------------------------------------------------------------
void right_solve_chol(const value_t* L,
                      int            rank,
                      int            rank_padded,
                      const value_t* B,
                      std::size_t    rows,
                      value_t*       X)
{
    const int R  = rank;
    const int Rp = rank_padded;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)rows; ++i) {
        const value_t* bi = B + (size_t)i * (size_t)Rp;
        value_t*       xi = X + (size_t)i * (size_t)Rp;

        // Copy bi -> xi (we solve in place in xi).
        for (int c = 0; c < R;  ++c) xi[c] = bi[c];
        for (int c = R; c < Rp; ++c) xi[c] = (value_t)0;

        // Forward: L z = b.  L is lower-tri in row-major with ld=Rp.
        for (int r = 0; r < R; ++r) {
            const value_t* Lr = L + (size_t)r * (size_t)Rp;
            value_t s = xi[r];
            for (int k = 0; k < r; ++k) s -= Lr[k] * xi[k];
            const value_t d = Lr[r];
            xi[r] = (d > (value_t)0) ? (s / d) : (value_t)0;
        }
        // Backward: L^T x = z.  L^T[r,c] = L[c,r].
        for (int r = R - 1; r >= 0; --r) {
            value_t s = xi[r];
            for (int k = r + 1; k < R; ++k)
                s -= L[(size_t)k * (size_t)Rp + r] * xi[k];
            const value_t d = L[(size_t)r * (size_t)Rp + r];
            xi[r] = (d > (value_t)0) ? (s / d) : (value_t)0;
        }
    }
}

// ---------------------------------------------------------------------------
//  col_norms:  norms[c] := ||Y[:, c]||_2  for c in [0, R).
// ---------------------------------------------------------------------------
void col_norms(const value_t* Y,
               std::size_t    rows,
               int            rank,
               int            rank_padded,
               value_t*       norms)
{
    const int R  = rank;
    const int Rp = rank_padded;

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    // Per-thread R-vector of squared sums, reduced serially at the end.
    std::vector<std::vector<value_t>> locals(nt, std::vector<value_t>(R, 0));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        value_t* acc = locals[tid].data();

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)rows; ++i) {
            const value_t* yi = Y + (size_t)i * (size_t)Rp;
            for (int c = 0; c < R; ++c) acc[c] += yi[c] * yi[c];
        }
    }

    for (int c = 0; c < R; ++c) {
        value_t s = 0;
        for (int t = 0; t < nt; ++t) s += locals[t][c];
        norms[c] = std::sqrt(s);
    }
}

// ---------------------------------------------------------------------------
//  col_scale_inv:  Y[:, c] /= max(norms[c], eps).
// ---------------------------------------------------------------------------
void col_scale_inv(value_t*       Y,
                   std::size_t    rows,
                   int            rank,
                   int            rank_padded,
                   const value_t* norms,
                   value_t        eps)
{
    const int R  = rank;
    const int Rp = rank_padded;

    // Precompute inverse scales once.
    std::vector<value_t> inv(R);
    for (int c = 0; c < R; ++c) {
        const value_t n = norms[c];
        inv[c] = (n > eps) ? ((value_t)1 / n) : (value_t)0;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)rows; ++i) {
        value_t* yi = Y + (size_t)i * (size_t)Rp;
        for (int c = 0; c < R; ++c) yi[c] *= inv[c];
    }
}

// ---------------------------------------------------------------------------
//  weighted_col_dot:  sum_r w[r] * <A[:,r], B[:,r]>.
//  Accumulates in double precision so the fit metric stays well-posed
//  when <T, M> and ||M||^2 are both ~ ||T||^2 and the CP model nearly
//  recovers T (the float32 residual can otherwise flip negative).
// ---------------------------------------------------------------------------
value_t weighted_col_dot(const value_t* A,
                         const value_t* B,
                         std::size_t    rows,
                         int            rank,
                         int            rank_padded,
                         const value_t* w)
{
    const int R  = rank;
    const int Rp = rank_padded;

#ifdef _OPENMP
    const int nt = omp_get_max_threads();
#else
    const int nt = 1;
#endif
    std::vector<std::vector<double>> locals(nt, std::vector<double>(R, 0.0));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* acc = locals[tid].data();

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)rows; ++i) {
            const value_t* ai = A + (size_t)i * (size_t)Rp;
            const value_t* bi = B + (size_t)i * (size_t)Rp;
            for (int c = 0; c < R; ++c)
                acc[c] += (double)ai[c] * (double)bi[c];
        }
    }

    double s = 0;
    for (int c = 0; c < R; ++c) {
        double colsum = 0;
        for (int t = 0; t < nt; ++t) colsum += locals[t][c];
        s += (double)w[c] * colsum;
    }
    return (value_t)s;
}

// ---------------------------------------------------------------------------
//  frob_norm_sq: sum over all R "real" entries of Y.
// ---------------------------------------------------------------------------
value_t frob_norm_sq(const value_t* Y,
                     std::size_t    rows,
                     int            rank,
                     int            rank_padded)
{
    const int R  = rank;
    const int Rp = rank_padded;
    double s = 0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for (int64_t i = 0; i < (int64_t)rows; ++i) {
        const value_t* yi = Y + (size_t)i * (size_t)Rp;
        double loc = 0;
        for (int c = 0; c < R; ++c) loc += (double)yi[c] * (double)yi[c];
        s += loc;
    }
    return (value_t)s;
}

// ---------------------------------------------------------------------------
//  tensor_frob_norm_sq: sum_i vals[i]^2.  Computed in double precision so
//  the accumulated rounding error does not dominate the fit metric when
//  ||T||^2 is 10^6..10^10 and we subtract ||T - M||^2 from it.
// ---------------------------------------------------------------------------
double tensor_frob_norm_sq(const value_t* vals, std::uint64_t nnz) {
    double s = 0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for (int64_t i = 0; i < (int64_t)nnz; ++i) {
        const double v = (double)vals[i];
        s += v * v;
    }
    return s;
}

} // namespace la
} // namespace dynasor
