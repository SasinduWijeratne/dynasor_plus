// ============================================================================
//  dynasor_morton.h  --  ALTO-style Morton / Z-curve linearized tensor
//  storage + all-modes MTTKRP driver built on top of it.
//
//  Each nonzero is stored as a single 64-bit Morton key (interleaving
//  the bits of its N mode indices with per-mode masks) plus a value.
//  This gives:
//    * 12 B / nnz footprint  (vs 16 B for 3D SoA, 20 B for 4D, 24 B for 5D),
//    * implicit all-mode locality: consecutive nnz are spatially close
//      in every mode simultaneously, so the all-modes MTTKRP kernel
//      gets better Yhat-row cache reuse than per-mode sorts can
//      provide.
//
//  Build path: (1) compute per-mode bit widths b_n = ceil(log2(I_n))
//  and bit plan (round-robin LSB-first assignment); (2) pack each nnz
//  using pdep; (3) sort (key, val) pairs by key ascending; (4) free
//  the original SoA idx columns.
//
//  Kernel path (see spmttkrp_all_modes_morton): strip-mine the sorted
//  (key, val) stream into batches of BATCH nnz, unpack the keys into
//  a thread-local idx_t[N][BATCH] buffer using pext, then dispatch
//  the existing templated all-modes kernel on that batch.  The
//  permanent storage stays 12 B/nnz -- the SoA batch buffer is L1-
//  resident scratch.
//
//  Applicability: sum_n ceil(log2(I_n)) <= 64.  The planner declines
//  Morton when this would overflow.
// ============================================================================
#ifndef DYNASOR_MORTON_H
#define DYNASOR_MORTON_H

#include "dynasor_common.h"

#include <cstdint>

namespace dynasor {

// ---------------------------------------------------------------------------
//  Build the Morton bit plan: per-mode bit count and pdep/pext mask.
//
//  Returns false iff sum_n ceil(log2(mode_size[n])) > 64 (the whole key
//  would overflow), in which case Morton cannot be used on this tensor.
//
//  Output arrays must have room for num_modes entries.
// ---------------------------------------------------------------------------
bool morton_build_plan(int             num_modes,
                       const idx_t*    mode_size,
                       uint64_t*       mask_out,      // [num_modes]
                       uint8_t*        bits_out);     // [num_modes]

// ---------------------------------------------------------------------------
//  Convert an already-loaded SoA tensor (T.vals + T.idx_buf[*]) into the
//  Morton layout in place:
//    1. Compute per-mode bit plan via morton_build_plan.
//    2. Allocate (morton_keys, morton_vals) = 12 B / nnz (aligned).
//    3. Pack keys from T.idx_buf[*]; copy T.vals into morton_vals.
//    4. Sort (key, val) pairs by key ascending (parallel quicksort /
//       LSD radix sort -- implementation chooses).
//    5. Free the SoA slabs (buf_raw, buf_scratch_raw, inplace_scr_raw)
//       and null all SoA pointers so no consumer can mistake them for
//       live data.
//    6. Set T.layout = Layout::Morton.
//
//  Returns true on success.  On failure (plan overflow or OOM), T is
//  left untouched and the caller should pick a different layout.
// ---------------------------------------------------------------------------
bool build_morton_layout(Tensor& T, int num_threads);

// ---------------------------------------------------------------------------
//  All-modes single-pass MTTKRP over Morton-stored tensor.
//
//  Semantically identical to spmttkrp_all_modes_dynasor: reads T once,
//  fills every F.Yhat[n] from a consistent snapshot of F.Y[*].  Uses
//  per-thread private ofibs + a final parallel reduction, same as the
//  SoA all-modes driver.
//
//  Requires T.layout == Layout::Morton with valid morton_* state.
//  Returns elapsed seconds.
// ---------------------------------------------------------------------------
double spmttkrp_all_modes_morton(Tensor& T, FactorMatrices& F, int num_threads);

} // namespace dynasor

#endif // DYNASOR_MORTON_H
