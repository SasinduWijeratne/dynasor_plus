// ============================================================================
//  dynasor_dnp.h -- Dynasor+ Processed-tensor binary format (.dnp).
//
//  The Dynasor+ framework supports two execution modes:
//
//    (1) "Complete processing + runtime" (default).
//        dynasor_plus tensor.tns --decompose cpals --rank R
//          [--save-processed tensor.dnp]
//        Loads raw .tns (or the compact .dnb cache), runs the full
//        preprocessing pipeline (build_flycoo, Morton build, NCopy CSR,
//        etc.), and runs CP-ALS.  With --save-processed, the post-
//        preprocessing state is written to <path>.dnp so subsequent runs
//        can use runtime-only mode.
//
//    (2) "Runtime only" (deployment).
//        dynasor_plus tensor.dnp --runtime --decompose cpals --rank R
//        Loads a previously-produced .dnp file and jumps straight to
//        CP-ALS.  build_flycoo / build_morton_layout / NCopy sort are all
//        skipped.  Useful for production deployments where the expensive
//        one-time preprocessing is amortized across many CP-ALS runs
//        (e.g. different ranks, seeds, or convergence tolerances).
//
//  The .dnp format is self-describing: a fixed header identifies the
//  stored layout (Morton / NCopy / PingPong / InPlace), and the loader
//  routes to the appropriate section parser.  A single binary covers all
//  four in-core layouts; OOC tensors keep using the existing .dnb cache
//  (streaming is already a runtime-only path).
//
//  The schedule (`ss_list`, `precomp_thr_off`) is thread-count specific.
//  The header records the thread count at save time; a load attempt with
//  a different thread count is rejected with an actionable error.
//
//  File layout (little-endian, packed):
//
//    Common header (fixed):
//      char     magic[4]        = "DNP1"
//      uint32_t version         = 1
//      uint32_t value_bytes     = sizeof(value_t)
//      uint32_t index_bytes     = sizeof(idx_t)
//      uint32_t layout_kind     // 0=Morton, 1=NCopy, 2=PingPong, 3=InPlace
//      uint32_t num_modes
//      uint64_t nnz
//      uint32_t mode_size[DYN_MAX_MODES]
//      uint32_t num_threads     // baked-in thread count for schedule
//      double   tnorm_sq        // cached ||T||^2 (0 if unknown)
//      uint32_t reserved[14]    // future flags
//
//    Morton block (layout_kind == 0):
//      uint64_t morton_mask[DYN_MAX_MODES]
//      uint8_t  morton_bits[DYN_MAX_MODES]
//      uint8_t  pad[align_to_8]
//      uint64_t morton_keys[nnz]
//      value_t  morton_vals[nnz]
//
//    For layout_kind in {NCopy, PingPong, InPlace} -- FLYCOO metadata:
//      idx_t    shards_per_mode[DYN_MAX_MODES]
//      idx_t    mn             [DYN_MAX_MODES]
//      int32_t  mn_shift       [DYN_MAX_MODES]
//      idx_t    shard_size
//      double   avg_fiber_len  [DYN_MAX_MODES]
//      uint32_t kernel_class   [DYN_MAX_MODES]
//      int8_t   inplace_sorted_mode
//      uint8_t  pad[7]
//      uint64_t remap_row_stride
//      for n in 0..N-1:
//         uint64_t touched_count
//         uint64_t touched_bytes
//         uint8_t  touched_bits[touched_bytes]
//      for n in 0..N-1:
//         uint64_t shard_count
//         uint64_t shard_begin[shard_count]
//         uint64_t shard_end  [shard_count]
//      for n in 0..N-1:
//         uint64_t num_threads
//         for t in 0..num_threads-1:
//            uint64_t list_len
//            uint64_t ss_list_widened[list_len]
//      for n in 0..N-1:
//         uint64_t thr_off_count
//         uint64_t precomp_thr_off[thr_off_count]
//
//    PingPong / InPlace slab:
//      value_t vals[nnz]
//      idx_t   idx_buf[0..N-1][nnz]
//
//    NCopy slab (per-copy):
//      uint8_t  ncopy_fiber_sorted[DYN_MAX_MODES]
//      uint8_t  ncopy_csr         [DYN_MAX_MODES]
//      for k in 0..N-1:
//         value_t vals_k[nnz]
//         for m in 0..N-1, skip if k==m && ncopy_csr[k]:
//            idx_t idx_k_m[nnz]
//         if ncopy_csr[k]:
//            uint64_t rowptr_k[mode_size[k]+1]
//
//  NOTE: NCopy CSR-compact copies (where the primary/scratch slab has
//        been freed and ix[k] dropped) are NOT supported by the .dnp
//        writer -- save_dnp_tensor returns false in that case and asks
//        the user to re-run with --ncopy-csr-compact=off.  Compaction
//        is a steady-state memory optimization whose savings do not
//        translate into the on-disk cache.
// ============================================================================
#ifndef DYNASOR_DNP_H
#define DYNASOR_DNP_H

#include "dynasor_common.h"

#include <string>

namespace dynasor {

// Save a fully-preprocessed tensor to a .dnp file.  The tensor must have
// completed build_flycoo (plus build_morton_layout when layout == Morton)
// so all scheduler / metadata fields are populated.  Returns false on
// I/O, format, or unsupported-layout errors (e.g. OutOfCore, compacted
// NCopy).
bool save_dnp_tensor(const std::string& path,
                     const Tensor&      T,
                     int                num_threads);

// Peek header metadata without loading bulk buffers.  Populates
// T.num_modes, T.nnz, T.mode_size[], T.layout.  Returns false on error.
bool load_dnp_header(const std::string& path, Tensor& T);

// Peek the thread count baked into a .dnp header without loading.
// Returns -1 on I/O or format errors.
int load_dnp_num_threads(const std::string& path);

// Peek the layout kind baked into a .dnp header without loading.
// Returns Layout::PingPong on error (and emits a stderr message) -- the
// caller should validate with load_dnp_header() first.
Layout load_dnp_layout(const std::string& path);

// Fully load a .dnp file into `T`.  After success, the tensor is ready
// for cpals() / spmttkrp_dynasor() with no further preprocessing.
//
// `num_threads` MUST equal the thread count stored in the file header
// (schedule is thread-specific).  Returns false on I/O, version, or
// thread-count mismatch errors; a detailed stderr diagnostic is emitted
// on failure so the user knows which flag to fix.
bool load_dnp_tensor(const std::string& path,
                     Tensor&            T,
                     int                num_threads);

} // namespace dynasor

#endif // DYNASOR_DNP_H
