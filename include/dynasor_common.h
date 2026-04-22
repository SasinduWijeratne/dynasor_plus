// ============================================================================
//  Dynasor+ : Dynamic Memory Layout spMTTKRP for multi-core CPUs
//  (portable vectorized implementation for x86 and ARM)
//
//  Reference:
//    S. Wijeratne, R. Kannan, V. Prasanna,
//    "Dynasor: A Dynamic Memory Layout for Accelerating Sparse MTTKRP
//     for Tensor Decomposition on Multi-core CPU",
//     arXiv:2309.09131, 2023.
// ============================================================================
#ifndef DYNASOR_COMMON_H
#define DYNASOR_COMMON_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace dynasor {

// Maximum number of modes the build supports (compile-time upper bound on
// the per-element metadata footprint).  The value 8 is more than enough for
// every tensor in the FROSTT corpus used by the paper.
#ifndef DYN_MAX_MODES
#define DYN_MAX_MODES 8
#endif

using idx_t   = uint32_t;           // tensor index (per-mode), zero-based
using sid_t   = uint32_t;           // shard id (zero-based)
using value_t = float;              // floating-point type used for values/factors

// ============================================================================
//  Tensor in FLYCOO form  --  Structure-of-Arrays layout.
//
//  For every nonzero i in [0, nnz):
//      vals   [i]            the scalar value
//      idx_buf[n][i]         the index in mode n           (0 <= n < num_modes)
//
//  The shard id for mode n of element i is *derived* on the fly as
//
//      sid = idx_buf[n][i] >> mn_shift[n]
//
//  where mn[n] is constrained to be a power of two (rounded up in pick_mn).
//  This eliminates an entire 32 B/element array (shard_ids[]) from the
//  hot path and compresses the per-element working-set by 3-4x on 3D
//  tensors relative to a plain AoS layout.
//
//  A matching "scratch" set (scr_vals / scr_idx[]) lives in the same
//  backing allocation and is used as the remap destination.  Every mode
//  ends with a pointer swap -- no data movement in the outer driver.
//
//  All six pointer groups are backed by a SINGLE 64 B-aligned allocation
//  (buf_raw), which:
//     * gives one malloc instead of 2*(N+1)  (big deal for large tensors),
//     * keeps NUMA first-touch coherent (all arrays start in the same pool),
//     * skips the zero-init that std::vector::resize would do,
//     * avoids per-element object-slot padding (sizeof(Element)=72 was
//       ~4x the useful payload per element for N=3).
// ============================================================================
// ---------------------------------------------------------------------------
//  Per-mode kernel classification produced at preprocessing time.
//  The metric is the average Yhat-row reuse opportunity when writing into
//  Yhat[n]: roughly nnz / mode_size[n].  Very short fibers (<8 on average)
//  mean the current element-at-a-time kernel is optimal; long fibers mean
//  a fiber-unrolled kernel could reuse each Yhat row many times and save
//  memory traffic.  These flags are informational today and will be used
//  by the runtime dispatcher once the fiber kernel path is wired in.
// ---------------------------------------------------------------------------
enum class KernelClass : uint8_t {
    SparseCOO  = 0,   // avg fiber length <  DYN_FIBER_LEN_MEDIUM  (element kernel)
    MediumCOO  = 1,   // avg fiber length in [medium, dense)
    DenseFiber = 2    // avg fiber length >= DYN_FIBER_LEN_DENSE   (fiber kernel pays off)
};

#ifndef DYN_FIBER_LEN_MEDIUM
  #define DYN_FIBER_LEN_MEDIUM  8.0      // below this, fiber kernel never helps
#endif
#ifndef DYN_FIBER_LEN_DENSE
  #define DYN_FIBER_LEN_DENSE   32.0     // at or above this, fiber kernel is the clear win
#endif

// ---------------------------------------------------------------------------
//  Tensor storage layout.  Picks one of two schemes at build_flycoo() time
//  based on the hybrid memory heuristic:
//
//    PingPong  -- one primary slab + one scratch slab, swapped each mode.
//                 Constant 2x nnz footprint regardless of mode count.
//                 Each mode-MTTKRP writes a full (val + N idx) remap copy
//                 to scratch before swapping -- extra store bandwidth on
//                 the hot path, but bounded memory use.
//
//    NCopy     -- N pre-sorted physical copies, one per mode's shard order.
//                 Footprint is N/2 x PingPong; the kernel reads directly
//                 from copy[n] for mode n and writes NO remap data, so
//                 the hot-path write bandwidth drops to roughly half.
//                 Only selected when the tensor + factors + OS headroom
//                 comfortably fit in physical RAM.
//
//    InPlace   -- single primary slab, no scratch, no N-copies.  Before
//                 each mode's MTTKRP the slab is re-sorted in place by
//                 ix[n] >> mn_shift[n] (cycle-following counting sort).
//                 The kernel reads from T.vals / T.idx_buf directly
//                 with scr_* = nullptr, so it follows the same no-scatter
//                 hot path the NCopy kernel uses.  Steady-state and peak
//                 footprint are both 1 x (vals + N*idx) * nnz -- the
//                 minimum possible for a COO tensor.  Trade-off: one
//                 extra tensor-slab pass per mode per iteration (the
//                 in-place sort), which raises per-sweep bandwidth by
//                 a few percent vs PingPong at rank >= 64 where the
//                 factor-gather dominates.  Selected only when PingPong
//                 does not fit and 1 x nnz + factors + slack does.
// ---------------------------------------------------------------------------
enum class Layout : uint8_t {
    PingPong  = 0,
    NCopy     = 1,
    InPlace   = 2,
    // -----------------------------------------------------------------
    //  OutOfCore: the tensor is larger than available RAM and therefore
    //  lives on disk (as a .dnb cache).  Only factor matrices, Yhat,
    //  per-thread ofibs, and a single streaming chunk buffer live in
    //  RAM.  Selected automatically when every in-RAM layout would
    //  exceed the memory budget, OR forced by --layout ooc / --ooc on.
    //
    //  Only the Jacobi CP-ALS kernel (all-modes single-pass) is
    //  supported under OutOfCore: it reads the tensor exactly once per
    //  iteration, matching the fundamental streaming pattern we can
    //  sustain from disk.  The per-mode Gauss-Seidel path would need
    //  N disk passes per iter and has no OOC driver -- callers fall
    //  back to Jacobi automatically.
    // -----------------------------------------------------------------
    OutOfCore = 3,
    // -----------------------------------------------------------------
    //  Morton (Stage 2): ALTO-style linearized storage.
    //
    //  Each nonzero is stored as a single 64-bit Morton / Z-curve key
    //  (bits of mode-n indices interleaved via a per-mode pdep/pext
    //  mask) plus a value.  Footprint is 12 B / nnz regardless of N,
    //  vs 4 + 4*N B for the SoA layouts (16 B for 3D, 20 B for 4D,
    //  24 B for 5D).
    //
    //  Benefits:
    //    * Bandwidth reduction: up to ~2x on 5D tensors vs PingPong.
    //    * Implicit all-mode locality: consecutive nnz are close in
    //      every mode simultaneously, so the all-modes MTTKRP kernel
    //      gets better Yhat-row cache reuse than any single-mode sort
    //      can provide.
    //    * One sort builds the layout -- no per-mode re-sort needed.
    //
    //  Constraint: sum_n ceil(log2(I_n)) <= 64.  When violated, the
    //  planner falls back to the next cheapest in-RAM layout.
    //
    //  Only the Jacobi (all-modes single-pass) MTTKRP path has a
    //  Morton driver; Gauss-Seidel runs on the SoA layouts.
    // -----------------------------------------------------------------
    Morton    = 4,
};

// Query physical RAM available to this process, in bytes.  Uses
// GlobalMemoryStatusEx on Windows, MemAvailable from /proc/meminfo on
// Linux, and host_statistics64 on macOS.  Returns 0 if unknown.
size_t dyn_memory_available_bytes();

// Query total physical RAM in bytes.  Same OS dispatch as above.
// Returns 0 if unknown.
size_t dyn_memory_total_bytes();

// Cost model helpers -- used by build_flycoo() to decide Layout and by
// the startup banner to explain the decision.  Sizes are in bytes.
//
//   tensor_pingpong_bytes = 2 * (vals + N*idx) * nnz
//   tensor_ncopy_bytes    = N * (vals + N*idx) * nnz
//   tensor_inplace_bytes  = 1 * (vals + N*idx) * nnz     (steady & peak)
size_t dyn_tensor_pingpong_bytes(uint64_t nnz, int num_modes);
size_t dyn_tensor_ncopy_bytes   (uint64_t nnz, int num_modes);
size_t dyn_tensor_inplace_bytes (uint64_t nnz, int num_modes);

// ---------------------------------------------------------------------------
//  StoragePlan -- unified storage-layout decision.
//
//  A single function (dyn_plan_storage) evaluates every candidate layout
//  (PingPong / NCopy / InPlace / OutOfCore) and picks one based on:
//    * tensor footprint per layout
//    * transient peak during preprocessing (sort scratch for dense fibers)
//    * factor-matrix footprint (2 x sum(I_n) x R_padded x sizeof(value_t))
//    * available RAM + configurable headroom
//    * user override (--layout or DYN_LAYOUT) or OOC override (--ooc / DYN_OOC)
//
//  The same function is called twice:
//    1. Pre-load, in main.cpp, after the .dnb header is peeked.  At this
//       point the caller knows nnz, num_modes, and mode sizes but not yet
//       whether any mode has DenseFiber kernel_class -- pass
//       has_dense_fiber_hint=false (or true to budget conservatively).
//       If the plan returns Layout::OutOfCore, the in-core load path is
//       skipped entirely, avoiding any OOM during load_coo_tensor.
//
//    2. Post-load, inside decide_and_populate_layout, with full accuracy
//       (kernel_class[] filled in by build_flycoo).  This second call
//       picks among {NCopy, PingPong, InPlace} -- the plan will never
//       downgrade from in-core to OOC here because the tensor has already
//       been loaded successfully.
// ---------------------------------------------------------------------------
struct StoragePlan {
    Layout      choice;                 // picked layout
    size_t      avail_bytes;            // RAM available to the process
    size_t      total_bytes;            // total physical RAM
    size_t      budget_bytes;           // (avail * frac - headroom)

    size_t      pingpong_bytes;         // tensor-only
    size_t      ncopy_bytes;            // tensor-only
    size_t      inplace_bytes;          // tensor-only

    size_t      overhead_bytes;         // factors + sort_scratch + slack

    size_t      pingpong_need;          // pingpong_bytes + overhead
    size_t      ncopy_need;             // ncopy_bytes    + overhead
    size_t      inplace_need;           // inplace_bytes  + overhead
    size_t      ooc_need;               // ~factors + ofibs + 1 chunk

    // Short, stable string literal describing why 'choice' was made.
    const char* reason;

    // "" | "cli-layout" | "env-layout" | "cli-ooc" | "env-ooc" | "auto"
    const char* override_source;
};

// Plan the storage layout.  Inputs are either from a .dnb header (pre-load)
// or from a fully populated Tensor (post-load).  mode_sizes may be nullptr
// pre-load; if so, factor-matrix overhead is estimated from rank_padded_est.
// layout_cli   : "", "auto", "pingpong", "ncopy", "inplace", or "ooc"
// ooc_cli      : "", "auto", "on", or "off"
StoragePlan dyn_plan_storage(uint64_t           nnz,
                             int                num_modes,
                             const idx_t*       mode_sizes,
                             int                rank_padded_est,
                             bool               has_dense_fiber_hint,
                             const std::string& layout_cli,
                             const std::string& ooc_cli);

// Print the plan banner.  One-line-per-row, consistent format between
// pre-load and post-load call sites.  prefix defaults to "[plan]".
void dyn_print_plan(const StoragePlan& p, const char* prefix = "[plan]");

// ---------------------------------------------------------------------------
//  Stage 4 -- unified ComputePlan.
//
//  Extends StoragePlan with the remaining runtime decisions that were
//  historically scattered across main.cpp, env vars, and per-mode
//  dispatch tables:
//
//    * CP-ALS kernel algorithm (Jacobi all-modes vs Gauss-Seidel per-mode)
//    * per-mode kernel implementation for Gauss-Seidel
//        (Element / Fiber / FiberCSR)
//    * L2 far-prefetch tier (previously DYN_PF_FAR env toggle)
//    * JIT specialization opt-in (previously --jit / DYN_JIT env toggle)
//
//  One function (dyn_build_compute_plan) encodes the whole decision
//  tree.  It is called from main.cpp AFTER build_flycoo() finishes so
//  all tensor statistics are fully known.  Every dispatch site downstream
//  then reads the plan instead of re-running its own heuristics.
// ---------------------------------------------------------------------------
enum class KernelImpl : uint8_t {
    Element  = 0,   // elementwise (works in every layout)
    Fiber    = 1,   // row-accumulator fiber kernel (needs DenseFiber class)
    FiberCSR = 2,   // CSR-indexed fiber kernel    (NCopy + ncopy_csr[n])
};

// Forward-declared; full definitions below (Tensor) and in dynasor_cpals.h.
enum class CpKernel;
struct Tensor;

struct ComputePlan {
    StoragePlan  storage;                            // layout + memory info

    // CP-ALS kernel algorithm.  For Morton / OOC this is forced to Jacobi
    // (the only supported path).  For in-core SoA layouts it is the
    // lifted pick_auto() decision.
    int          kernel_enum = 0;                    // CpKernel value (0=GS, 1=Jacobi)
    const char*  kernel_reason  = "";
    const char*  kernel_source  = "auto";            // "cli-kernel" | "env-kernel" | "auto"

    // Per-mode kernel implementation when Gauss-Seidel is selected.
    // Ignored under Jacobi.  Filled from T.kernel_class[] + ncopy_csr[].
    KernelImpl   per_mode[DYN_MAX_MODES] = {KernelImpl::Element};

    // L2 far-prefetch tier.  Enabled when the factor-matrix footprint
    // overflows a conservative L3 estimate so demand misses pay back
    // the extra T2 hints; otherwise LFB contention costs more than it
    // saves.
    bool         pf_far  = false;

    // JIT opt-in.  True if --jit / DYN_JIT is set AND the plan has a
    // shape that a JIT specialization will accelerate (rare-grid Rp/N,
    // or Morton where the per-mode masks can be baked in as immediates).
    bool         use_jit = false;

    // Human-readable banner values; populated by dyn_build_compute_plan
    // and printed by dyn_print_compute_plan.
    int          num_modes   = 0;
    int          rank_padded = 0;
    int          num_threads = 0;
    size_t       ofibs_bytes = 0;        // per-thread private output totals
    size_t       factor_bytes = 0;       // sum(I_n) * Rp * 4 * 2 (Y + Yhat)
};

// Build a ComputePlan for a fully-preprocessed Tensor.  `layout_cli`,
// `ooc_cli`, and `kernel_cli` accept the same keywords that main.cpp
// used to dispatch on (empty / "auto" selects the default policy).  The
// returned plan contains the already-printed StoragePlan and the new
// kernel/pf/JIT decisions.
ComputePlan dyn_build_compute_plan(const Tensor&      T,
                                   int                rank_padded,
                                   int                num_threads,
                                   const std::string& layout_cli,
                                   const std::string& ooc_cli,
                                   const std::string& kernel_cli);

// Print the full compute plan -- a superset of dyn_print_plan.  The
// storage rows are identical; extra rows show kernel / per-mode impl /
// pf_far / JIT state.
void dyn_print_compute_plan(const ComputePlan& p, const char* prefix = "[plan]");

struct Tensor {
    int       num_modes = 0;
    uint64_t  nnz       = 0;
    idx_t     mode_size      [DYN_MAX_MODES] = {0};   // |I_n|
    idx_t     shards_per_mode[DYN_MAX_MODES] = {0};   // k_n = ceil(|I_n|/m_n)
    idx_t     mn             [DYN_MAX_MODES] = {0};   // rows per super-shard  (power of 2)
    int       mn_shift       [DYN_MAX_MODES] = {0};   // log2(mn[n])
    idx_t     shard_size = 1;                         // g (paper notation)

    // Per-mode analysis output, filled by build_flycoo.  Lets the runtime
    // dispatcher choose an appropriate kernel variant per mode without
    // another pass over the data.
    double       avg_fiber_len[DYN_MAX_MODES] = {0.0};  // ~ nnz / |I_n|
    KernelClass  kernel_class [DYN_MAX_MODES] = {KernelClass::SparseCOO};

    // Touched-row bitmap per mode.  Bit `r` of touched_bits[n] is set iff
    // at least one nonzero has idx_buf[n] == r.  Populated by build_flycoo
    // as a fused side-effect of the histogram pass.  Used by the runtime
    // driver to skip zeroing Yhat[n] rows that the kernel will never
    // touch (matters most on heavily skewed / sparse modes).
    std::vector<std::vector<uint8_t>> touched_bits;   // size [N][ceil(I_n/8)]
    std::vector<uint64_t>             touched_count;  // popcount per mode

    // ------------------------------------------------------------------------
    //  SoA element buffers and matching scratch.  All point into `buf_raw`.
    // ------------------------------------------------------------------------
    value_t*  vals = nullptr;                          // size nnz
    idx_t*    idx_buf[DYN_MAX_MODES] = {nullptr};      // idx_buf[n] size nnz

    value_t*  scr_vals = nullptr;
    idx_t*    scr_idx[DYN_MAX_MODES] = {nullptr};

    // Primary backing buffer owns [vals][idx[0..N-1]].  Scratch buffer --
    // allocated lazily via ensure_scratch() -- owns [scr_vals][scr_idx[0..N-1]].
    // Splitting the two halves avoids committing the scratch SoA slab during
    // the text parse / binary load, roughly halving peak RSS before the
    // first remap pass.  Pointers (vals/scr_vals/idx_buf/scr_idx) may be
    // SWAPPED after swap_buffers(), but buf_raw/buf_scratch_raw themselves
    // are stable ownership handles used only for free().
    void*     buf_raw         = nullptr;               // primary aligned alloc
    size_t    buf_capacity    = 0;                     // elements currently backed
    size_t    buf_bytes       = 0;                     // primary allocation size
    void*     buf_scratch_raw = nullptr;               // scratch aligned alloc
    size_t    buf_scratch_bytes = 0;                   // scratch allocation size

    // ------------------------------------------------------------------------
    //  FLYCOO shard indices and scheduler output.  These are small (size
    //  ~ sum of k_n) and live outside the hot loop.
    // ------------------------------------------------------------------------
    std::vector<std::vector<uint64_t>> shard_begin;    // [n][sid] -> row
    std::vector<std::vector<uint64_t>> shard_end;      // [n][sid] -> row

    std::vector<std::vector<std::vector<sid_t>>> ss_list;
                                                       // ss_list[n][tid] -> super-shards

    // ------------------------------------------------------------------------
    //  Precomputed remap plan (Section III of the paper).  Built once by
    //  build_flycoo; the runtime driver only does memcpy + kernel pass.
    // ------------------------------------------------------------------------
    size_t                             remap_row_stride = 0;
    std::vector<std::vector<uint64_t>> precomp_thr_off;
    int                                precomp_num_threads = 0;

    // ------------------------------------------------------------------------
    //  Storage layout.  Picked by build_flycoo() from the memory heuristic
    //  unless overridden via DYN_LAYOUT or --layout.  Downstream code
    //  branches on `layout` to skip the remap scatter in NCopy mode.
    // ------------------------------------------------------------------------
    Layout layout = Layout::PingPong;

    // ------------------------------------------------------------------------
    //  OutOfCore (OOC) streaming state.  Populated when layout ==
    //  Layout::OutOfCore and consulted by the streaming MTTKRP driver
    //  (spmttkrp_all_modes_ooc) and the OOC CP-ALS path.
    //
    //  Invariants when ooc_enabled is true:
    //    - layout == Layout::OutOfCore.
    //    - ooc_path  points to an existing .dnb cache on disk whose header
    //      is internally consistent with (num_modes, nnz, mode_size[]).
    //    - vals / idx_buf[*] / buf_raw / ncopy_*  are all nullptr (no
    //      in-RAM slab is kept around -- the point of OOC is that the
    //      tensor slab is too big to hold in RAM).
    //    - ooc_chunk_nnz  is the streaming window size; the driver reads
    //      this many nonzeros' worth of (vals, idx[0..N-1]) at a time.
    //    - ooc_header_bytes  is the file offset (in bytes) of vals[0] in
    //      the .dnb file.  idx[n] starts at
    //          ooc_header_bytes + nnz*sizeof(value_t) + n*nnz*sizeof(idx_t).
    //    - ooc_tnorm_sq  is the precomputed ||T||^2, cached so the CP-ALS
    //      fit metric does not require another streaming pass per iter.
    // ------------------------------------------------------------------------
    bool        ooc_enabled     = false;
    std::string ooc_path;
    uint64_t    ooc_chunk_nnz   = 0;
    uint64_t    ooc_header_bytes = 0;
    double      ooc_tnorm_sq    = 0.0;

    // ------------------------------------------------------------------------
    //  Morton (Layout::Morton) linearized storage.
    //
    //  Each nonzero i is represented as:
    //    morton_keys[i]  -- 64-bit Morton key, a bit-interleaving of the
    //                       N mode indices using per-mode masks.
    //    morton_vals[i]  -- float value.
    //
    //  morton_mask[n] is the mask used by pdep/pext to pack/unpack
    //  mode-n bits into/out of the 64-bit key:
    //    pack:   key |=  _pdep_u64(idx[n],      morton_mask[n])
    //    unpack: idx  = _pext_u64(key,         morton_mask[n])
    //
    //  morton_bits[n] is the number of bits mode n occupies in the
    //  key (= popcount(morton_mask[n]) = ceil(log2(mode_size[n]))).
    //
    //  Both buffers are backed by a single aligned allocation owned by
    //  morton_buf_raw / morton_buf_bytes.  Freed in free_elements().
    //
    //  Invariants when layout == Layout::Morton:
    //    * morton_keys, morton_vals are non-null and sorted by key
    //      (ascending), giving Z-order traversal.
    //    * vals / idx_buf[*] / buf_raw  are nullptr (SoA slab freed).
    //    * sum(morton_bits[0..N-1]) <= 64.
    // ------------------------------------------------------------------------
    uint64_t*   morton_keys       = nullptr;
    value_t*    morton_vals       = nullptr;
    uint64_t    morton_mask[DYN_MAX_MODES] = {0};
    uint8_t     morton_bits[DYN_MAX_MODES] = {0};
    void*       morton_buf_raw    = nullptr;
    size_t      morton_buf_bytes  = 0;

    // --- InPlace layout state: tracks which mode the primary slab is
    //     currently sorted by.  Starts at 0 because build_flycoo leaves the
    //     slab sorted by mode-0 shard after sort_and_count_all.  Updated by
    //     SpmttkrpSweep::process whenever it invokes inplace_sort_slab_by_mode
    //     so the same mode run twice back-to-back skips the redundant sort.
    //     Values outside [0, num_modes) mean "unknown" and force a sort.
    int8_t inplace_sorted_mode = 0;

    // --- InPlace parallel-sort scratch: sizeof(value_t) * nnz aligned
    //     buffer used by inplace_sort_slab_by_mode's parallel scatter +
    //     copy-back path.  Allocated once by decide_and_populate_layout
    //     when Layout::InPlace is selected AND DYN_INPLACE_PARALLEL!=0,
    //     reused across all sorts, freed when the Tensor is destroyed.
    //     Kept separate from buf_scratch_raw because it's only the size
    //     of one SoA column (max(sizeof(value_t), sizeof(idx_t))*nnz),
    //     not the full slab.  Adds ~8 B/nnz to InPlace's steady-state
    //     footprint (vs 20-24 B/nnz for the primary slab itself, and
    //     40-48 B/nnz for PingPong).  If the tensor is so large that
    //     even this small extra fails to allocate, the runtime falls
    //     back to the serial in-place cycle sort automatically.
    void*  inplace_scr_raw   = nullptr;
    size_t inplace_scr_bytes = 0;

    // --- NCopy layout only: ncopy_vals[n] and ncopy_idx[n][m] are the
    // SoA pointers for the mode-n shard-sorted physical copy.
    //
    //   - ncopy_vals[0] == vals   (aliases the primary slab)
    //   - ncopy_idx [0] == idx_buf
    //   - ncopy_vals[1] and ncopy_idx[1] alias the scratch slab when
    //     scratch is kept alive (it holds the mode-1 copy).
    //   - copies 2..N-1 are backed by separately allocated slabs held in
    //     buf_ncopy_raw[i] (i = copy_index - 2).
    //
    // In PingPong mode these arrays are nullptr and must not be used.
    value_t*  ncopy_vals[DYN_MAX_MODES]                      = {nullptr};
    idx_t*    ncopy_idx [DYN_MAX_MODES][DYN_MAX_MODES]       = {{nullptr}};
    void*     buf_ncopy_raw  [DYN_MAX_MODES]                 = {nullptr};
    size_t    buf_ncopy_bytes[DYN_MAX_MODES]                 = {0};

    // Transient scratch slab used during NCopy populate to fiber-sort
    // DenseFiber copies (counting_sort can't run fully in place).  Sized
    // like any copy slab and freed at end of build_flycoo, so it does not
    // contribute to the steady-state NCopy footprint reported in
    // dyn_tensor_ncopy_bytes.
    void*     buf_ncopy_sort_raw   = nullptr;
    size_t    buf_ncopy_sort_bytes = 0;
    value_t*  ncopy_sort_vals      = nullptr;
    idx_t*    ncopy_sort_idx[DYN_MAX_MODES] = {nullptr};

    // Per-copy "is fiber-sorted" flag.  Set to true after a successful
    // counting-sort of copy k by ix[k] within each shard; the runtime
    // dispatch in SpmttkrpSweep::process consults this to decide whether
    // the fiber kernel is safe for mode k under NCopy (if false, fall
    // back to element kernel for that mode only).
    bool      ncopy_fiber_sorted[DYN_MAX_MODES] = {false};

    // --------------------- NCopy + CSR layout (per copy) ------------------
    //
    //  Once a copy has been fiber-sorted (ncopy_fiber_sorted[k] == true),
    //  the ix[k] column is monotone-nondecreasing within every super-
    //  shard, i.e. every fiber is a contiguous run.  We convert that run
    //  structure into a classical CSR row-pointer array:
    //
    //      ncopy_rowptr[k][r]   = first nnz index with ix[k] == r (copy k)
    //      ncopy_rowptr[k][r+1] = one past the last such index
    //
    //  Layout invariants when `ncopy_csr[k]` is true:
    //    * ncopy_rowptr[k] has length (mode_size[k] + 1).
    //    * Entries are 0 at rows that no shard owns (padding at the tail
    //      of the last super-shard) but are always monotone overall.
    //    * ix[k] may still live in ncopy_idx[k][k] (Stage 2A) or be NULL
    //      (Stage 2B, compaction).  The CSR kernel never reads ix[k].
    //
    //  Per-shard row range is derived in O(1) at dispatch time from the
    //  super-shard id:
    //      row_lo = s << mn_shift[k]
    //      row_hi = min((s+1) << mn_shift[k], mode_size[k])
    //  so no per-shard row-range array is needed.
    //
    //  The rowptr slab is owned by this struct and must be freed during
    //  teardown or layout switch (free_ncopy_csr()).  It is small --
    //  (I_k+1) * 8 bytes per CSR-enabled mode -- and therefore does not
    //  materially change the layout-choice heuristic.
    // ----------------------------------------------------------------------
    uint64_t* ncopy_rowptr[DYN_MAX_MODES] = {nullptr};
    size_t    ncopy_rowptr_bytes[DYN_MAX_MODES] = {0};
    bool      ncopy_csr[DYN_MAX_MODES] = {false};

    // --------------- NCopy + CSR compact (Stage 2B) slab handles ----------
    //
    //  When CSR compaction is enabled for copy k, we allocate a new
    //  aligned buffer that is laid out as
    //      [vals | ix[0] | ... | ix[k-1] | ix[k+1] | ... | ix[N-1]]
    //  -- exactly one idx column shorter than the pre-compaction slab.
    //  ncopy_vals[k] and ncopy_idx[k][m != k] are repointed into this
    //  buffer; ncopy_idx[k][k] becomes nullptr (the rowptr replaces it).
    //
    //  For compacted copy 0 we additionally free the primary slab
    //  (buf_raw / vals / idx_buf) since it is no longer referenced;
    //  for compacted copy 1 we free the scratch slab (buf_scratch_raw
    //  / scr_vals / scr_idx) likewise.  The primary/scratch pointers
    //  are nulled so the destructor does not double-free.
    //
    //  ncopy_compact[k] is non-null iff copy k has been compacted.  It
    //  is indexed directly by copy index (unlike buf_ncopy_raw which
    //  uses copy_index-2) since we need slots for copies 0/1 too.
    // ----------------------------------------------------------------------
    void*  buf_ncopy_compact      [DYN_MAX_MODES] = {nullptr};
    size_t buf_ncopy_compact_bytes[DYN_MAX_MODES] = {0};

    // ------------------------------------------------------------------------
    //  Memory-management helpers.
    // ------------------------------------------------------------------------
    // Allocate the SoA backing buffer sized for `nnz` elements across
    // `nmodes` modes.  Does NOT zero-fill.  Previous contents are freed.
    void allocate_elements(uint64_t nnz, int nmodes);
    // Ensure scr_vals / scr_idx[0..num_modes-1] are backed.  Idempotent.
    // Called by the preprocessor before the first remap pass; until then
    // the scratch slab is not committed, which halves peak memory while
    // the text / binary loader is still streaming.
    void ensure_scratch();
    void free_elements();
    // Swap source <-> scratch pointer groups (O(num_modes) assignments).
    void swap_buffers();

    // NCopy-layout helpers.  The extra slabs are materialised lazily by
    // ensure_ncopy_slab(n) and torn down in free_elements().
    //   - ensure_ncopy_slab(n): allocate buf_ncopy_raw[n-2] (for n>=2) and
    //     point ncopy_vals[n] / ncopy_idx[n][*] at it.  For n==0 the slab
    //     aliases the primary; for n==1 it aliases scratch.  Idempotent.
    //   - free_ncopy_extra(): free any buf_ncopy_raw[*] slabs; leaves
    //     ncopy_vals / ncopy_idx pointers nulled so reuse is safe.
    void ensure_ncopy_slab(int copy_index);
    void free_ncopy_extra();
    // Transient fiber-sort scratch (NCopy-only).  Materialized only while
    // populate_ncopy_layout() is running; freed before build_flycoo
    // returns so it does not contribute to steady-state memory use.
    void ensure_ncopy_sort_temp();
    void free_ncopy_sort_temp();

    // Allocate/free the per-mode CSR row-pointer slab.  allocate_ncopy_rowptr
    // is idempotent and a no-op if already sized correctly.  free_ncopy_csr
    // releases every rowptr slab and clears the per-copy csr flag.
    void ensure_ncopy_rowptr(int n);
    void free_ncopy_csr();

    Tensor() = default;
    ~Tensor();
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
};

// ---------- dense factor matrices ----------

struct FactorMatrices {
    int                   num_modes   = 0;
    int                   rank        = 0;            // R
    int                   rank_padded = 0;            // R padded to SIMD+line
    std::vector<value_t*> Y;                          // Y[n] : I_n x rank_padded
    std::vector<value_t*> Yhat;

    void allocate(int nmodes, const idx_t* mode_sizes, int R);
    void free_all();
    void zero_output();
    void init_random(uint64_t seed = 42);
};

// ---------- public entry points ----------

// Read a sparse tensor.
//
// Dispatch policy (fastest first):
//   1. If `<path>.dnb` exists and is newer than `path`, load the binary cache
//      (a single bulk read, ~50-100x faster than parsing text).
//   2. Linux: mmap the text file and parse it in parallel across OpenMP
//      threads (10-30x faster than the serial fgets path on huge FROSTT
//      tensors).
//   3. Portable fallback: serial fgets+strtod parser.
//
// When the text path succeeds and `cache_on_success` is true, a `.dnb` file
// is written alongside the source so subsequent loads take path 1.
bool load_coo_tensor(const std::string& path, Tensor& T,
                     bool cache_on_success = true);

// Write the tensor in the compact binary cache format (.dnb).  Returns false
// on I/O error.  Safe to call from main after load_coo_tensor.
bool save_dnb_tensor(const std::string& path, const Tensor& T);

// Build FLYCOO + schedule + remap plan.  num_threads must match the runtime
// thread count used by spmttkrp_dynasor (the plan is thread-count-specific).
void build_flycoo(Tensor& T,
                  int num_threads,
                  const idx_t* mn_target,            // may be nullptr
                  idx_t shard_size);

// Sort nonzeros within every shard of mode n by the composite key
// (idx_buf[n], idx_buf[w_sort]) -- primary = mode n, secondary = the first
// non-n mode (matches the kernel's w0 pick: w_sort = (n == 0) ? 1 : 0).
//
// Primary key:   groups every fiber into a contiguous run so the fiber
//                kernel can amortize a single Yhat row load/store over the
//                whole fiber.
// Secondary key: makes Y[w_sort] factor-row gathers monotone WITHIN each
//                fiber, converting random L2/L3 misses into essentially
//                sequential L1 hits for typical dense fibers.
//
// Implementation: two back-to-back stable counting sorts per shard (LSD
// over w_sort, then stable MSD over mode n).  Pass 1 uses scr_* as scratch,
// Pass 2 lands the final ordering in T.vals / T.idx_buf[*] directly, so
// this function does NOT swap_buffers() on exit.
//
// The secondary pass is OPT-IN via DYN_SECONDARY_SORT=1 because its
// cost (~2x the legacy single-pass sort) is fixed but its benefit is
// tensor-dependent.  It pays off when Y[w_sort] is large enough to
// spill past L2 AND software prefetch cannot hide the miss latency
// (typical for large-I FROSTT tensors, high rank, power-law
// distributions).  On uniform / well-prefetched workloads it is a net
// loss, so the default remains the legacy single-pass sort (primary
// key only, ends with swap_buffers()).  The single-pass path is also
// taken when |I_{w_sort}| is too large to keep the per-thread cnt[]
// histogram cache-resident.
//
// Per-shard cost:
//   single-pass : O(K_shard + m_n)
//   two-pass    : O(K_shard + m_n + I_{w_sort})
// Parallelised across shards either way.
void fiber_sort_shards(Tensor& T, int n);

// ---------------------------------------------------------------------------
//  InPlace layout helper: sort the primary slab in-place by ix[n].  After
//  this call the SoA (vals + idx_buf) is fully sorted by mode n -- both
//  shard-contiguous AND within-shard row-contiguous (fiber-ready) -- with
//  zero scratch allocation beyond two small histogram arrays.
//
//  Algorithm: cycle-following counting sort ("American flag sort").  Each
//  element moves at most twice (once out of place, once into place).
//
//  Called by SpmttkrpSweep::process() on every mode when T.layout == InPlace
//  to re-sort the slab before that mode's kernel runs.  The kernel then
//  reads directly from T.vals / T.idx_buf with scr_* = nullptr, matching
//  the NCopy hot path.  No swap_buffers() afterwards -- the primary slab
//  stays "live" and is re-sorted at the start of the next mode.
// ---------------------------------------------------------------------------
void inplace_sort_slab_by_mode(Tensor& T, int n, int num_threads);

// Reference (single-thread) spMTTKRP for all modes, used as ground truth.
void spmttkrp_reference(const Tensor& T, FactorMatrices& F);

// Dynasor parallel spMTTKRP for all modes; returns elapsed seconds.
double spmttkrp_dynasor(Tensor& T, FactorMatrices& F, int num_threads);

// ---------------------------------------------------------------------------
//  All-modes single-pass spMTTKRP driver.
//
//  Unlike spmttkrp_dynasor which sweeps the tensor N times (once per target
//  mode), this variant streams the tensor ONCE per sweep and updates every
//  Yhat[n] along the way.  Structural advantages on medium tensors (10M-
//  100M nnz) where factors fit cache but the slab doesn't:
//
//     (a) N-fold tensor-bandwidth reduction: the slab is read once per
//         iteration instead of N times.
//     (b) Factor-row reuse: for each nnz we load factor[w][idx[w][i]]
//         once and reuse it in (N-1) per-mode accumulations, vs once
//         per accumulation in the per-mode driver.
//     (c) Zero sort cost regardless of tensor layout: the kernel accesses
//         all modes symmetrically so any ordering of the slab works.
//         InPlace's per-mode re-sort is skipped.
//
//  Correctness note: all Yhat[n] are computed from the caller-supplied
//  factors Y[0..N-1] -- i.e. a single consistent snapshot.  This is
//  MATHEMATICALLY IDENTICAL to the reference and to the per-mode driver
//  when used for one-shot MTTKRP (factor inputs are not mutated between
//  modes).  In an iterative CP-ALS loop this corresponds to Jacobi-style
//  sub-iteration updates rather than Gauss-Seidel; the two differ only
//  in convergence rate, not in each individual MTTKRP's correctness.
//
//  Memory cost: per-thread private output buffers ("ofibs") sized
//  num_threads * sum_n(mode_size[n] * rank_padded) * sizeof(value_t).
//  For the 10M-30M medium regime with rank <= 128 this is typically a
//  few MiB and fits in L2 per thread easily.  If the allocation fails
//  (very large mode sizes), the driver transparently falls back to
//  spmttkrp_dynasor.
//
//  Returns elapsed seconds.
// ---------------------------------------------------------------------------
double spmttkrp_all_modes_dynasor(Tensor& T, FactorMatrices& F, int num_threads);

// Same Jacobi all-modes semantics as above, but the tensor is streamed from
// T.ooc_path (.dnb).  Requires T.ooc_enabled.  Implementation: dynasor_ooc.cpp.
double spmttkrp_all_modes_ooc(Tensor& T, FactorMatrices& F, int num_threads);

// ---------------------------------------------------------------------------
//  Per-mode sweep API (used by CP-ALS and other iterative decompositions).
//
//  A "sweep" visits modes 0..N-1 in order.  For each mode the driver:
//     1. optionally runs fiber_sort_shards() for DenseFiber modes,
//     2. zeroes F.Yhat[n] (bulk or lazy),
//     3. runs the element/fiber kernel, accumulating into F.Yhat[n],
//     4. emits an sfence so the kernel's NT remap stores become visible,
//     5. swaps source <-> scratch SoA pointers so the next mode reads the
//        freshly remapped layout.
//
//  After process(n) returns, Yhat[n] is valid and the tensor has been
//  remapped for mode (n+1) % N -- including for n = N-1, which leaves the
//  tensor primed for the next sweep starting at mode 0 again.
//
//  Callers insert their own per-mode work (e.g. the CP-ALS factor update)
//  between successive process(n) calls.  Mode order MUST be strictly
//  0, 1, ..., N-1 -- the remap is one-shot and not random-access.
// ---------------------------------------------------------------------------
struct SpmttkrpSweep {
    SpmttkrpSweep(Tensor& T, FactorMatrices& F, int num_threads,
                  bool print_banner = false);
    SpmttkrpSweep(const SpmttkrpSweep&)            = delete;
    SpmttkrpSweep& operator=(const SpmttkrpSweep&) = delete;

    // Process mode `n`.  Must be called in order n = 0, 1, ..., N-1.
    void process(int n);

    // Accumulated kernel time across process() calls so far.
    double elapsed_s() const { return elapsed_; }
    int    fiber_modes_used() const { return n_fiber_used_; }
    int    num_threads()      const { return num_threads_; }

    // Access to shared driver state (rare -- mostly for diagnostics).
    Tensor&         tensor()  { return T_; }
    FactorMatrices& factors() { return F_; }

private:
    Tensor&         T_;
    FactorMatrices& F_;
    int             num_threads_;
    bool            fiber_off_;
    int             n_fiber_used_ = 0;
    double          elapsed_     = 0.0;
    std::vector<uint64_t> work_off_;
};

// Compare factor matrices (Frobenius-relative per mode + max abs/rel).
double compare_factors(const FactorMatrices& A, const FactorMatrices& B);
double compare_factors_detail(const FactorMatrices& A,
                              const FactorMatrices& B,
                              const idx_t* mode_sizes);

// Write one factor matrix (mode n) in human-readable text.  Slow (~MB/s)
// but diff-able for debugging.
bool dump_factor_matrix(const std::string& path,
                        const value_t* data, idx_t rows, int rank,
                        int rank_padded);

// Write one factor matrix in raw little-endian float32 (row-major, `rank`
// columns; internal padding is stripped).  Orders of magnitude faster than
// the text dumper on multi-GiB factor matrices.
bool dump_factor_matrix_bin(const std::string& path,
                            const value_t* data, idx_t rows, int rank,
                            int rank_padded);

const char* simd_backend_name();

} // namespace dynasor

#endif // DYNASOR_COMMON_H
