// ============================================================================
//  dynasor_io.cpp -- tensor and factor-matrix I/O for Dynasor+
//
//  Uses C stdio (fopen/fgets/strtod) rather than C++ iostream.  This is
//  intentional: MinGW-w64 + libstdc++ has a long-standing code-generation
//  bug where std::ifstream combined with OpenMP and -O1+ can crash at
//  stream construction.  C stdio is also faster for the large FROSTT
//  tensors used in the paper.
//
//  The loader writes directly into the Tensor's Structure-of-Arrays
//  backing allocation (Tensor::vals / Tensor::idx_buf[]).  No std::vector
//  is materialized on the hot load path, and there is no default-init
//  zero-fill of the per-element storage.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace dynasor {

static inline int dyn_io_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ============================================================================
//  Memory introspection.  Used by the hybrid layout selector in
//  build_flycoo() to decide whether an N-copy layout fits, and by the
//  startup banner to explain the decision.  All numbers are in bytes.
//
//  Windows : GlobalMemoryStatusEx (ullAvailPhys / ullTotalPhys).
//  Linux   : /proc/meminfo "MemAvailable" / "MemTotal" (kB -> bytes).
//  Fallback: sysconf(_SC_PHYS_PAGES * _SC_PAGESIZE) on POSIX, else 0.
//
//  We use "available" (not "free") so the heuristic correctly avoids
//  NCopy on a box where another process is sitting on most of RAM.
// ============================================================================
size_t dyn_memory_total_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (::GlobalMemoryStatusEx(&ms)) return (size_t)ms.ullTotalPhys;
    return 0;
#elif defined(__linux__)
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    size_t total = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        unsigned long long kb;
        if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
            total = (size_t)kb * 1024ULL;
            break;
        }
    }
    std::fclose(fp);
    if (total) return total;
    long pages = sysconf(_SC_PHYS_PAGES);
    long psz   = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psz > 0) return (size_t)pages * (size_t)psz;
    return 0;
#else
    return 0;
#endif
}

size_t dyn_memory_available_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (::GlobalMemoryStatusEx(&ms)) return (size_t)ms.ullAvailPhys;
    return 0;
#elif defined(__linux__)
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    size_t avail = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        unsigned long long kb;
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            avail = (size_t)kb * 1024ULL;
            break;
        }
    }
    std::fclose(fp);
    return avail;   // 0 if MemAvailable line absent (very old kernels)
#else
    return 0;
#endif
}

// Per-nnz footprint: one value + num_modes indices, each padded to the
// SoA slab's DYN_ALIGN.  We return the *unpadded* arithmetic since the
// padding (~DYN_ALIGN per array) is negligible at the tensor scale where
// the heuristic matters.
static inline size_t dyn_per_nnz_bytes(int num_modes) {
    return sizeof(value_t) + (size_t)num_modes * sizeof(idx_t);
}

size_t dyn_tensor_pingpong_bytes(uint64_t nnz, int num_modes) {
    return 2ULL * dyn_per_nnz_bytes(num_modes) * (size_t)nnz;
}

size_t dyn_tensor_ncopy_bytes(uint64_t nnz, int num_modes) {
    return (size_t)num_modes * dyn_per_nnz_bytes(num_modes) * (size_t)nnz;
}

size_t dyn_tensor_inplace_bytes(uint64_t nnz, int num_modes) {
    // Primary slab (1 x NNZ) plus a single-column scratch of
    // sizeof(value_t) bytes/nnz used by the parallel sort path.  The
    // scratch is optional (DYN_INPLACE_PARALLEL=0 disables it and uses
    // a serial cycle sort with no extra memory), but we include it in
    // the default cost model so the layout heuristic reserves room for
    // it.  If allocation fails at runtime, the sort silently falls back
    // to serial.
    return 1ULL * dyn_per_nnz_bytes(num_modes) * (size_t)nnz
         + sizeof(value_t) * (size_t)nnz;
}

// ---------------------------------------------------------------------------
// FactorMatrices : allocation / initialization helpers.
// ---------------------------------------------------------------------------
void FactorMatrices::allocate(int nmodes, const idx_t* mode_sizes, int R) {
    free_all();
    num_modes   = nmodes;
    rank        = R;

    // Pad each row to a multiple of DYN_SIMD_WIDTH (aligned SIMD) and at
    // least one full cache line (prevents false sharing between threads
    // writing adjacent Yhat rows -- critical on NEON/small-rank AVX2).
    constexpr int kLineFloats = (int)(DYN_ALIGN / sizeof(value_t));  // = 16
    const int lane_pad        = dyn_round_up(R, DYN_SIMD_WIDTH);
    const int line_pad        = dyn_round_up(R, kLineFloats);
    rank_padded = (lane_pad > line_pad) ? lane_pad : line_pad;

    Y   .assign(nmodes, nullptr);
    Yhat.assign(nmodes, nullptr);

    for (int n = 0; n < nmodes; ++n) {
        size_t bytes = (size_t)mode_sizes[n] * rank_padded * sizeof(value_t);
        Y[n]    = (value_t*) dyn_aligned_alloc(bytes);
        Yhat[n] = (value_t*) dyn_aligned_alloc(bytes);
        if (!Y[n] || !Yhat[n]) {
            std::fprintf(stderr, "FactorMatrices::allocate: OOM for mode %d\n", n);
            std::exit(1);
        }
        // Hint the kernel to back these with 2 MiB pages when available
        // (Linux only -- no-op elsewhere).  Y[] is read, Yhat[] is
        // written, by every thread in the kernel, so both benefit from
        // lower TLB pressure on large factor matrices.
        dyn_advise_huge(Y[n],    bytes);
        dyn_advise_huge(Yhat[n], bytes);

        // NUMA policy: Y is read with random index patterns, so ask the
        // kernel to interleave its pages across all online NUMA nodes.
        // Yhat is written (and subsequently read) by a coherent set of
        // threads chosen via the greedy LPT schedule, so we keep the
        // parallel first-touch heuristic for it -- each page ends up on
        // the node of the thread that will own that shard.  On single-
        // socket / non-Linux boxes dyn_interleave_pages is a no-op.
        dyn_interleave_pages(Y[n], bytes);

        // Parallel first-touch + zero: splits the buffer into per-thread
        // chunks so on a 2-socket server each page lands on the NUMA node
        // that will later access it, and the zero step itself happens at
        // the aggregate bandwidth of every memory controller.
        auto zero_par = [](void* p, size_t n) {
        #pragma omp parallel
            {
            #ifdef _OPENMP
                const int  nt  = omp_get_num_threads();
                const int  tid = omp_get_thread_num();
            #else
                const int  nt  = 1;
                const int  tid = 0;
            #endif
                const size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;
                const size_t b     = std::min(n, (size_t)tid * chunk);
                const size_t e     = std::min(n, b + chunk);
                if (e > b) std::memset((char*)p + b, 0, e - b);
            }
        };
        zero_par(Y[n],    bytes);
        zero_par(Yhat[n], bytes);
    }
}

void FactorMatrices::free_all() {
    for (auto* p : Y)    dyn_aligned_free(p);
    for (auto* p : Yhat) dyn_aligned_free(p);
    Y.clear();
    Yhat.clear();
    num_modes = 0;
    rank = rank_padded = 0;
}

void FactorMatrices::zero_output()     { /* handled externally; see main */ }
void FactorMatrices::init_random(uint64_t) { /* done in CLI driver */ }

// ===========================================================================
//  Tensor SoA memory management
// ===========================================================================
Tensor::~Tensor() { free_elements(); }

void Tensor::free_elements() {
    if (buf_raw)         dyn_aligned_free(buf_raw);
    if (buf_scratch_raw) dyn_aligned_free(buf_scratch_raw);
    if (inplace_scr_raw) dyn_aligned_free(inplace_scr_raw);
    if (morton_buf_raw)  dyn_aligned_free(morton_buf_raw);
    buf_raw           = nullptr;
    buf_scratch_raw   = nullptr;
    inplace_scr_raw   = nullptr;
    inplace_scr_bytes = 0;
    buf_capacity      = 0;
    buf_bytes         = 0;
    buf_scratch_bytes = 0;
    morton_buf_raw    = nullptr;
    morton_buf_bytes  = 0;
    morton_keys       = nullptr;
    morton_vals       = nullptr;
    for (int n = 0; n < DYN_MAX_MODES; ++n) {
        morton_mask[n] = 0;
        morton_bits[n] = 0;
    }
    vals         = nullptr;
    scr_vals     = nullptr;
    for (int n = 0; n < DYN_MAX_MODES; ++n) {
        idx_buf[n] = nullptr;
        scr_idx[n] = nullptr;
    }
    free_ncopy_extra();
    free_ncopy_sort_temp();
    free_ncopy_csr();
}

// Release any NCopy-only extra slabs (copies 2..N-1).  Copies 0 and 1
// alias the primary / scratch slabs respectively and are freed by the
// regular free_elements() dyn_aligned_free() calls above.  This helper
// is called both from free_elements() (whole-tensor teardown) and from
// a mid-run layout switch (e.g. if build_flycoo picks PingPong after
// a prior NCopy build).
void Tensor::free_ncopy_extra() {
    for (int i = 0; i < DYN_MAX_MODES; ++i) {
        if (buf_ncopy_raw[i]) {
            dyn_aligned_free(buf_ncopy_raw[i]);
            buf_ncopy_raw  [i] = nullptr;
            buf_ncopy_bytes[i] = 0;
        }
    }
    for (int n = 0; n < DYN_MAX_MODES; ++n) {
        ncopy_vals[n] = nullptr;
        for (int m = 0; m < DYN_MAX_MODES; ++m) ncopy_idx[n][m] = nullptr;
        ncopy_fiber_sorted[n] = false;
    }
    free_ncopy_csr();
}

// Carve the primary aligned buffer into cache-line-aligned slabs:
//     [vals][idx[0]]...[idx[N-1]]
//
// The matching scratch slab [scr_vals][scr_idx[0]]...[scr_idx[N-1]] lives in
// a SEPARATE backing allocation, materialized on first ensure_scratch() call
// (typically from build_flycoo).  Splitting the two halves means the text /
// binary loader never commits the scratch RAM, roughly halving peak RSS on
// large-tensor runs.  For a 300 M-nnz 4-mode tensor this saves ~6 GiB during
// the parse.  Each slab is padded to DYN_ALIGN so consecutive slabs start on
// a fresh cache line.
void Tensor::allocate_elements(uint64_t nnz_, int nmodes_) {
    constexpr size_t kA = DYN_ALIGN;
    const size_t v_bytes = (nnz_ * sizeof(value_t) + kA - 1) & ~(kA - 1);
    const size_t i_bytes = (nnz_ * sizeof(idx_t)   + kA - 1) & ~(kA - 1);
    const size_t primary = v_bytes + (size_t)nmodes_ * i_bytes + kA;

    if (!buf_raw || primary > buf_bytes) {
        if (buf_raw) dyn_aligned_free(buf_raw);
        buf_raw = dyn_aligned_alloc(primary);
        if (!buf_raw) {
            std::fprintf(stderr,
                "Tensor::allocate_elements: OOM (requested %.2f MiB)\n",
                primary / (1024.0 * 1024.0));
            std::exit(1);
        }
        buf_bytes = primary;
        // Ask the kernel to back the huge tensor buffer with 2 MiB pages
        // (Linux-only hint).
        dyn_advise_huge(buf_raw, buf_bytes);
    }

    // If we already held a scratch buffer from a previous smaller tensor
    // drop it: ensure_scratch() will rebuild it at the new size on demand.
    if (buf_scratch_raw) {
        dyn_aligned_free(buf_scratch_raw);
        buf_scratch_raw   = nullptr;
        buf_scratch_bytes = 0;
    }

    nnz          = nnz_;
    num_modes    = nmodes_;
    buf_capacity = nnz_;

    char* p = (char*)buf_raw;
    vals = reinterpret_cast<value_t*>(p);          p += v_bytes;
    for (int n = 0; n < nmodes_; ++n) {
        idx_buf[n] = reinterpret_cast<idx_t*>(p);  p += i_bytes;
    }
    scr_vals = nullptr;
    for (int n = 0; n < DYN_MAX_MODES; ++n) scr_idx[n] = nullptr;
    for (int n = nmodes_; n < DYN_MAX_MODES; ++n) idx_buf[n] = nullptr;
}

void Tensor::ensure_scratch() {
    constexpr size_t kA = DYN_ALIGN;
    const size_t v_bytes = (nnz * sizeof(value_t) + kA - 1) & ~(kA - 1);
    const size_t i_bytes = (nnz * sizeof(idx_t)   + kA - 1) & ~(kA - 1);
    const size_t need    = v_bytes + (size_t)num_modes * i_bytes + kA;

    if (!buf_scratch_raw || need > buf_scratch_bytes) {
        if (buf_scratch_raw) dyn_aligned_free(buf_scratch_raw);
        buf_scratch_raw = dyn_aligned_alloc(need);
        if (!buf_scratch_raw) {
            std::fprintf(stderr,
                "Tensor::ensure_scratch: OOM (requested %.2f MiB)\n",
                need / (1024.0 * 1024.0));
            std::exit(1);
        }
        buf_scratch_bytes = need;
        dyn_advise_huge(buf_scratch_raw, buf_scratch_bytes);
    }

    char* p = (char*)buf_scratch_raw;
    scr_vals = reinterpret_cast<value_t*>(p);          p += v_bytes;
    for (int n = 0; n < num_modes; ++n) {
        scr_idx[n] = reinterpret_cast<idx_t*>(p);      p += i_bytes;
    }
    for (int n = num_modes; n < DYN_MAX_MODES; ++n) scr_idx[n] = nullptr;
}

// ---------------------------------------------------------------------------
//  NCopy slab allocator.  Copies are named 0..N-1 and store mode-n shard-
//  sorted nonzeros for direct kernel read.  The allocation policy reuses
//  existing slabs wherever possible:
//
//    copy 0  -> aliases the primary slab     (buf_raw / vals / idx_buf)
//    copy 1  -> aliases the scratch slab     (buf_scratch_raw / scr_vals / scr_idx)
//    copy n (n>=2) -> private slab in buf_ncopy_raw[n-2], laid out the same
//                     way [vals_slab, idx[0], ..., idx[N-1]].
//
//  Callers invoke ensure_ncopy_slab(n) in order n = 0, 1, ..., N-1 during
//  preprocessing.  For n=0 the function just populates the alias
//  pointers; for n=1 it calls ensure_scratch() first; for n>=2 it
//  allocates a fresh slab.  Idempotent: re-calling with the same n is a
//  no-op once the aliases are in place.
// ---------------------------------------------------------------------------
void Tensor::ensure_ncopy_slab(int copy_index) {
    if (copy_index < 0 || copy_index >= DYN_MAX_MODES) return;
    if (num_modes <= 0 || nnz == 0)                    return;

    constexpr size_t kA = DYN_ALIGN;
    const size_t v_bytes = (nnz * sizeof(value_t) + kA - 1) & ~(kA - 1);
    const size_t i_bytes = (nnz * sizeof(idx_t)   + kA - 1) & ~(kA - 1);
    const size_t need    = v_bytes + (size_t)num_modes * i_bytes + kA;

    if (copy_index == 0) {
        // Alias the primary slab; assumes allocate_elements() already ran.
        ncopy_vals[0] = vals;
        for (int m = 0; m < num_modes; ++m) ncopy_idx[0][m] = idx_buf[m];
        for (int m = num_modes; m < DYN_MAX_MODES; ++m) ncopy_idx[0][m] = nullptr;
        return;
    }
    if (copy_index == 1) {
        // IMPORTANT: do NOT call ensure_scratch() here.  By the time
        // build_flycoo's hybrid-layout selector runs, sort_and_count_all
        // has already executed swap_buffers() at least once, so `vals`
        // may point to the backing slab `buf_scratch_raw` (and vice
        // versa).  Re-carving scr_vals from buf_scratch_raw.start would
        // collapse both vals and scr_vals onto the same slab, corrupting
        // copy 0.
        //
        // Instead we alias ncopy[1] to the CURRENT scr_vals / scr_idx
        // pointers -- whichever slab they resolve to post-swap, that
        // slab is the correct mate for copy 0 (they always point to
        // different backing buffers as long as scratch was allocated).
        if (!scr_vals) {
            std::fprintf(stderr,
                "Tensor::ensure_ncopy_slab(1): scratch not allocated.  "
                "Call ensure_scratch() earlier in the preprocess.\n");
            std::exit(1);
        }
        ncopy_vals[1] = scr_vals;
        for (int m = 0; m < num_modes; ++m) ncopy_idx[1][m] = scr_idx[m];
        for (int m = num_modes; m < DYN_MAX_MODES; ++m) ncopy_idx[1][m] = nullptr;
        return;
    }

    // copy_index >= 2 -- private slab.  buf_ncopy_raw is indexed by
    // (copy_index - 2) so we don't waste two permanently-null slots.
    const int slot = copy_index - 2;
    if (!buf_ncopy_raw[slot] || need > buf_ncopy_bytes[slot]) {
        if (buf_ncopy_raw[slot]) dyn_aligned_free(buf_ncopy_raw[slot]);
        buf_ncopy_raw[slot] = dyn_aligned_alloc(need);
        if (!buf_ncopy_raw[slot]) {
            std::fprintf(stderr,
                "Tensor::ensure_ncopy_slab(%d): OOM (requested %.2f MiB).  "
                "Consider --layout pingpong.\n",
                copy_index, need / (1024.0 * 1024.0));
            std::exit(1);
        }
        buf_ncopy_bytes[slot] = need;
        dyn_advise_huge(buf_ncopy_raw[slot], buf_ncopy_bytes[slot]);
    }

    char* p = (char*)buf_ncopy_raw[slot];
    ncopy_vals[copy_index] = reinterpret_cast<value_t*>(p); p += v_bytes;
    for (int m = 0; m < num_modes; ++m) {
        ncopy_idx[copy_index][m] = reinterpret_cast<idx_t*>(p); p += i_bytes;
    }
    for (int m = num_modes; m < DYN_MAX_MODES; ++m)
        ncopy_idx[copy_index][m] = nullptr;
}

// Transient fiber-sort scratch for the NCopy preprocess.  Same shape as
// any copy slab: [vals_slab, idx[0], ..., idx[N-1]].  Freed explicitly
// by free_ncopy_sort_temp() at end of populate -- it never outlives
// build_flycoo(), so steady-state memory is still N full copies.
void Tensor::ensure_ncopy_sort_temp() {
    if (num_modes <= 0 || nnz == 0) return;
    constexpr size_t kA = DYN_ALIGN;
    const size_t v_bytes = (nnz * sizeof(value_t) + kA - 1) & ~(kA - 1);
    const size_t i_bytes = (nnz * sizeof(idx_t)   + kA - 1) & ~(kA - 1);
    const size_t need    = v_bytes + (size_t)num_modes * i_bytes + kA;

    if (!buf_ncopy_sort_raw || need > buf_ncopy_sort_bytes) {
        if (buf_ncopy_sort_raw) dyn_aligned_free(buf_ncopy_sort_raw);
        buf_ncopy_sort_raw = dyn_aligned_alloc(need);
        if (!buf_ncopy_sort_raw) {
            std::fprintf(stderr,
                "Tensor::ensure_ncopy_sort_temp: OOM (requested %.2f MiB).  "
                "Consider DYN_NCOPY_FIBER_SORT=0 or --layout pingpong.\n",
                need / (1024.0 * 1024.0));
            std::exit(1);
        }
        buf_ncopy_sort_bytes = need;
        dyn_advise_huge(buf_ncopy_sort_raw, buf_ncopy_sort_bytes);
    }

    char* p = (char*)buf_ncopy_sort_raw;
    ncopy_sort_vals = reinterpret_cast<value_t*>(p);       p += v_bytes;
    for (int n = 0; n < num_modes; ++n) {
        ncopy_sort_idx[n] = reinterpret_cast<idx_t*>(p);   p += i_bytes;
    }
    for (int n = num_modes; n < DYN_MAX_MODES; ++n)
        ncopy_sort_idx[n] = nullptr;
}

void Tensor::free_ncopy_sort_temp() {
    if (buf_ncopy_sort_raw) {
        dyn_aligned_free(buf_ncopy_sort_raw);
        buf_ncopy_sort_raw   = nullptr;
        buf_ncopy_sort_bytes = 0;
    }
    ncopy_sort_vals = nullptr;
    for (int n = 0; n < DYN_MAX_MODES; ++n) ncopy_sort_idx[n] = nullptr;
}

// ---------------------------------------------------------------------------
//  NCopy + CSR rowptr helpers.  Each rowptr slab is a single aligned
//  allocation of (mode_size[n] + 1) * 8 bytes.  Small compared to the
//  per-copy SoA slab so the layout heuristic doesn't need updating; we
//  simply allocate on demand during decide_and_populate_layout() for
//  DenseFiber modes when the user opts into DYN_NCOPY_CSR.
// ---------------------------------------------------------------------------
void Tensor::ensure_ncopy_rowptr(int n) {
    if (n < 0 || n >= DYN_MAX_MODES) return;
    if (num_modes <= 0)              return;

    constexpr size_t kA = DYN_ALIGN;
    const size_t rows     = (size_t)mode_size[n];
    const size_t nbytes   = ((rows + 1) * sizeof(uint64_t) + kA - 1) & ~(kA - 1);

    if (!ncopy_rowptr[n] || nbytes > ncopy_rowptr_bytes[n]) {
        if (ncopy_rowptr[n]) dyn_aligned_free(ncopy_rowptr[n]);
        ncopy_rowptr[n] = (uint64_t*) dyn_aligned_alloc(nbytes);
        if (!ncopy_rowptr[n]) {
            std::fprintf(stderr,
                "Tensor::ensure_ncopy_rowptr(%d): OOM (requested %.2f MiB).  "
                "Consider --ncopy-csr off.\n",
                n, nbytes / (1024.0 * 1024.0));
            std::exit(1);
        }
        ncopy_rowptr_bytes[n] = nbytes;
    }
}

void Tensor::free_ncopy_csr() {
    for (int n = 0; n < DYN_MAX_MODES; ++n) {
        if (ncopy_rowptr[n]) {
            dyn_aligned_free(ncopy_rowptr[n]);
            ncopy_rowptr[n]       = nullptr;
            ncopy_rowptr_bytes[n] = 0;
        }
        if (buf_ncopy_compact[n]) {
            dyn_aligned_free(buf_ncopy_compact[n]);
            buf_ncopy_compact[n]       = nullptr;
            buf_ncopy_compact_bytes[n] = 0;
        }
        ncopy_csr[n] = false;
    }
}

void Tensor::swap_buffers() {
    // Swap_buffers re-binds the SoA pointers between primary and scratch.
    // Requires scratch to already be materialized (ensure_scratch() must
    // have been called at least once for the current tensor size).
    std::swap(vals, scr_vals);
    for (int n = 0; n < num_modes; ++n) std::swap(idx_buf[n], scr_idx[n]);
}

// ===========================================================================
//  Sparse COO tensor loader  --  FROSTT / ParTI format
//
//  <i0> <i1> ... <iN-1> <val>\n                (1-based indices)
//
//  Strategy:
//    1. stat() the file to estimate an initial row capacity, so most
//       tensors require zero reallocations during load.
//    2. Parse tokens straight into three raw aligned buffers:
//            tmp_vals[i],  tmp_idx[n][i]
//       kept as plain malloc'd pointers (no std::vector -> no
//       default-init zero-fill, no repeated type-erased allocations).
//    3. On overflow, grow each buffer to 2x capacity via
//       dyn_aligned_alloc + memcpy + free (one doubling at most for a
//       tensor that fits the initial estimate, zero for well-estimated
//       tensors).
//    4. After parsing, hand the populated buffers over to
//       Tensor::allocate_elements + per-array memcpy -- one contiguous
//       aligned backing block, ready for SIMD.
// ===========================================================================

// ------------- tiny growable-aligned-buffer helpers (file-local) ----------
namespace {
static void* grow_raw(void* old, size_t old_bytes, size_t new_bytes) {
    void* n = dyn_aligned_alloc(new_bytes);
    if (!n) return nullptr;
    if (old && old_bytes) std::memcpy(n, old, old_bytes);
    if (old) dyn_aligned_free(old);
    return n;
}
} // namespace

static bool load_tns_serial(const std::string& path, Tensor& T) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot open tensor file '%s' (%s)\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }

    // Start capacity guess from the file size: ~24 bytes/line on average for
    // FROSTT-style tensors.  Clamp to [64 Ki, 64 Mi] rows so we don't over-
    // or under-commit on outliers.
    uint64_t guess = 1ULL << 16;
    {
        struct stat st{};
        if (::stat(path.c_str(), &st) == 0 && st.st_size > 0) {
            guess = (uint64_t)st.st_size / 24ULL;
            if (guess < (1ULL << 16)) guess = 1ULL << 16;
            if (guess > (1ULL << 26)) guess = 1ULL << 26;   // 64 Mi
        }
    }

    // Temporary parallel raw buffers (no std::vector).
    uint64_t cap = guess;
    uint64_t cnt = 0;
    size_t   v_bytes = cap * sizeof(value_t);
    size_t   i_bytes = cap * sizeof(idx_t);

    value_t* tmp_vals = (value_t*) dyn_aligned_alloc(v_bytes);
    idx_t*   tmp_idx[DYN_MAX_MODES] = {nullptr};
    // tmp_idx[n] allocated lazily once we know num_modes

    T.num_modes = 0;
    for (int n = 0; n < DYN_MAX_MODES; ++n) T.mode_size[n] = 0;

    constexpr size_t kBufSize = 8192;
    char   buf[kBufSize];
    double tok[DYN_MAX_MODES + 1];

    int  header_expect_sizes = 0;
    bool header_consumed     = false;

    auto grow_all = [&]() -> bool {
        uint64_t new_cap = cap * 2ULL;
        size_t   nv_bytes = new_cap * sizeof(value_t);
        size_t   ni_bytes = new_cap * sizeof(idx_t);
        value_t* nv = (value_t*) grow_raw(tmp_vals, cnt * sizeof(value_t), nv_bytes);
        if (!nv) return false;
        tmp_vals = nv;
        for (int n = 0; n < T.num_modes; ++n) {
            idx_t* ni = (idx_t*) grow_raw(tmp_idx[n], cnt * sizeof(idx_t), ni_bytes);
            if (!ni) return false;
            tmp_idx[n] = ni;
        }
        cap      = new_cap;
        v_bytes  = nv_bytes;
        i_bytes  = ni_bytes;
        return true;
    };

    auto alloc_mode_arrays = [&](int nm) -> bool {
        T.num_modes = nm;
        for (int n = 0; n < nm; ++n) {
            tmp_idx[n] = (idx_t*) dyn_aligned_alloc(i_bytes);
            if (!tmp_idx[n]) return false;
        }
        return true;
    };

    while (std::fgets(buf, (int)kBufSize, fp)) {
        char* p = buf;
        while (*p && std::isspace((unsigned char)*p)) ++p;
        if (*p == '\0' || *p == '#') continue;

        int ntok = 0;
        while (*p) {
            while (*p && std::isspace((unsigned char)*p)) ++p;
            if (!*p) break;
            if (ntok >= DYN_MAX_MODES + 1) {
                std::fprintf(stderr, "ERROR: line has more than %d tokens\n",
                             DYN_MAX_MODES + 1);
                goto fail;
            }
            char* end = nullptr;
            double v = std::strtod(p, &end);
            if (end == p) {
                std::fprintf(stderr, "ERROR: cannot parse token near: %s\n", p);
                goto fail;
            }
            tok[ntok++] = v;
            p = end;
        }

        // Optional SPLATT-style header detection.
        if (!header_consumed) {
            if (ntok == 1 && T.num_modes == 0 &&
                tok[0] == (double)(int)tok[0] &&
                tok[0] >= 2 && tok[0] <= DYN_MAX_MODES)
            {
                if (!alloc_mode_arrays((int)tok[0])) goto fail;
                header_expect_sizes = T.num_modes;
                continue;
            }
            if (header_expect_sizes > 0 && ntok == header_expect_sizes) {
                for (int n = 0; n < ntok; ++n) T.mode_size[n] = (idx_t)tok[n];
                header_expect_sizes = 0;
                header_consumed     = true;
                continue;
            }
            header_consumed = true;
        }

        if (ntok < 2) continue;
        int nm = ntok - 1;
        if (T.num_modes == 0) {
            if (nm > DYN_MAX_MODES) {
                std::fprintf(stderr, "ERROR: tensor has %d modes (max %d)\n",
                             nm, DYN_MAX_MODES);
                goto fail;
            }
            if (!alloc_mode_arrays(nm)) goto fail;
        } else if (nm != T.num_modes) {
            std::fprintf(stderr,
                "ERROR: inconsistent mode count (line expects %d, got %d)\n",
                T.num_modes, nm);
            goto fail;
        }

        if (cnt == cap && !grow_all()) {
            std::fprintf(stderr, "ERROR: OOM growing tensor buffers\n");
            goto fail;
        }

        for (int n = 0; n < nm; ++n) {
            long long idx_1b = (long long)tok[n];
            if (idx_1b <= 0) {
                std::fprintf(stderr, "ERROR: non-positive index %lld\n", idx_1b);
                goto fail;
            }
            idx_t idx = (idx_t)(idx_1b - 1);
            tmp_idx[n][cnt] = idx;
            if (idx + 1 > T.mode_size[n]) T.mode_size[n] = idx + 1;
        }
        tmp_vals[cnt] = (value_t)tok[nm];
        ++cnt;
    }

    std::fclose(fp);

    // Transfer into the Tensor's SoA backing buffer (single aligned alloc).
    // Free each tmp array immediately after its memcpy; that way we never
    // hold both the full tmp set AND the full primary alloc at the same
    // time -- transient peak becomes final_size + one_tmp_slab instead of
    // final_size + full_tmp_set (saves ~N * nnz * sizeof(idx_t) bytes).
    T.allocate_elements(cnt, T.num_modes);
    std::memcpy(T.vals, tmp_vals, cnt * sizeof(value_t));
    dyn_aligned_free(tmp_vals);
    tmp_vals = nullptr;
    for (int n = 0; n < T.num_modes; ++n) {
        std::memcpy(T.idx_buf[n], tmp_idx[n], cnt * sizeof(idx_t));
        dyn_aligned_free(tmp_idx[n]);
        tmp_idx[n] = nullptr;
    }
    return true;

fail:
    std::fclose(fp);
    dyn_aligned_free(tmp_vals);
    for (int n = 0; n < DYN_MAX_MODES; ++n) dyn_aligned_free(tmp_idx[n]);
    return false;
}

// ===========================================================================
//  Binary tensor cache -- .dnb format
//
//  Layout (little-endian, packed -- no implicit struct padding used):
//     char     magic[4]      = "DNB1"
//     uint32_t version       = 1
//     uint32_t value_bytes   = sizeof(value_t)
//     uint32_t index_bytes   = sizeof(idx_t)
//     uint32_t num_modes
//     uint64_t nnz
//     uint32_t mode_size[DYN_MAX_MODES]   // zero-padded tail
//     uint32_t reserved[7]                // future flags (endianness etc.)
//     value_t  vals[nnz]
//     idx_t    idx_buf[0][nnz]
//     idx_t    idx_buf[1][nnz]
//     ...
//     idx_t    idx_buf[num_modes-1][nnz]
//
//  A single text parse populates the file alongside the source tensor on
//  first load; every subsequent load skips parsing entirely and does one
//  bulk fread per array (~saturated storage bandwidth).  Validity is keyed
//  on mtime (file is regenerated if the source .tns is newer).
// ===========================================================================
namespace {

constexpr char     kDnbMagic[4]   = {'D','N','B','1'};
constexpr uint32_t kDnbVersion    = 1;
constexpr size_t   kDnbHeaderSize = 4 + 4 + 4 + 4 + 4 + 8 + 4 * DYN_MAX_MODES + 4 * 7;

static bool dnb_cache_is_fresh(const std::string& tns, const std::string& dnb) {
    // On MinGW32 / MSYS2, the default `struct stat` has a 32-bit `off_t`
    // and a 32-bit `time_t`.  For >= 2 GiB cache files (our 335 M-nnz 3-mode
    // cache is 5.4 GiB) the default `st_size` wraps to a small / negative
    // value, so the header-size sanity check below rejects every large
    // cache.  Use the explicit 64-bit stat to avoid silent cache misses.
    //
    // If the source .tns has been deleted (common after generating large
    // benchmark tensors and keeping only the compact binary cache), we fall
    // back to "cache is fresh if it exists and has a plausible header",
    // because there's nothing to compare against.  Without this fallback a
    // standalone .dnb would be unusable.
#ifdef _WIN32
    struct __stat64 b{};
    if (::_stat64(dnb.c_str(), &b) != 0) return false;
    if ((uint64_t)b.st_size < (uint64_t)kDnbHeaderSize) return false;
    struct __stat64 a{};
    if (::_stat64(tns.c_str(), &a) != 0) return true; // no source -> trust cache
    return b.st_mtime >= a.st_mtime;
#else
    struct stat b{};
    if (::stat(dnb.c_str(), &b) != 0) return false;
    if ((uint64_t)b.st_size < (uint64_t)kDnbHeaderSize) return false;
    struct stat a{};
    if (::stat(tns.c_str(), &a) != 0) return true; // no source -> trust cache
    return b.st_mtime >= a.st_mtime;
#endif
}

static bool load_dnb_tensor(const std::string& path, Tensor& T) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    char     magic[4];
    uint32_t version, value_bytes, index_bytes, num_modes_u;
    uint64_t nnz_u;
    uint32_t mode_sizes[DYN_MAX_MODES];
    uint32_t reserved[7];

    bool ok = std::fread(magic,        sizeof(magic),        1, fp) == 1 &&
              std::fread(&version,     sizeof(version),      1, fp) == 1 &&
              std::fread(&value_bytes, sizeof(value_bytes),  1, fp) == 1 &&
              std::fread(&index_bytes, sizeof(index_bytes),  1, fp) == 1 &&
              std::fread(&num_modes_u, sizeof(num_modes_u),  1, fp) == 1 &&
              std::fread(&nnz_u,       sizeof(nnz_u),        1, fp) == 1 &&
              std::fread(mode_sizes,   sizeof(mode_sizes),   1, fp) == 1 &&
              std::fread(reserved,     sizeof(reserved),     1, fp) == 1;
    if (!ok || std::memcmp(magic, kDnbMagic, 4) != 0 ||
        version != kDnbVersion ||
        value_bytes != sizeof(value_t) ||
        index_bytes != sizeof(idx_t) ||
        num_modes_u == 0 || num_modes_u > (uint32_t)DYN_MAX_MODES ||
        nnz_u == 0)
    {
        std::fclose(fp);
        return false;
    }

    T.num_modes = (int)num_modes_u;
    for (int n = 0; n < T.num_modes; ++n) T.mode_size[n] = (idx_t)mode_sizes[n];
    T.allocate_elements(nnz_u, T.num_modes);

    if (std::fread(T.vals, sizeof(value_t), nnz_u, fp) != (size_t)nnz_u) {
        std::fclose(fp);
        return false;
    }
    for (int n = 0; n < T.num_modes; ++n) {
        if (std::fread(T.idx_buf[n], sizeof(idx_t), nnz_u, fp) != (size_t)nnz_u) {
            std::fclose(fp);
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

} // namespace

bool save_dnb_tensor(const std::string& path, const Tensor& T) {
    if (T.nnz == 0 || T.num_modes <= 0 || T.num_modes > DYN_MAX_MODES) return false;

    // Write to a temp path and rename, so a crash mid-write never leaves a
    // truncated cache on disk.
    std::string tmp = path + ".tmp";
    std::FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (!fp) return false;

    uint32_t version     = kDnbVersion;
    uint32_t value_bytes = (uint32_t)sizeof(value_t);
    uint32_t index_bytes = (uint32_t)sizeof(idx_t);
    uint32_t num_modes_u = (uint32_t)T.num_modes;
    uint64_t nnz_u       = (uint64_t)T.nnz;
    uint32_t mode_sizes[DYN_MAX_MODES] = {0};
    for (int n = 0; n < T.num_modes; ++n) mode_sizes[n] = (uint32_t)T.mode_size[n];
    uint32_t reserved[7] = {0};

    bool ok = std::fwrite(kDnbMagic,    sizeof(kDnbMagic),   1, fp) == 1 &&
              std::fwrite(&version,     sizeof(version),     1, fp) == 1 &&
              std::fwrite(&value_bytes, sizeof(value_bytes), 1, fp) == 1 &&
              std::fwrite(&index_bytes, sizeof(index_bytes), 1, fp) == 1 &&
              std::fwrite(&num_modes_u, sizeof(num_modes_u), 1, fp) == 1 &&
              std::fwrite(&nnz_u,       sizeof(nnz_u),       1, fp) == 1 &&
              std::fwrite(mode_sizes,   sizeof(mode_sizes),  1, fp) == 1 &&
              std::fwrite(reserved,     sizeof(reserved),    1, fp) == 1 &&
              std::fwrite(T.vals, sizeof(value_t), nnz_u, fp) == (size_t)nnz_u;
    for (int n = 0; ok && n < T.num_modes; ++n) {
        ok = std::fwrite(T.idx_buf[n], sizeof(idx_t), nnz_u, fp) == (size_t)nnz_u;
    }
    std::fclose(fp);

    if (!ok) { std::remove(tmp.c_str()); return false; }
    std::remove(path.c_str());  // best effort; ignore error
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

// ===========================================================================
//  Linux-only: memory-mapped, OpenMP-parallel FROSTT text parser.
//
//  On 1.7 B-nonzero tensors (Amazon) the serial strtod loop runs at
//  ~30 MB/s which translates to ~15 minutes on a 30 GiB text file.
//  The parallel mmap parser hits the per-thread strtod ceiling (~40-60
//  MB/s each) so a 32-thread Zen 5 / V2 server finishes the same file
//  in under a minute.
//
//  Strategy:
//    Pass 1 (parallel): each thread counts newlines in an equal byte
//                       slice, after aligning slice boundaries to the
//                       next '\n' so no line spans two workers.
//    Prefix-scan:       per-thread line counts -> per-thread output
//                       offsets into the final SoA arrays.
//    Pass 2 (parallel): each thread parses its slice with a hand-rolled
//                       integer scanner for indices plus strtod for the
//                       floating-point value, writing directly to the
//                       final aligned buffers.
// ===========================================================================
#ifdef __linux__
namespace {

static inline bool is_digit_(char c) { return c >= '0' && c <= '9'; }

// Parse a 1-based integer; advance *pp past the digits.  Returns false if
// no digit was seen (caller decides whether that is a failure).
static inline bool parse_uint(const char*& p, const char* ep, uint64_t& out) {
    uint64_t v = 0;
    bool any = false;
    while (p < ep && (*p == ' ' || *p == '\t')) ++p;
    while (p < ep && is_digit_(*p)) {
        v = v * 10 + (uint64_t)(*p - '0');
        ++p;
        any = true;
    }
    if (!any) return false;
    out = v;
    return true;
}

// strtod needs a NUL-terminated string.  We can't modify the mmapped
// buffer, so copy up to 63 characters of the current token into a stack
// buffer (values never exceed ~20 chars in FROSTT tensors) and parse.
static inline double parse_double(const char*& p, const char* ep) {
    while (p < ep && (*p == ' ' || *p == '\t')) ++p;
    char local[64];
    size_t ll = 0;
    while (p < ep && *p != ' ' && *p != '\t' &&
           *p != '\n' && *p != '\r' && ll < 63)
    {
        local[ll++] = *p++;
    }
    local[ll] = 0;
    char* endp;
    return std::strtod(local, &endp);
}

// Find the start of the next line at or after `pos`.
static inline size_t align_to_next_line(const char* buf, size_t pos, size_t bytes) {
    while (pos < bytes && buf[pos] != '\n') ++pos;
    if (pos < bytes) ++pos;
    return pos;
}

static bool load_tns_mmap(const std::string& path, Tensor& T) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) < 0 || st.st_size <= 0) { ::close(fd); return false; }

    const size_t bytes = (size_t)st.st_size;
    void* map = ::mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) return false;
    const char* buf = (const char*)map;
    ::madvise((void*)buf, bytes, MADV_SEQUENTIAL);
    ::madvise((void*)buf, bytes, MADV_WILLNEED);

    auto fail = [&](bool ret = false) { ::munmap((void*)buf, bytes); return ret; };

    // ---- Skip leading blank / comment lines; optionally detect a SPLATT
    //      style header ("N\n" followed by "I0 I1 ... I_{N-1}\n"). ----
    size_t data_start = 0;
    int   splatt_nm   = 0;
    idx_t splatt_sz[DYN_MAX_MODES] = {0};

    auto skip_blank_comment = [&](size_t& p) -> bool {
        while (p < bytes) {
            char c = buf[p];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++p;
            else if (c == '#') {
                while (p < bytes && buf[p] != '\n') ++p;
            } else return true;
        }
        return false;
    };

    {
        size_t p = 0;
        if (!skip_blank_comment(p)) return fail();
        size_t line_start = p;
        size_t line_end   = p;
        while (line_end < bytes && buf[line_end] != '\n') ++line_end;

        // Parse tokens on that line into a local buffer for peek.
        char local[256];
        size_t ll = std::min((size_t)255, line_end - line_start);
        std::memcpy(local, buf + line_start, ll);
        local[ll] = 0;

        int ntok = 0;
        double toks[DYN_MAX_MODES + 1];
        char* lp = local; char* endp;
        while (*lp && ntok <= DYN_MAX_MODES) {
            while (*lp && std::isspace((unsigned char)*lp)) ++lp;
            if (!*lp) break;
            double v = std::strtod(lp, &endp);
            if (endp == lp) break;
            toks[ntok++] = v;
            lp = endp;
        }

        bool is_splatt_header =
            ntok == 1 && toks[0] == (double)(int)toks[0] &&
            toks[0] >= 2 && toks[0] <= DYN_MAX_MODES;

        if (is_splatt_header) {
            splatt_nm = (int)toks[0];
            size_t q = line_end < bytes ? line_end + 1 : line_end;
            if (!skip_blank_comment(q)) return fail();
            size_t s = q;
            size_t e = q;
            while (e < bytes && buf[e] != '\n') ++e;
            char local2[256];
            size_t ll2 = std::min((size_t)255, e - s);
            std::memcpy(local2, buf + s, ll2);
            local2[ll2] = 0;
            int parsed = 0;
            char* lp2 = local2;
            while (*lp2 && parsed < splatt_nm) {
                while (*lp2 && std::isspace((unsigned char)*lp2)) ++lp2;
                if (!*lp2) break;
                double v = std::strtod(lp2, &endp);
                if (endp == lp2) break;
                splatt_sz[parsed++] = (idx_t)v;
                lp2 = endp;
            }
            if (parsed != splatt_nm) return fail();
            data_start = e < bytes ? e + 1 : e;
        } else {
            data_start = line_start;
        }
    }

    if (data_start >= bytes) return fail();

    // ---- Pass 1 (parallel): count lines per chunk. ----
    const int nt = dyn_io_threads();
    std::vector<size_t>   chunk_start(nt + 1, 0);
    std::vector<uint64_t> line_count (nt, 0);

    chunk_start[0]  = data_start;
    chunk_start[nt] = bytes;
    for (int t = 1; t < nt; ++t) {
        size_t pos = data_start + (bytes - data_start) * (size_t)t / (size_t)nt;
        chunk_start[t] = align_to_next_line(buf, pos, bytes);
    }

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; ++t) {
        const size_t lo = chunk_start[t];
        const size_t hi = chunk_start[t + 1];
        uint64_t lc = 0;
        // Newline-count a chunk.  Each non-empty, non-comment line yields
        // a nonzero; FROSTT files never mix comments with data mid-file,
        // so a simple newline count is accurate.
        for (size_t i = lo; i < hi; ++i) if (buf[i] == '\n') ++lc;
        // Trailing line without a '\n' belongs only to the last chunk.
        if (t == nt - 1 && hi > lo && buf[hi - 1] != '\n') ++lc;
        line_count[t] = lc;
    }

    std::vector<uint64_t> tof(nt + 1, 0);
    for (int t = 0; t < nt; ++t) tof[t + 1] = tof[t] + line_count[t];
    const uint64_t total = tof[nt];
    if (total == 0) return fail();

    // ---- Detect num_modes from first data line (if no SPLATT header). ----
    int num_modes = splatt_nm;
    if (num_modes == 0) {
        size_t p = data_start;
        size_t e = p;
        while (e < bytes && buf[e] != '\n') ++e;
        char local[256];
        size_t ll = std::min((size_t)255, e - p);
        std::memcpy(local, buf + p, ll);
        local[ll] = 0;
        int nm = 0;
        char* lp = local; char* endp;
        while (*lp && nm <= DYN_MAX_MODES + 1) {
            while (*lp && std::isspace((unsigned char)*lp)) ++lp;
            if (!*lp) break;
            std::strtod(lp, &endp);
            if (endp == lp) break;
            ++nm;
            lp = endp;
        }
        if (nm < 2 || nm > DYN_MAX_MODES + 1) return fail();
        num_modes = nm - 1;
    }

    // ---- Allocate final SoA buffers. ----
    T.num_modes = num_modes;
    for (int n = 0; n < num_modes; ++n) T.mode_size[n] = splatt_sz[n];
    T.allocate_elements(total, num_modes);

    // ---- Pass 2 (parallel): parse each chunk straight into T. ----
    std::vector<uint8_t> chunk_ok(nt, 1);
    std::vector<idx_t>   local_max(nt * DYN_MAX_MODES, 0);

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; ++t) {
        const char* p  = buf + chunk_start[t];
        const char* ep = buf + chunk_start[t + 1];
        uint64_t i = tof[t];
        idx_t mx[DYN_MAX_MODES] = {0};
        bool ok = true;

        while (p < ep) {
            while (p < ep && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
            if (p >= ep) break;
            if (*p == '\n') { ++p; continue; }

            uint64_t idxs[DYN_MAX_MODES];
            bool bad = false;
            for (int n = 0; n < num_modes; ++n) {
                if (!parse_uint(p, ep, idxs[n])) { bad = true; break; }
            }
            if (bad) { ok = false; break; }

            double v = parse_double(p, ep);
            T.vals[i] = (value_t)v;

            for (int n = 0; n < num_modes; ++n) {
                idx_t ix = (idx_t)(idxs[n] - 1);
                T.idx_buf[n][i] = ix;
                if (ix + 1 > mx[n]) mx[n] = ix + 1;
            }
            ++i;

            while (p < ep && *p != '\n') ++p;
            if (p < ep) ++p;
        }

        // Record that we wrote at least the lines we said we would.  If we
        // wrote fewer (malformed line, etc.), flag the fail.
        if (i != tof[t + 1]) ok = false;

        for (int n = 0; n < DYN_MAX_MODES; ++n)
            local_max[(size_t)t * DYN_MAX_MODES + n] = mx[n];
        chunk_ok[t] = ok ? 1 : 0;
    }

    for (int t = 0; t < nt; ++t) if (!chunk_ok[t]) return fail();

    // Reduce per-thread max indices into mode_size when no SPLATT header
    // provided the true dimensions.
    if (splatt_nm == 0) {
        for (int n = 0; n < num_modes; ++n) {
            idx_t mx = 0;
            for (int t = 0; t < nt; ++t) {
                idx_t v = local_max[(size_t)t * DYN_MAX_MODES + n];
                if (v > mx) mx = v;
            }
            T.mode_size[n] = mx;
        }
    }

    ::munmap((void*)buf, bytes);
    return true;
}

} // namespace
#endif // __linux__

// ===========================================================================
//  Public entry point: dispatches across all available loaders.
// ===========================================================================
bool load_coo_tensor(const std::string& path, Tensor& T,
                     bool cache_on_success)
{
    const std::string dnb_path = path + ".dnb";

    // 1. Fastest path -- reusable binary cache.
    //    The caller can disable both the read and the write by passing
    //    cache_on_success = false (e.g. --no-cache on the CLI), which is
    //    how we benchmark the text-parse path on demand.
    if (cache_on_success && dnb_cache_is_fresh(path, dnb_path)) {
        double t0 = 0.0, t1 = 0.0;
#ifdef _OPENMP
        t0 = omp_get_wtime();
#endif
        if (load_dnb_tensor(dnb_path, T)) {
#ifdef _OPENMP
            t1 = omp_get_wtime();
#endif
            std::printf("[io] loaded %llu nonzeros, %d modes, shape =",
                        (unsigned long long)T.nnz, T.num_modes);
            for (int n = 0; n < T.num_modes; ++n)
                std::printf(" %u", (unsigned)T.mode_size[n]);
            std::printf("  (from %s, %.3f s)\n", dnb_path.c_str(),
                        (t1 - t0));
            return true;
        }
        std::fprintf(stderr,
            "[io] WARN: stale or invalid binary cache '%s', reparsing text\n",
            dnb_path.c_str());
    }

    // 2. Text parse.  Prefer the parallel mmap path on Linux; fall back to
    //    the portable serial loader on any failure (and on non-Linux).
    double t0 = 0.0, t1 = 0.0;
#ifdef _OPENMP
    t0 = omp_get_wtime();
#endif
    bool ok = false;
    const char* parser = "serial";

#ifdef __linux__
    ok = load_tns_mmap(path, T);
    if (ok) parser = "mmap";
    else if (T.nnz != 0) {
        // mmap parser allocated T but aborted: drop it and let serial retry.
        T.free_elements();
        T.num_modes = 0;
        for (int n = 0; n < DYN_MAX_MODES; ++n) T.mode_size[n] = 0;
    }
#endif
    if (!ok) ok = load_tns_serial(path, T);
    if (!ok) return false;

#ifdef _OPENMP
    t1 = omp_get_wtime();
#endif

    std::printf("[io] loaded %llu nonzeros, %d modes, shape =",
                (unsigned long long)T.nnz, T.num_modes);
    for (int n = 0; n < T.num_modes; ++n)
        std::printf(" %u", (unsigned)T.mode_size[n]);
    std::printf("  (%s text, %.3f s)\n", parser, (t1 - t0));

    // 3. Write the binary cache as a side effect so the next run is fast.
    if (cache_on_success) {
#ifdef _OPENMP
        t0 = omp_get_wtime();
#endif
        if (save_dnb_tensor(dnb_path, T)) {
#ifdef _OPENMP
            t1 = omp_get_wtime();
#endif
            std::printf("[io] wrote binary cache '%s' (%.3f s)\n",
                        dnb_path.c_str(), (t1 - t0));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Dump one factor matrix (mode n) to a text file.
// ---------------------------------------------------------------------------
bool dump_factor_matrix(const std::string& path,
                        const value_t* data, idx_t rows, int rank,
                        int rank_padded)
{
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot write '%s'\n", path.c_str());
        return false;
    }
    for (idx_t i = 0; i < rows; ++i) {
        const value_t* row = data + (size_t)i * rank_padded;
        for (int r = 0; r < rank; ++r) {
            std::fprintf(fp, "%.7g", (double)row[r]);
            if (r + 1 < rank) std::fputc(' ', fp);
        }
        std::fputc('\n', fp);
    }
    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// Dump one factor matrix as raw little-endian float32 (row-major, `rank`
// columns; SIMD padding stripped).  File layout:
//
//   uint32_t  magic         = 'DYFM'   (Dynasor Factor Matrix)
//   uint32_t  version       = 1
//   uint32_t  value_bytes   = sizeof(value_t)
//   uint32_t  rank
//   uint64_t  rows
//   value_t   data[rows * rank]
//
// Roughly 10x smaller than the text dumper, and written in one contiguous
// fwrite per row-group (or per-chunk with OpenMP parallel packing), so a
// multi-GiB factor matrix lands on disk at near-PCIe speed.
// ---------------------------------------------------------------------------
bool dump_factor_matrix_bin(const std::string& path,
                            const value_t* data, idx_t rows, int rank,
                            int rank_padded)
{
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "ERROR: cannot write '%s'\n", path.c_str());
        return false;
    }

    const char     magic[4]      = {'D','Y','F','M'};
    const uint32_t version       = 1;
    const uint32_t value_bytes   = (uint32_t)sizeof(value_t);
    const uint32_t rank_u        = (uint32_t)rank;
    const uint64_t rows_u        = (uint64_t)rows;

    bool ok = std::fwrite(magic,        sizeof(magic),       1, fp) == 1 &&
              std::fwrite(&version,     sizeof(version),     1, fp) == 1 &&
              std::fwrite(&value_bytes, sizeof(value_bytes), 1, fp) == 1 &&
              std::fwrite(&rank_u,      sizeof(rank_u),      1, fp) == 1 &&
              std::fwrite(&rows_u,      sizeof(rows_u),      1, fp) == 1;

    if (ok && rank == rank_padded) {
        // Trivial case: no internal padding to strip, bulk-write.
        ok = std::fwrite(data, sizeof(value_t), (size_t)rows * rank, fp)
             == (size_t)rows * rank;
    } else if (ok) {
        // Pack into a row-group buffer so we still issue big fwrites.
        constexpr size_t kGroupRows = 4096;
        std::vector<value_t> pack((size_t)kGroupRows * rank);
        for (idx_t base = 0; base < rows && ok; base += (idx_t)kGroupRows) {
            const idx_t this_rows = std::min((idx_t)kGroupRows, rows - base);
            for (idx_t i = 0; i < this_rows; ++i) {
                std::memcpy(pack.data() + (size_t)i * rank,
                            data + (size_t)(base + i) * rank_padded,
                            (size_t)rank * sizeof(value_t));
            }
            ok = std::fwrite(pack.data(), sizeof(value_t),
                             (size_t)this_rows * rank, fp)
                 == (size_t)this_rows * rank;
        }
    }

    std::fclose(fp);
    if (!ok) std::remove(path.c_str());
    return ok;
}

} // namespace dynasor
