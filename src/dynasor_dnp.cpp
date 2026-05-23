// ============================================================================
//  dynasor_dnp.cpp -- Dynasor+ Processed-tensor binary I/O.
//
//  Writes / reads the .dnp file format described in include/dynasor_dnp.h.
//  One format covers all four in-core layouts (Morton, NCopy, PingPong,
//  InPlace) via a layout_kind byte in the common header; the loader routes
//  to the appropriate section parser.
//
//  The header records the thread count at save-time; a load with a
//  different --threads is rejected because the schedule (ss_list,
//  precomp_thr_off) is thread-specific.
// ============================================================================
#include "dynasor_dnp.h"
#include "dynasor_common.h"
#include "dynasor_simd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dynasor {

namespace {

constexpr char     kDnpMagic[4]  = {'D','N','P','1'};
constexpr uint32_t kDnpVersion   = 1;

// layout_kind encoding.  Kept stable across versions.
constexpr uint32_t kLayoutMorton   = 0;
constexpr uint32_t kLayoutNCopy    = 1;
constexpr uint32_t kLayoutPingPong = 2;
constexpr uint32_t kLayoutInPlace  = 3;

static uint32_t layout_to_kind(Layout L) {
    switch (L) {
        case Layout::Morton:   return kLayoutMorton;
        case Layout::NCopy:    return kLayoutNCopy;
        case Layout::PingPong: return kLayoutPingPong;
        case Layout::InPlace:  return kLayoutInPlace;
        default:               return (uint32_t)-1;
    }
}
static Layout kind_to_layout(uint32_t k) {
    switch (k) {
        case kLayoutMorton:   return Layout::Morton;
        case kLayoutNCopy:    return Layout::NCopy;
        case kLayoutPingPong: return Layout::PingPong;
        case kLayoutInPlace:  return Layout::InPlace;
        default:              return Layout::PingPong;
    }
}
static const char* kind_name(uint32_t k) {
    switch (k) {
        case kLayoutMorton:   return "Morton";
        case kLayoutNCopy:    return "NCopy";
        case kLayoutPingPong: return "PingPong";
        case kLayoutInPlace:  return "InPlace";
        default:              return "??";
    }
}

// --------------------------------------------------------------------
//  Thin I/O wrappers.  We do not chase every possible short read/write;
//  fread/fwrite errors are sticky (fp->error flag), and we check the
//  `ok` accumulator at each section boundary to short-circuit.
// --------------------------------------------------------------------
struct Out {
    std::FILE* fp = nullptr;
    bool       ok = true;

    template <typename T>
    void w(const T& v)           { if (ok) ok = std::fwrite(&v, sizeof(T), 1, fp) == 1; }
    void wb(const void* p, size_t n) {
        if (ok && n) ok = std::fwrite(p, 1, n, fp) == n;
    }
};
struct In {
    std::FILE* fp = nullptr;
    bool       ok = true;

    template <typename T>
    void r(T& v)                  { if (ok) ok = std::fread(&v, sizeof(T), 1, fp) == 1; }
    void rb(void* p, size_t n) {
        if (ok && n) ok = std::fread(p, 1, n, fp) == n;
    }
};

// --------------------------------------------------------------------
//  Common header.  Fixed size to make header-only peeks cheap.
//
//  Layout on disk is a sequence of fixed-width fields; we do NOT use
//  a struct to avoid implicit padding surprises across toolchains.
// --------------------------------------------------------------------
struct HeaderRec {
    char     magic[4];
    uint32_t version;
    uint32_t value_bytes;
    uint32_t index_bytes;
    uint32_t layout_kind;
    uint32_t num_modes;
    uint64_t nnz;
    uint32_t mode_size[DYN_MAX_MODES];
    uint32_t num_threads;
    double   tnorm_sq;
    uint32_t reserved[14];
};

static void write_header(Out& o, const Tensor& T, int num_threads,
                         uint32_t layout_kind)
{
    HeaderRec h{};
    std::memcpy(h.magic, kDnpMagic, 4);
    h.version     = kDnpVersion;
    h.value_bytes = (uint32_t)sizeof(value_t);
    h.index_bytes = (uint32_t)sizeof(idx_t);
    h.layout_kind = layout_kind;
    h.num_modes   = (uint32_t)T.num_modes;
    h.nnz         = T.nnz;
    for (int n = 0; n < DYN_MAX_MODES; ++n)
        h.mode_size[n] = (n < T.num_modes) ? (uint32_t)T.mode_size[n] : 0u;
    h.num_threads = (uint32_t)num_threads;
    h.tnorm_sq    = T.ooc_tnorm_sq;  // may be zero when unknown

    o.w(h.magic);
    o.w(h.version);
    o.w(h.value_bytes);
    o.w(h.index_bytes);
    o.w(h.layout_kind);
    o.w(h.num_modes);
    o.w(h.nnz);
    o.wb(h.mode_size,  sizeof(h.mode_size));
    o.w(h.num_threads);
    o.w(h.tnorm_sq);
    o.wb(h.reserved,  sizeof(h.reserved));
}

static bool read_header(In& i, HeaderRec& h) {
    i.rb(h.magic, 4);
    i.r(h.version);
    i.r(h.value_bytes);
    i.r(h.index_bytes);
    i.r(h.layout_kind);
    i.r(h.num_modes);
    i.r(h.nnz);
    i.rb(h.mode_size,  sizeof(h.mode_size));
    i.r(h.num_threads);
    i.r(h.tnorm_sq);
    i.rb(h.reserved,  sizeof(h.reserved));
    return i.ok;
}

static bool validate_header(const HeaderRec& h, const char* path) {
    if (std::memcmp(h.magic, kDnpMagic, 4) != 0) {
        std::fprintf(stderr,
            "[dnp] %s: bad magic (not a Dynasor+ processed-tensor file).\n",
            path);
        return false;
    }
    if (h.version != kDnpVersion) {
        std::fprintf(stderr,
            "[dnp] %s: version %u not supported (this build expects %u).\n",
            path, h.version, kDnpVersion);
        return false;
    }
    if (h.value_bytes != sizeof(value_t)) {
        std::fprintf(stderr,
            "[dnp] %s: value_bytes=%u but build uses sizeof(value_t)=%zu.\n",
            path, h.value_bytes, sizeof(value_t));
        return false;
    }
    if (h.index_bytes != sizeof(idx_t)) {
        std::fprintf(stderr,
            "[dnp] %s: index_bytes=%u but build uses sizeof(idx_t)=%zu.\n",
            path, h.index_bytes, sizeof(idx_t));
        return false;
    }
    if (h.num_modes == 0 || h.num_modes > (uint32_t)DYN_MAX_MODES) {
        std::fprintf(stderr,
            "[dnp] %s: invalid num_modes=%u (max %d).\n",
            path, h.num_modes, DYN_MAX_MODES);
        return false;
    }
    if (h.nnz == 0) {
        std::fprintf(stderr, "[dnp] %s: nnz=0 is not supported.\n", path);
        return false;
    }
    if (h.layout_kind > kLayoutInPlace) {
        std::fprintf(stderr,
            "[dnp] %s: unknown layout_kind=%u.\n", path, h.layout_kind);
        return false;
    }
    return true;
}

// --------------------------------------------------------------------
//  Morton save / load.
// --------------------------------------------------------------------
static bool save_morton(Out& o, const Tensor& T) {
    if (!T.morton_keys || !T.morton_vals) {
        std::fprintf(stderr,
            "[dnp] save: Layout::Morton but morton_keys/vals are null.\n");
        return false;
    }
    o.wb(T.morton_mask, sizeof(T.morton_mask));
    o.wb(T.morton_bits, sizeof(T.morton_bits));
    // Align tail to 8 so morton_keys starts aligned on disk (cosmetic).
    const size_t bits_bytes = sizeof(T.morton_bits);
    const size_t pad = (8 - (bits_bytes & 7)) & 7;
    uint8_t zeros[8] = {0};
    if (pad) o.wb(zeros, pad);
    o.wb(T.morton_keys, (size_t)T.nnz * sizeof(uint64_t));
    o.wb(T.morton_vals, (size_t)T.nnz * sizeof(value_t));
    return o.ok;
}

static bool load_morton(In& in, Tensor& T) {
    in.rb(T.morton_mask, sizeof(T.morton_mask));
    in.rb(T.morton_bits, sizeof(T.morton_bits));
    const size_t bits_bytes = sizeof(T.morton_bits);
    const size_t pad = (8 - (bits_bytes & 7)) & 7;
    uint8_t scratch[8] = {0};
    if (pad) in.rb(scratch, pad);
    if (!in.ok) return false;

    constexpr size_t kA = DYN_ALIGN;
    const size_t keys_bytes = ((size_t)T.nnz * sizeof(uint64_t) + kA - 1) & ~(kA - 1);
    const size_t vals_bytes = ((size_t)T.nnz * sizeof(value_t)  + kA - 1) & ~(kA - 1);
    const size_t total      = keys_bytes + vals_bytes + kA;

    void* raw = dyn_aligned_alloc(total);
    if (!raw) {
        std::fprintf(stderr,
            "[dnp] load: OOM allocating %.2f GiB for morton keys+vals.\n",
            (double)total / (double)(1ULL << 30));
        return false;
    }
    uint64_t* keys = (uint64_t*)raw;
    value_t*  vals = (value_t*)((char*)raw + keys_bytes);

    in.rb(keys, (size_t)T.nnz * sizeof(uint64_t));
    in.rb(vals, (size_t)T.nnz * sizeof(value_t));
    if (!in.ok) {
        dyn_aligned_free(raw);
        return false;
    }

    T.morton_buf_raw   = raw;
    T.morton_buf_bytes = total;
    T.morton_keys      = keys;
    T.morton_vals      = vals;
    T.layout           = Layout::Morton;
    return true;
}

// --------------------------------------------------------------------
//  FLYCOO metadata block -- shared by NCopy / PingPong / InPlace.
//
//  Serializes every field the CP-ALS runtime reads at dispatch time:
//    - per-mode shard geometry (shards_per_mode, mn, mn_shift, shard_size)
//    - per-mode analytics (avg_fiber_len, kernel_class)
//    - touched bitmaps (size ceil(I_n/8) per mode)
//    - shard_begin / shard_end (size shards_per_mode[n])
//    - ss_list (per-mode, per-thread -- thread-specific!)
//    - precomp_thr_off (per-mode, size num_threads * row_stride[n])
//    - remap_row_stride, inplace_sorted_mode
// --------------------------------------------------------------------
static bool save_flycoo_meta(Out& o, const Tensor& T, int num_threads) {
    const int N = T.num_modes;

    // Fixed-size arrays (always DYN_MAX_MODES slots even when N < MAX).
    uint32_t spm32[DYN_MAX_MODES] = {0};
    uint32_t mn32 [DYN_MAX_MODES] = {0};
    int32_t  shft [DYN_MAX_MODES] = {0};
    double   afl  [DYN_MAX_MODES] = {0.0};
    uint32_t kcls [DYN_MAX_MODES] = {0};
    for (int n = 0; n < N; ++n) {
        spm32[n] = (uint32_t)T.shards_per_mode[n];
        mn32 [n] = (uint32_t)T.mn[n];
        shft [n] = (int32_t) T.mn_shift[n];
        afl  [n] = T.avg_fiber_len[n];
        kcls [n] = (uint32_t)T.kernel_class[n];
    }
    o.wb(spm32, sizeof(spm32));
    o.wb(mn32,  sizeof(mn32));
    o.wb(shft,  sizeof(shft));
    uint32_t shard_size32 = (uint32_t)T.shard_size;
    o.w(shard_size32);
    o.wb(afl,   sizeof(afl));
    o.wb(kcls,  sizeof(kcls));
    int8_t  inp_mode = (int8_t)T.inplace_sorted_mode;
    o.w(inp_mode);
    uint8_t pad7[7] = {0};
    o.wb(pad7, 7);
    uint64_t remap_stride = (uint64_t)T.remap_row_stride;
    o.w(remap_stride);

    // Touched bitmaps.
    for (int n = 0; n < N; ++n) {
        const uint64_t bytes = (T.touched_bits.size() > (size_t)n)
                             ? (uint64_t)T.touched_bits[n].size() : 0;
        const uint64_t cnt   = (T.touched_count.size() > (size_t)n)
                             ? T.touched_count[n] : 0;
        o.w(cnt);
        o.w(bytes);
        if (bytes) o.wb(T.touched_bits[n].data(), bytes);
    }

    // shard_begin / shard_end.
    for (int n = 0; n < N; ++n) {
        const uint64_t K = (uint64_t)T.shards_per_mode[n];
        o.w(K);
        if (K) {
            o.wb(T.shard_begin[n].data(), K * sizeof(uint64_t));
            o.wb(T.shard_end  [n].data(), K * sizeof(uint64_t));
        }
    }

    // ss_list: per-mode, per-thread.  Widen sid_t -> uint64_t for
    // format stability (sid_t is uint32 today but the format should
    // survive a future 64-bit widening).
    for (int n = 0; n < N; ++n) {
        const uint64_t NT = (uint64_t)num_threads;
        o.w(NT);
        if ((int)T.ss_list.size() < N || (int)T.ss_list[n].size() < num_threads) {
            std::fprintf(stderr,
                "[dnp] save: ss_list[%d] shape mismatch (expected [%d][%d]).\n",
                n, N, num_threads);
            o.ok = false;
            return false;
        }
        for (int t = 0; t < num_threads; ++t) {
            const auto&    src = T.ss_list[n][t];
            const uint64_t L   = (uint64_t)src.size();
            o.w(L);
            if (L) {
                // Widen sid_t -> uint64_t.
                std::vector<uint64_t> wide(L);
                for (uint64_t k = 0; k < L; ++k) wide[k] = (uint64_t)src[k];
                o.wb(wide.data(), L * sizeof(uint64_t));
            }
        }
    }

    // precomp_thr_off: per-mode flat uint64 array of size
    // num_threads * row_stride[n].  Written verbatim.
    for (int n = 0; n < N; ++n) {
        const uint64_t len = (uint64_t)T.precomp_thr_off[n].size();
        o.w(len);
        if (len) o.wb(T.precomp_thr_off[n].data(), len * sizeof(uint64_t));
    }

    return o.ok;
}

static bool load_flycoo_meta(In& in, Tensor& T, int num_threads) {
    const int N = T.num_modes;

    uint32_t spm32[DYN_MAX_MODES] = {0};
    uint32_t mn32 [DYN_MAX_MODES] = {0};
    int32_t  shft [DYN_MAX_MODES] = {0};
    double   afl  [DYN_MAX_MODES] = {0.0};
    uint32_t kcls [DYN_MAX_MODES] = {0};

    in.rb(spm32, sizeof(spm32));
    in.rb(mn32,  sizeof(mn32));
    in.rb(shft,  sizeof(shft));
    uint32_t shard_size32 = 0;
    in.r(shard_size32);
    in.rb(afl,   sizeof(afl));
    in.rb(kcls,  sizeof(kcls));
    int8_t inp_mode = 0;
    in.r(inp_mode);
    uint8_t pad7[7];
    in.rb(pad7, 7);
    uint64_t remap_stride = 0;
    in.r(remap_stride);
    if (!in.ok) return false;

    for (int n = 0; n < N; ++n) {
        T.shards_per_mode[n] = (idx_t)spm32[n];
        T.mn[n]              = (idx_t)mn32[n];
        T.mn_shift[n]        = (int)  shft[n];
        T.avg_fiber_len[n]   = afl[n];
        T.kernel_class[n]    = (KernelClass)kcls[n];
    }
    T.shard_size           = (idx_t)shard_size32;
    T.inplace_sorted_mode  = (int8_t)inp_mode;
    T.remap_row_stride     = (size_t)remap_stride;

    // Touched bitmaps.
    T.touched_bits.assign(N, {});
    T.touched_count.assign(N, 0);
    for (int n = 0; n < N; ++n) {
        uint64_t cnt = 0, bytes = 0;
        in.r(cnt);
        in.r(bytes);
        if (!in.ok) return false;
        T.touched_count[n] = cnt;
        if (bytes) {
            T.touched_bits[n].assign((size_t)bytes, 0);
            in.rb(T.touched_bits[n].data(), (size_t)bytes);
            if (!in.ok) return false;
        }
    }

    // shard_begin / shard_end.
    T.shard_begin.assign(N, {});
    T.shard_end  .assign(N, {});
    for (int n = 0; n < N; ++n) {
        uint64_t K = 0;
        in.r(K);
        if (!in.ok) return false;
        if (K != (uint64_t)T.shards_per_mode[n]) {
            std::fprintf(stderr,
                "[dnp] load: shard count mismatch mode %d (header=%u stream=%llu).\n",
                n, (unsigned)T.shards_per_mode[n], (unsigned long long)K);
            return false;
        }
        if (K) {
            T.shard_begin[n].assign((size_t)K, 0);
            T.shard_end  [n].assign((size_t)K, 0);
            in.rb(T.shard_begin[n].data(), (size_t)K * sizeof(uint64_t));
            in.rb(T.shard_end  [n].data(), (size_t)K * sizeof(uint64_t));
            if (!in.ok) return false;
        }
    }

    // ss_list.
    T.ss_list.assign(N, std::vector<std::vector<sid_t>>(num_threads));
    for (int n = 0; n < N; ++n) {
        uint64_t NT = 0;
        in.r(NT);
        if (!in.ok) return false;
        if (NT != (uint64_t)num_threads) {
            std::fprintf(stderr,
                "[dnp] load: ss_list thread count mismatch mode %d "
                "(file=%llu, caller=%d).\n",
                n, (unsigned long long)NT, num_threads);
            return false;
        }
        for (int t = 0; t < num_threads; ++t) {
            uint64_t L = 0;
            in.r(L);
            if (!in.ok) return false;
            if (L) {
                std::vector<uint64_t> wide((size_t)L);
                in.rb(wide.data(), (size_t)L * sizeof(uint64_t));
                if (!in.ok) return false;
                T.ss_list[n][t].resize((size_t)L);
                for (size_t k = 0; k < (size_t)L; ++k)
                    T.ss_list[n][t][k] = (sid_t)wide[k];
            }
        }
    }

    // precomp_thr_off.
    T.precomp_thr_off.assign(N, {});
    T.precomp_num_threads = num_threads;
    for (int n = 0; n < N; ++n) {
        uint64_t len = 0;
        in.r(len);
        if (!in.ok) return false;
        if (len) {
            T.precomp_thr_off[n].assign((size_t)len, 0);
            in.rb(T.precomp_thr_off[n].data(), (size_t)len * sizeof(uint64_t));
            if (!in.ok) return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------
//  PingPong / InPlace slab: one contiguous SoA block.
// --------------------------------------------------------------------
static bool save_soa_slab(Out& o, const Tensor& T) {
    if (!T.vals) {
        std::fprintf(stderr,
            "[dnp] save: primary SoA slab not present (vals=nullptr). "
            "Cannot serialize PingPong/InPlace under compaction.\n");
        return false;
    }
    const uint64_t n = T.nnz;
    o.wb(T.vals, (size_t)n * sizeof(value_t));
    for (int m = 0; m < T.num_modes; ++m) {
        if (!T.idx_buf[m]) {
            std::fprintf(stderr,
                "[dnp] save: idx_buf[%d] is null -- cannot serialize.\n", m);
            return false;
        }
        o.wb(T.idx_buf[m], (size_t)n * sizeof(idx_t));
    }
    return o.ok;
}

static bool load_soa_slab(In& in, Tensor& T) {
    T.allocate_elements(T.nnz, T.num_modes);
    in.rb(T.vals, (size_t)T.nnz * sizeof(value_t));
    for (int m = 0; m < T.num_modes; ++m) {
        in.rb(T.idx_buf[m], (size_t)T.nnz * sizeof(idx_t));
    }
    return in.ok;
}

// --------------------------------------------------------------------
//  NCopy slabs: N per-copy SoA slabs + optional CSR rowptrs.
//
//  Compacted NCopy (where buf_ncopy_compact[k] is non-null and
//  ncopy_idx[k][k] may be null) is rejected by the writer because
//  compaction frees the primary/scratch slabs that the .dnp loader
//  would need to re-materialize the canonical copy-0/copy-1 aliases.
//  Users with compact runs should re-run with --ncopy-csr-compact off
//  before --save-processed.
// --------------------------------------------------------------------
static bool ncopy_is_compact(const Tensor& T) {
    for (int k = 0; k < T.num_modes; ++k)
        if (T.buf_ncopy_compact[k] != nullptr) return true;
    return false;
}

static bool save_ncopy_slabs(Out& o, const Tensor& T) {
    if (ncopy_is_compact(T)) {
        std::fprintf(stderr,
            "[dnp] save: NCopy + CSR compaction is not serializable.\n"
            "       Re-run with --ncopy-csr-compact off and --save-processed.\n");
        return false;
    }
    const int      N = T.num_modes;
    const uint64_t n = T.nnz;

    uint8_t fiber_sorted[DYN_MAX_MODES] = {0};
    uint8_t csr_flag    [DYN_MAX_MODES] = {0};
    for (int k = 0; k < N; ++k) {
        fiber_sorted[k] = T.ncopy_fiber_sorted[k] ? 1 : 0;
        csr_flag    [k] = T.ncopy_csr         [k] ? 1 : 0;
    }
    o.wb(fiber_sorted, sizeof(fiber_sorted));
    o.wb(csr_flag,     sizeof(csr_flag));

    for (int k = 0; k < N; ++k) {
        if (!T.ncopy_vals[k]) {
            std::fprintf(stderr,
                "[dnp] save: ncopy_vals[%d] is null.\n", k);
            return false;
        }
        o.wb(T.ncopy_vals[k], (size_t)n * sizeof(value_t));
        for (int m = 0; m < N; ++m) {
            if (!T.ncopy_idx[k][m]) {
                std::fprintf(stderr,
                    "[dnp] save: ncopy_idx[%d][%d] is null (compact run?).\n",
                    k, m);
                return false;
            }
            o.wb(T.ncopy_idx[k][m], (size_t)n * sizeof(idx_t));
        }
        if (T.ncopy_csr[k]) {
            if (!T.ncopy_rowptr[k]) {
                std::fprintf(stderr,
                    "[dnp] save: ncopy_csr[%d]=true but rowptr is null.\n", k);
                return false;
            }
            const uint64_t cnt = (uint64_t)T.mode_size[k] + 1ULL;
            o.wb(T.ncopy_rowptr[k], (size_t)cnt * sizeof(uint64_t));
        }
    }
    return o.ok;
}

static bool load_ncopy_slabs(In& in, Tensor& T) {
    const int      N = T.num_modes;
    const uint64_t n = T.nnz;

    uint8_t fiber_sorted[DYN_MAX_MODES] = {0};
    uint8_t csr_flag    [DYN_MAX_MODES] = {0};
    in.rb(fiber_sorted, sizeof(fiber_sorted));
    in.rb(csr_flag,     sizeof(csr_flag));
    if (!in.ok) return false;

    // Set up copy-0 / copy-1 slab aliases via the canonical primary +
    // scratch allocations, then read each copy's data directly into
    // its backing slab.  Copies 2..N-1 get dedicated slabs.
    T.allocate_elements(n, N);
    T.ensure_scratch();

    for (int k = 0; k < N; ++k) {
        value_t* dst_vals = nullptr;
        idx_t*   dst_idx[DYN_MAX_MODES] = {nullptr};

        if (k == 0) {
            dst_vals = T.vals;
            for (int m = 0; m < N; ++m) dst_idx[m] = T.idx_buf[m];
        } else if (k == 1) {
            dst_vals = T.scr_vals;
            for (int m = 0; m < N; ++m) dst_idx[m] = T.scr_idx[m];
        } else {
            T.ensure_ncopy_slab(k);
            dst_vals = T.ncopy_vals[k];
            for (int m = 0; m < N; ++m) dst_idx[m] = T.ncopy_idx[k][m];
        }

        if (!dst_vals) {
            std::fprintf(stderr, "[dnp] load: ncopy slab %d not allocated.\n", k);
            return false;
        }
        in.rb(dst_vals, (size_t)n * sizeof(value_t));
        for (int m = 0; m < N; ++m) {
            in.rb(dst_idx[m], (size_t)n * sizeof(idx_t));
        }
        if (!in.ok) return false;

        // Read CSR rowptr if flagged.
        if (csr_flag[k]) {
            T.ensure_ncopy_rowptr(k);
            const uint64_t cnt = (uint64_t)T.mode_size[k] + 1ULL;
            in.rb(T.ncopy_rowptr[k], (size_t)cnt * sizeof(uint64_t));
            if (!in.ok) return false;
        }

        T.ncopy_fiber_sorted[k] = fiber_sorted[k] ? true : false;
        T.ncopy_csr[k]          = csr_flag[k]    ? true : false;
    }

    // Wire copy-0 / copy-1 aliases to match populate_ncopy_layout's
    // final state.  ensure_ncopy_slab() is idempotent for slots 0/1.
    T.ensure_ncopy_slab(0);
    T.ensure_ncopy_slab(1);
    T.layout = Layout::NCopy;
    return true;
}

} // namespace

// --------------------------------------------------------------------
//  Public save / load entry points.
// --------------------------------------------------------------------

bool save_dnp_tensor(const std::string& path,
                     const Tensor&      T,
                     int                num_threads)
{
    if (T.num_modes <= 0 || T.nnz == 0) {
        std::fprintf(stderr,
            "[dnp] save: empty tensor (num_modes=%d, nnz=%llu).\n",
            T.num_modes, (unsigned long long)T.nnz);
        return false;
    }
    if (T.layout == Layout::OutOfCore) {
        std::fprintf(stderr,
            "[dnp] save: Layout::OutOfCore is not serialized as .dnp. "
            "The existing .dnb cache already provides runtime-only "
            "execution for OOC tensors (no preprocessing happens).\n");
        return false;
    }
    const uint32_t lk = layout_to_kind(T.layout);
    if (lk == (uint32_t)-1) {
        std::fprintf(stderr, "[dnp] save: unknown layout (%d).\n",
                     (int)T.layout);
        return false;
    }

    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "[dnp] save: fopen '%s' failed.\n", path.c_str());
        return false;
    }
    Out o{fp, true};

    write_header(o, T, num_threads, lk);

    bool body_ok = false;
    switch (lk) {
        case kLayoutMorton:
            body_ok = save_morton(o, T);
            break;
        case kLayoutPingPong:
        case kLayoutInPlace:
            body_ok = save_flycoo_meta(o, T, num_threads) &&
                      save_soa_slab(o, T);
            break;
        case kLayoutNCopy:
            body_ok = save_flycoo_meta(o, T, num_threads) &&
                      save_ncopy_slabs(o, T);
            break;
    }

    const bool ok = o.ok && body_ok;
    std::fclose(fp);
    if (!ok) {
        std::fprintf(stderr,
            "[dnp] save: I/O failure writing '%s'. The file may be corrupt.\n",
            path.c_str());
        std::remove(path.c_str());
    }
    return ok;
}

bool load_dnp_header(const std::string& path, Tensor& T) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    In in{fp, true};
    HeaderRec h{};
    const bool ok = read_header(in, h) && validate_header(h, path.c_str());
    std::fclose(fp);
    if (!ok) return false;

    T.num_modes = (int)h.num_modes;
    T.nnz       = h.nnz;
    for (int n = 0; n < T.num_modes; ++n)
        T.mode_size[n] = (idx_t)h.mode_size[n];
    T.layout = kind_to_layout(h.layout_kind);
    return true;
}

int load_dnp_num_threads(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return -1;
    In in{fp, true};
    HeaderRec h{};
    const bool ok = read_header(in, h) && validate_header(h, path.c_str());
    std::fclose(fp);
    if (!ok) return -1;
    return (int)h.num_threads;
}

Layout load_dnp_layout(const std::string& path) {
    Tensor tmp;
    if (!load_dnp_header(path, tmp)) return Layout::PingPong;
    return tmp.layout;
}

bool load_dnp_tensor(const std::string& path,
                     Tensor&            T,
                     int                num_threads)
{
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[dnp] load: fopen '%s' failed.\n", path.c_str());
        return false;
    }
    In in{fp, true};
    HeaderRec h{};
    if (!read_header(in, h) || !validate_header(h, path.c_str())) {
        std::fclose(fp);
        return false;
    }
    if ((int)h.num_threads != num_threads) {
        std::fprintf(stderr,
            "[dnp] load: %s was saved with %u threads but you requested %d.\n"
            "       The FLYCOO schedule and per-thread cursors are baked\n"
            "       in for that specific thread count.  Re-run with\n"
            "       --threads %u, or regenerate the .dnp with your new\n"
            "       thread count.\n",
            path.c_str(), h.num_threads, num_threads, h.num_threads);
        std::fclose(fp);
        return false;
    }

    // Populate header-level tensor fields (allocate_elements is called
    // later by the layout-specific loader when a slab is required).
    T.num_modes = (int)h.num_modes;
    T.nnz       = h.nnz;
    for (int n = 0; n < T.num_modes; ++n)
        T.mode_size[n] = (idx_t)h.mode_size[n];
    T.ooc_tnorm_sq = h.tnorm_sq;

    bool body_ok = false;
    switch (h.layout_kind) {
        case kLayoutMorton:
            body_ok = load_morton(in, T);
            break;
        case kLayoutPingPong:
            if (load_flycoo_meta(in, T, num_threads) &&
                load_soa_slab(in, T))
            {
                T.layout = Layout::PingPong;
                body_ok = true;
            }
            break;
        case kLayoutInPlace:
            if (load_flycoo_meta(in, T, num_threads) &&
                load_soa_slab(in, T))
            {
                T.layout = Layout::InPlace;
                body_ok = true;
            }
            break;
        case kLayoutNCopy:
            if (load_flycoo_meta(in, T, num_threads) &&
                load_ncopy_slabs(in, T))
            {
                body_ok = true;
            }
            break;
    }

    std::fclose(fp);
    if (!body_ok) {
        std::fprintf(stderr,
            "[dnp] load: '%s' (layout=%s) body read failed or truncated.\n",
            path.c_str(), kind_name(h.layout_kind));
        return false;
    }
    return true;
}

} // namespace dynasor
