// ============================================================================
//  dynasor_ooc.cpp -- out-of-core (streaming from disk) MTTKRP + CP-ALS
//  support.  See dynasor_ooc.h for the overall design.
//
//  The .dnb cache format (dynasor_io.cpp) has a compact SoA layout:
//
//      [HEADER (100 B)] [vals[nnz]] [idx[0][nnz]] [idx[1][nnz]] ...
//
//  so reading one chunk of nonzeros requires exactly (N+1) seek+fread
//  pairs, all at large and aligned offsets.  No format conversion is
//  needed.  When the file is smaller than free RAM the OS page cache
//  transparently serves subsequent iterations from memory without
//  any code changes on our side.
// ============================================================================
#include "dynasor_ooc.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

#ifdef _WIN32
#  include <sys/stat.h>
#  define DYN_FSEEK64 ::_fseeki64
#else
#  include <sys/stat.h>
#  define DYN_FSEEK64 ::fseeko
#endif

namespace dynasor {

// Forward declaration: lives in dynasor_kernel.cpp (namespace dynasor) and
// drives every all-modes chunk of nonzeros.  Streaming is entirely a data-
// access concern, so we reuse the existing in-core kernel verbatim once a
// chunk is resident in RAM.
void dynasor_process_shard_all_modes(
    const value_t* vals,
    const idx_t*   const* idx,
    uint64_t b, uint64_t e,
    int num_modes, int rank, int rank_padded,
    const value_t* const* Y,
    value_t* const* Yout);

// Resolve OpenMP thread count from the caller's hint, clamping to the
// system maximum.  Duplicated from dynasor.cpp rather than #included to
// keep this translation unit free of the rest of that driver's state.
static int ooc_resolve_threads(int num_threads) {
#ifdef _OPENMP
    int maxt = omp_get_max_threads();
#else
    int maxt = 1;
#endif
    if (num_threads <= 0) num_threads = maxt;
    if (num_threads > maxt) num_threads = maxt;
    return num_threads;
}

// ---------------------------------------------------------------------------
//  .dnb header layout (mirrored from dynasor_io.cpp -- kept in-sync-by-hand
//  since the writer lives there and the header size is small/stable).
// ---------------------------------------------------------------------------
namespace {

constexpr char     kDnbMagic[4]  = {'D','N','B','1'};
constexpr uint32_t kDnbVersion   = 1;
constexpr size_t   kDnbHeaderSize =
    4 /*magic*/ + 4 /*version*/ + 4 /*vbytes*/ + 4 /*ibytes*/ +
    4 /*nmodes*/ + 8 /*nnz*/ + 4 * DYN_MAX_MODES /*mode_sizes*/ +
    4 * 7 /*reserved*/;

static_assert(kDnbHeaderSize == 4 + 4 + 4 + 4 + 4 + 8 + 4 * DYN_MAX_MODES + 4 * 7,
              "OOC header size must match dynasor_io.cpp");

} // anonymous namespace

// ---------------------------------------------------------------------------
//  ooc_load_header -- read metadata only, no slab allocation.
// ---------------------------------------------------------------------------
bool ooc_load_header(const std::string& path, Tensor& T) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[ooc] open failed: %s (%s)\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }

    char     magic[4];
    uint32_t version = 0, value_bytes = 0, index_bytes = 0, num_modes_u = 0;
    uint64_t nnz_u   = 0;
    uint32_t mode_sizes[DYN_MAX_MODES] = {0};
    uint32_t reserved  [7]             = {0};

    const bool ok =
        std::fread(magic,        sizeof(magic),        1, fp) == 1 &&
        std::fread(&version,     sizeof(version),      1, fp) == 1 &&
        std::fread(&value_bytes, sizeof(value_bytes),  1, fp) == 1 &&
        std::fread(&index_bytes, sizeof(index_bytes),  1, fp) == 1 &&
        std::fread(&num_modes_u, sizeof(num_modes_u),  1, fp) == 1 &&
        std::fread(&nnz_u,       sizeof(nnz_u),        1, fp) == 1 &&
        std::fread(mode_sizes,   sizeof(mode_sizes),   1, fp) == 1 &&
        std::fread(reserved,     sizeof(reserved),     1, fp) == 1;
    std::fclose(fp);

    if (!ok || std::memcmp(magic, kDnbMagic, 4) != 0 ||
        version != kDnbVersion ||
        value_bytes != sizeof(value_t) ||
        index_bytes != sizeof(idx_t) ||
        num_modes_u == 0 || num_modes_u > (uint32_t)DYN_MAX_MODES ||
        nnz_u == 0)
    {
        std::fprintf(stderr, "[ooc] invalid .dnb header: %s\n", path.c_str());
        return false;
    }

    T.num_modes = (int)num_modes_u;
    T.nnz       = nnz_u;
    for (int n = 0; n < T.num_modes; ++n) {
        T.mode_size[n] = (idx_t)mode_sizes[n];
    }
    T.ooc_header_bytes = (uint64_t)kDnbHeaderSize;
    return true;
}

// ---------------------------------------------------------------------------
//  ooc_precompute_norm -- streaming pass over vals[nnz].
//
//  Reads only the vals column (no idx traffic) in large aligned chunks.
//  Accumulates in double to avoid catastrophic cancellation when nnz is
//  in the hundreds of millions.  Per-chunk partial sums are reduced
//  across OpenMP threads.
// ---------------------------------------------------------------------------
bool ooc_precompute_norm(Tensor& T) {
    if (T.ooc_path.empty() || T.nnz == 0) return false;

    std::FILE* fp = std::fopen(T.ooc_path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[ooc] norm: open failed: %s\n",
                     T.ooc_path.c_str());
        return false;
    }
    if (DYN_FSEEK64(fp, (int64_t)T.ooc_header_bytes, SEEK_SET) != 0) {
        std::fclose(fp);
        return false;
    }

    // 32 MiB per read -- big enough that syscall / metadata cost is
    // negligible, small enough that the buffer fits in L2 when the
    // page cache is already warm.
    constexpr uint64_t kNormBatch = (32ULL << 20) / sizeof(value_t);
    std::vector<value_t> buf((size_t)std::min<uint64_t>(kNormBatch, T.nnz));

    double   accum     = 0.0;
    uint64_t remaining = T.nnz;
    while (remaining > 0) {
        const uint64_t want = std::min<uint64_t>(remaining, (uint64_t)buf.size());
        const size_t   got  = std::fread(buf.data(), sizeof(value_t),
                                         (size_t)want, fp);
        if (got != (size_t)want) {
            std::fprintf(stderr,
                "[ooc] norm: short read (got %zu, want %llu) at offset %llu\n",
                got, (unsigned long long)want,
                (unsigned long long)(T.nnz - remaining));
            std::fclose(fp);
            return false;
        }
        double partial = 0.0;
        #pragma omp parallel for reduction(+:partial) schedule(static)
        for (int64_t i = 0; i < (int64_t)got; ++i) {
            const double v = (double)buf[(size_t)i];
            partial += v * v;
        }
        accum     += partial;
        remaining -= want;
    }

    std::fclose(fp);
    T.ooc_tnorm_sq = accum;
    return true;
}

// ---------------------------------------------------------------------------
//  ooc_default_chunk_nnz -- target ~256 MiB per chunk, clamped sensibly.
// ---------------------------------------------------------------------------
uint64_t ooc_default_chunk_nnz(const Tensor& T) {
    if (T.num_modes <= 0 || T.nnz == 0) return 0;
    const uint64_t bytes_per_nnz =
        (uint64_t)sizeof(value_t) + (uint64_t)T.num_modes * (uint64_t)sizeof(idx_t);

    const char* env = std::getenv("DYN_OOC_CHUNK_BYTES");
    uint64_t target_bytes = (256ULL << 20);   // 256 MiB default
    if (env && env[0]) {
        const long long v = std::atoll(env);
        if (v > 0) target_bytes = (uint64_t)v;
    }

    uint64_t chunk = target_bytes / bytes_per_nnz;
    if (chunk < 65536) chunk = 65536;        // floor: 64 ki nnz
    if (chunk > T.nnz) chunk = T.nnz;
    chunk &= ~(uint64_t)63;                   // multiple of 64 for SIMD
    if (chunk == 0) chunk = T.nnz;
    return chunk;
}

// ===========================================================================
//  OocStream
// ===========================================================================
OocStream::OocStream() = default;
OocStream::~OocStream() { close(); }

bool OocStream::open(const Tensor& T,
                     const std::string& path,
                     uint64_t chunk_nnz_in)
{
    close();

    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) {
        std::fprintf(stderr, "[ooc] stream: open failed: %s (%s)\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    // Reduce stdio buffering noise: we do our own large reads.
    std::setvbuf(fp_, nullptr, _IONBF, 0);

    // Light header validation.  We trust T because the caller obtained
    // it via ooc_load_header from the same file.
    char magic[4];
    if (std::fread(magic, sizeof(magic), 1, fp_) != 1 ||
        std::memcmp(magic, kDnbMagic, 4) != 0)
    {
        std::fprintf(stderr, "[ooc] stream: bad magic: %s\n", path.c_str());
        close();
        return false;
    }

    num_modes_    = T.num_modes;
    total_nnz_    = T.nnz;
    header_bytes_ = (T.ooc_header_bytes != 0)
                        ? T.ooc_header_bytes
                        : (uint64_t)kDnbHeaderSize;

    chunk_nnz_ = chunk_nnz_in;
    if (chunk_nnz_ == 0) chunk_nnz_ = ooc_default_chunk_nnz(T);
    if (chunk_nnz_ > total_nnz_) chunk_nnz_ = total_nnz_;

    // Allocate one contiguous buffer: [vals_pad][idx0][idx1]...
    const size_t val_bytes = (size_t)chunk_nnz_ * sizeof(value_t);
    const size_t idx_bytes = (size_t)chunk_nnz_ * sizeof(idx_t);
    const size_t val_pad   = (val_bytes + 63ULL) & ~63ULL;
    const size_t idx_pad   = (idx_bytes + 63ULL) & ~63ULL;
    buf_bytes_ = val_pad + (size_t)num_modes_ * idx_pad;
    buf_raw_   = dyn_aligned_alloc(buf_bytes_);
    if (!buf_raw_) {
        std::fprintf(stderr,
            "[ooc] stream: cannot allocate %.2f MiB chunk buffer\n",
            (double)buf_bytes_ / (double)(1ULL << 20));
        close();
        return false;
    }
    char* base = (char*)buf_raw_;
    vals_ = (value_t*)base;
    base += val_pad;
    for (int n = 0; n < num_modes_; ++n) {
        idx_cols_[n] = (idx_t*)base;
        idx_ptrs_[n] = idx_cols_[n];
        base += idx_pad;
    }
    for (int n = num_modes_; n < DYN_MAX_MODES; ++n) {
        idx_cols_[n] = nullptr;
        idx_ptrs_[n] = nullptr;
    }
    return true;
}

void OocStream::close() {
    if (fp_) { std::fclose(fp_); fp_ = nullptr; }
    if (buf_raw_) { dyn_aligned_free(buf_raw_); buf_raw_ = nullptr; }
    buf_bytes_ = 0;
    vals_      = nullptr;
    for (int n = 0; n < DYN_MAX_MODES; ++n) {
        idx_cols_[n] = nullptr;
        idx_ptrs_[n] = nullptr;
    }
}

uint64_t OocStream::read_chunk(uint64_t b) {
    if (!fp_ || b >= total_nnz_) return 0;
    const uint64_t e = std::min<uint64_t>(total_nnz_, b + chunk_nnz_);
    const uint64_t len = e - b;

    // vals
    {
        const int64_t off = (int64_t)(header_bytes_ + b * sizeof(value_t));
        if (DYN_FSEEK64(fp_, off, SEEK_SET) != 0) return 0;
        const size_t got = std::fread(vals_, sizeof(value_t),
                                      (size_t)len, fp_);
        if (got != (size_t)len) {
            std::fprintf(stderr,
                "[ooc] read_chunk: short vals read at b=%llu (got %zu / %llu)\n",
                (unsigned long long)b, got, (unsigned long long)len);
            return 0;
        }
    }
    // idx[n]
    const uint64_t idx_base = header_bytes_ + total_nnz_ * sizeof(value_t);
    for (int n = 0; n < num_modes_; ++n) {
        const int64_t off = (int64_t)(idx_base
                          + (uint64_t)n * total_nnz_ * sizeof(idx_t)
                          + b * sizeof(idx_t));
        if (DYN_FSEEK64(fp_, off, SEEK_SET) != 0) return 0;
        const size_t got = std::fread(idx_cols_[n], sizeof(idx_t),
                                      (size_t)len, fp_);
        if (got != (size_t)len) {
            std::fprintf(stderr,
                "[ooc] read_chunk: short idx[%d] read at b=%llu (got %zu / %llu)\n",
                n, (unsigned long long)b, got, (unsigned long long)len);
            return 0;
        }
    }
    return len;
}

// ===========================================================================
//  spmttkrp_all_modes_ooc
// ===========================================================================
double spmttkrp_all_modes_ooc(Tensor& T,
                              FactorMatrices& F,
                              int num_threads)
{
    if (!T.ooc_enabled || T.ooc_path.empty()) {
        std::fprintf(stderr,
            "[ooc] spmttkrp_all_modes_ooc called but T.ooc_enabled is false; "
            "this is a bug in the dispatch path.\n");
        return 0.0;
    }

    const int      N   = T.num_modes;
    const int      R   = F.rank;
    const int      Rp  = F.rank_padded;
    const uint64_t nnz = T.nnz;
    (void)R;

    num_threads = ooc_resolve_threads(num_threads);

    const uint64_t chunk_nnz = (T.ooc_chunk_nnz != 0)
                                   ? T.ooc_chunk_nnz
                                   : ooc_default_chunk_nnz(T);

    OocStream stream;
    if (!stream.open(T, T.ooc_path, chunk_nnz)) {
        return 0.0;
    }

    const char* timing_env = std::getenv("DYN_OOC_TIMING");
    const bool  timing_on  = timing_env && timing_env[0] && timing_env[0] != '0';

    // ---- ofibs layout (identical to in-core all-modes driver) ----
    std::vector<size_t> ofib_off((size_t)N + 1, 0);
    for (int n = 0; n < N; ++n) {
        size_t rows  = (size_t)T.mode_size[n];
        size_t bytes = rows * (size_t)Rp * sizeof(value_t);
        bytes = (bytes + 63ULL) & ~63ULL;
        ofib_off[n + 1] = ofib_off[n] + bytes / sizeof(value_t);
    }
    const size_t stride_values = ofib_off[N];
    const size_t stride_bytes  = stride_values * sizeof(value_t);
    const size_t total_bytes   = stride_bytes * (size_t)num_threads;

    value_t* ofibs_raw = (value_t*)dyn_aligned_alloc(total_bytes);
    if (!ofibs_raw) {
        std::fprintf(stderr,
            "[ooc] cannot allocate %.2f GiB of per-thread Yhat buffers\n",
            (double)total_bytes / (double)(1ULL << 30));
        return 0.0;
    }

    std::vector<std::vector<value_t*>> Yout_by_thread(
        (size_t)num_threads, std::vector<value_t*>((size_t)N, nullptr));
    for (int tid = 0; tid < num_threads; ++tid) {
        value_t* base = ofibs_raw + (size_t)tid * stride_values;
        for (int n = 0; n < N; ++n)
            Yout_by_thread[(size_t)tid][(size_t)n] = base + ofib_off[(size_t)n];
    }

    const value_t* const* Y_in =
        const_cast<const value_t* const*>(F.Y.data());

    const double t0 = omp_get_wtime();

    // ---- A. zero ofibs (parallel) ----
    #pragma omp parallel num_threads(num_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (tid < num_threads) {
            std::memset(ofibs_raw + (size_t)tid * stride_values,
                        0, stride_bytes);
        }
    }

    // ---- B. streaming loop ----
    //
    //  Each chunk is: (1) read from disk synchronously, (2) processed in
    //  parallel across num_threads workers, each writing into its own
    //  private ofibs.  We partition the chunk's nnz range evenly across
    //  workers; no cross-thread contention.
    double io_time   = 0.0;
    double comp_time = 0.0;
    uint64_t chunks_done = 0;

    for (uint64_t b = 0; b < nnz; b += chunk_nnz) {
        const double ti0 = omp_get_wtime();
        const uint64_t got = stream.read_chunk(b);
        const double ti1 = omp_get_wtime();
        io_time += (ti1 - ti0);
        if (got == 0) {
            std::fprintf(stderr,
                "[ooc] streaming aborted at chunk start b=%llu\n",
                (unsigned long long)b);
            dyn_aligned_free(ofibs_raw);
            stream.close();
            return 0.0;
        }

        // Process this chunk in parallel.  Each worker handles a
        // contiguous slice [wb, we) inside [0, got).
        const uint64_t per = (got + (uint64_t)num_threads - 1) /
                             (uint64_t)num_threads;
        const value_t*       vals_c = stream.vals();
        const idx_t* const*  idx_c  = stream.idx();

        #pragma omp parallel num_threads(num_threads)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            if (tid < num_threads) {
                const uint64_t wb = (uint64_t)tid * per;
                const uint64_t we = std::min(got, wb + per);
                if (wb < we) {
                    value_t* Yout_local[DYN_MAX_MODES];
                    for (int n = 0; n < N; ++n)
                        Yout_local[n] =
                            Yout_by_thread[(size_t)tid][(size_t)n];

                    dynasor_process_shard_all_modes(
                        vals_c,
                        idx_c,
                        wb, we,
                        N, R, Rp,
                        Y_in,
                        Yout_local);
                }
            }
        }
        const double ti2 = omp_get_wtime();
        comp_time += (ti2 - ti1);

        if (timing_on) {
            std::printf("[ooc]   chunk %3llu  b=%11llu  got=%9llu  "
                        "io=%6.3f s  comp=%6.3f s\n",
                        (unsigned long long)chunks_done,
                        (unsigned long long)b,
                        (unsigned long long)got,
                        ti1 - ti0, ti2 - ti1);
        }
        ++chunks_done;
    }

    // ---- C. reduction ofibs -> Yhat (identical to in-core path) ----
    const double tr0 = omp_get_wtime();
    for (int n = 0; n < N; ++n) {
        const idx_t rows = T.mode_size[n];
        value_t* DYN_RESTRICT dst = F.Yhat[n];
        const size_t row_stride = (size_t)Rp;

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int64_t r64 = 0; r64 < (int64_t)rows; ++r64) {
            const size_t r = (size_t)r64;
            value_t* DYN_RESTRICT out = dst + r * row_stride;

            const value_t* DYN_RESTRICT src0 =
                ofibs_raw + 0 * stride_values + ofib_off[(size_t)n]
                          + r * row_stride;
            for (int c = 0; c < (int)row_stride; c += DYN_SIMD_WIDTH) {
                dyn_vstore(out + c, dyn_vload(src0 + c));
            }
            for (int t = 1; t < num_threads; ++t) {
                const value_t* DYN_RESTRICT src =
                    ofibs_raw + (size_t)t * stride_values
                              + ofib_off[(size_t)n]
                              + r * row_stride;
                for (int c = 0; c < (int)row_stride; c += DYN_SIMD_WIDTH) {
                    dyn_vstore(out + c,
                               dyn_vadd(dyn_vload(out + c),
                                        dyn_vload(src + c)));
                }
            }
        }
    }
    const double tr1 = omp_get_wtime();
    const double reduce_time = tr1 - tr0;

    const double t1 = omp_get_wtime();
    const double elapsed = t1 - t0;

    std::printf("[ooc] all-modes streamed: %llu chunks of up to %llu nnz each  "
                "(io=%.3f s  comp=%.3f s  reduce=%.3f s  total=%.3f s)\n",
                (unsigned long long)chunks_done,
                (unsigned long long)chunk_nnz,
                io_time, comp_time, reduce_time, elapsed);

    dyn_aligned_free(ofibs_raw);
    stream.close();
    return elapsed;
}

} // namespace dynasor
