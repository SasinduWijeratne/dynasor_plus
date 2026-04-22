// ----------------------------------------------------------------------------
// dnb_to_tns -- standalone converter from Dynasor+'s .dnb binary tensor cache
//               to the text .tns format used by ALTO / FROSTT.
//
// Usage: dnb_to_tns <input.dnb> <output.tns> [--one-indexed]
//
// The .dnb layout is documented in src/dynasor_io.cpp:
//   char     magic[4]   = "DNB1"
//   uint32_t version
//   uint32_t value_bytes    (= 4, float32)
//   uint32_t index_bytes    (= 4, uint32)
//   uint32_t num_modes
//   uint64_t nnz
//   uint32_t mode_size[DYN_MAX_MODES=8]
//   uint32_t reserved[7]
//   float    vals[nnz]
//   uint32_t idx[0][nnz] ... idx[N-1][nnz]
//
// FROSTT / ALTO tensors are 1-indexed, so we shift indices by +1 by default.
// Pass --zero-indexed to keep them zero-based.
// ----------------------------------------------------------------------------
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int  DYN_MAX_MODES      = 8;
constexpr char kMagic[4]          = {'D','N','B','1'};
constexpr size_t kHeaderBytes     = 4 + 4 + 4 + 4 + 4 + 8 + 4 * DYN_MAX_MODES + 4 * 7;

struct Header {
    uint32_t version;
    uint32_t value_bytes;
    uint32_t index_bytes;
    uint32_t num_modes;
    uint64_t nnz;
    uint32_t mode_size[DYN_MAX_MODES];
};

static bool read_header(std::FILE* fp, Header& h) {
    char magic[4];
    uint32_t reserved[7];
    bool ok = std::fread(magic,            sizeof(magic),         1, fp) == 1 &&
              std::fread(&h.version,       sizeof(h.version),     1, fp) == 1 &&
              std::fread(&h.value_bytes,   sizeof(h.value_bytes), 1, fp) == 1 &&
              std::fread(&h.index_bytes,   sizeof(h.index_bytes), 1, fp) == 1 &&
              std::fread(&h.num_modes,     sizeof(h.num_modes),   1, fp) == 1 &&
              std::fread(&h.nnz,           sizeof(h.nnz),         1, fp) == 1 &&
              std::fread(h.mode_size,      sizeof(h.mode_size),   1, fp) == 1 &&
              std::fread(reserved,         sizeof(reserved),      1, fp) == 1;
    if (!ok) return false;
    if (std::memcmp(magic, kMagic, 4) != 0) return false;
    if (h.version != 1u) return false;
    if (h.value_bytes != 4u || h.index_bytes != 4u) return false;
    if (h.num_modes == 0u || h.num_modes > (uint32_t)DYN_MAX_MODES) return false;
    if (h.nnz == 0ull) return false;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <input.dnb> <output.tns> [--zero-indexed]\n"
            "  Default is 1-indexed output (FROSTT/ALTO convention).\n",
            argv[0]);
        return 2;
    }
    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    bool one_indexed = true;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--zero-indexed") one_indexed = false;
        else if (std::string(argv[i]) == "--one-indexed") one_indexed = true;
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    std::FILE* fp = std::fopen(in_path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "error: cannot open %s\n", in_path.c_str());
        return 1;
    }
    Header h{};
    if (!read_header(fp, h)) {
        std::fprintf(stderr, "error: %s is not a valid .dnb cache\n", in_path.c_str());
        std::fclose(fp);
        return 1;
    }

    const int      N    = (int)h.num_modes;
    const uint64_t NNZ  = h.nnz;
    const uint64_t base = kHeaderBytes;

    std::fprintf(stdout,
        "[dnb_to_tns] %s  -- nnz=%llu, modes=%d, dims=[",
        in_path.c_str(), (unsigned long long)NNZ, N);
    for (int n = 0; n < N; ++n)
        std::fprintf(stdout, "%u%s", (unsigned)h.mode_size[n], (n+1<N?"x":""));
    std::fprintf(stdout, "], %s-indexed output\n",
                 one_indexed ? "1" : "0");
    std::fflush(stdout);

    // Streaming strategy: load chunks of (vals, idx[0..N-1]) for CHUNK nnz,
    // then write lines.  Each chunk requires N+1 seeks into the file.
    constexpr uint64_t CHUNK = 1ull << 20;  // 1 M nnz -> ~4 MiB per stream buffer

    std::vector<float>    vals(CHUNK);
    std::vector<std::vector<uint32_t>> idx((size_t)N, std::vector<uint32_t>(CHUNK));

    // Precompute file offsets of each stripe.
    const uint64_t vals_off = base;
    std::vector<uint64_t> idx_off((size_t)N);
    for (int n = 0; n < N; ++n)
        idx_off[(size_t)n] = base + (uint64_t)sizeof(float) * NNZ
                                  + (uint64_t)n * sizeof(uint32_t) * NNZ;

    std::FILE* out = std::fopen(out_path.c_str(), "wb");
    if (!out) {
        std::fprintf(stderr, "error: cannot open %s for writing\n", out_path.c_str());
        std::fclose(fp);
        return 1;
    }
    // Big output buffer -- ~4 MiB cuts syscall overhead for 335M lines.
    static char out_buf[1 << 22];
    std::setvbuf(out, out_buf, _IOFBF, sizeof(out_buf));

    auto seek_set = [](std::FILE* f, uint64_t off) -> bool {
#ifdef _WIN32
        return _fseeki64(f, (long long)off, SEEK_SET) == 0;
#else
        return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
    };

    const uint64_t shift = one_indexed ? 1ull : 0ull;

    char line[256];
    uint64_t emitted = 0;
    for (uint64_t b = 0; b < NNZ; b += CHUNK) {
        const uint64_t k = (NNZ - b < CHUNK) ? (NNZ - b) : CHUNK;

        if (!seek_set(fp, vals_off + b * sizeof(float))) goto io_err;
        if (std::fread(vals.data(), sizeof(float), (size_t)k, fp) != (size_t)k) goto io_err;
        for (int n = 0; n < N; ++n) {
            if (!seek_set(fp, idx_off[(size_t)n] + b * sizeof(uint32_t))) goto io_err;
            if (std::fread(idx[(size_t)n].data(), sizeof(uint32_t), (size_t)k, fp)
                != (size_t)k) goto io_err;
        }

        for (uint64_t i = 0; i < k; ++i) {
            int len = 0;
            for (int n = 0; n < N; ++n) {
                len += std::snprintf(line + len, sizeof(line) - (size_t)len,
                                     "%llu ",
                                     (unsigned long long)idx[(size_t)n][i] + shift);
            }
            len += std::snprintf(line + len, sizeof(line) - (size_t)len,
                                 "%.9g\n", (double)vals[i]);
            if ((size_t)len > 0 &&
                std::fwrite(line, 1, (size_t)len, out) != (size_t)len) {
                std::fprintf(stderr, "error: write failed at nz %llu\n",
                             (unsigned long long)(emitted + i));
                std::fclose(fp); std::fclose(out);
                return 1;
            }
        }
        emitted += k;
        if ((b / CHUNK) % 32 == 0) {
            std::fprintf(stdout, "\r[dnb_to_tns] %llu / %llu (%.1f%%)",
                         (unsigned long long)emitted,
                         (unsigned long long)NNZ,
                         100.0 * (double)emitted / (double)NNZ);
            std::fflush(stdout);
        }
    }
    std::fprintf(stdout, "\r[dnb_to_tns] %llu / %llu (100.0%%)\n",
                 (unsigned long long)emitted, (unsigned long long)NNZ);

    std::fclose(fp);
    std::fclose(out);
    return 0;

io_err:
    std::fprintf(stderr, "\nerror: I/O failed reading .dnb\n");
    std::fclose(fp);
    std::fclose(out);
    return 1;
}
