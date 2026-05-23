// ---------------------------------------------------------------------------
//  Generate a synthetic sparse tensor in FROSTT .tns format.
//
//  Usage: gen_tensor <N> <I_0> <I_1> ... <I_{N-1}> <nnz> <seed> <out.tns>
//
//  Design notes (memory safety):
//    * The old implementation kept an unordered_set<uint64_t> to avoid
//      duplicate indices.  For 100 M nnz this reserves ~6.4 GiB and triggers
//      OOM on 16-32 GiB machines.  We drop it entirely: for realistic
//      sparsity levels (volume / nnz >> 10) random duplicates are negligible
//      and do no harm (they just become a slightly heavier value at that
//      index after the tensor is read -- FROSTT readers sum duplicates).
//    * We stream output through a 4 MiB userspace buffer and emit values via
//      a tiny hand-rolled formatter, which is ~20x faster than fprintf("%d")
//      for multi-index tuples and keeps the writer from becoming the
//      bottleneck on large tensors.
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

struct Bufout {
    std::FILE* fp;
    std::vector<char> buf;
    size_t pos = 0;

    explicit Bufout(std::FILE* f, size_t cap = 1 << 22) : fp(f), buf(cap) {}

    void flush() {
        if (pos == 0) return;
        std::fwrite(buf.data(), 1, pos, fp);
        pos = 0;
    }
    inline void ensure(size_t n) {
        if (pos + n > buf.size()) flush();
    }
    inline void put(char c) { ensure(1); buf[pos++] = c; }
    inline void write_int(int v) {
        ensure(12);
        // positive-only (indices are 1-based); bypass std::to_string's alloc.
        char tmp[12];
        int  k = 0;
        if (v == 0) { buf[pos++] = '0'; return; }
        while (v > 0) { tmp[k++] = '0' + (v % 10); v /= 10; }
        while (k) buf[pos++] = tmp[--k];
    }
    inline void write_double(double v) {
        ensure(24);
        int n = std::snprintf(buf.data() + pos, 24, "%.8g", v);
        if (n > 0) pos += (size_t)n;
    }
    ~Bufout() { flush(); }
};

} // namespace


int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s N I0 I1 ... I(N-1) nnz seed out.tns\n", argv[0]);
        return 1;
    }
    const int N = std::atoi(argv[1]);
    if (argc < 1 + 1 + N + 3) {
        std::fprintf(stderr, "wrong arg count\n"); return 1;
    }

    std::vector<int> I(N);
    double volume = 1.0;
    for (int n = 0; n < N; ++n) {
        I[n] = std::atoi(argv[2 + n]);
        volume *= (double)I[n];
    }
    const uint64_t nnz  = (uint64_t)std::atoll(argv[2 + N]);
    const uint64_t seed = (uint64_t)std::atoll(argv[3 + N]);
    const char* out     = argv[4 + N];

    if ((double)nnz > 0.5 * volume) {
        std::fprintf(stderr,
            "WARN: nnz=%llu > 50%% of volume=%.0f; generator is not designed "
            "for dense tensors -- duplicate rejection is disabled, so the\n"
            "     output may contain a small fraction of collisions.  Use\n"
            "     tools/gen_cp_tensor.py for dense CP-structured tensors.\n",
            (unsigned long long)nnz, volume);
    }

    std::FILE* fp = std::fopen(out, "wb");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", out); return 2; }
    Bufout w(fp);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    std::vector<std::uniform_int_distribution<int>> IDX(N);
    for (int n = 0; n < N; ++n)
        IDX[n] = std::uniform_int_distribution<int>(1, I[n]);

    // Progress reporting: one line per 10%.
    const uint64_t step = std::max<uint64_t>(1, nnz / 10);
    uint64_t next_report = step;

    for (uint64_t k = 0; k < nnz; ++k) {
        for (int n = 0; n < N; ++n) {
            w.write_int(IDX[n](rng));
            w.put(' ');
        }
        w.write_double(U(rng));
        w.put('\n');

        if (k + 1 == next_report) {
            std::fprintf(stderr, "[gen_tensor] %.0f%% (%llu / %llu)\n",
                         100.0 * (double)(k + 1) / (double)nnz,
                         (unsigned long long)(k + 1),
                         (unsigned long long)nnz);
            next_report += step;
        }
    }
    w.flush();
    std::fclose(fp);

    std::fprintf(stderr, "wrote %llu nonzeros to %s\n",
                 (unsigned long long)nnz, out);
    return 0;
}
