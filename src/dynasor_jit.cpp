// ============================================================================
//  dynasor_jit.cpp -- shell-out C++ JIT for the fiber_csr kernel.
//
//  Concept:
//    1. The baked binary ships `process_range_fiber_csr_T<N,Rp,TGT>`
//       specializations across a fixed grid.  Good enough for most
//       tensors, but inert for odd shapes and for the runtime pf_far
//       toggle (which stays as a LICM-hoisted load even when OFF).
//    2. On first request for (N, Rp, TGT, pf_far) the JIT emits a
//       ~120-line stub C++ file that hard-codes those four constants,
//       #includes dynasor_simd.h, and compiles to a shared library
//       via the same $(CXX) used to build the driver.
//    3. The shared library is loaded with dlopen / LoadLibraryA and its
//       `extern "C"` entry symbol is cached in a small hash table
//       keyed on the tuple.  Misses hit the disk cache first
//       (.dyn_jit_cache/<key>.{dll,so}); only first-ever compilations
//       pay the 2-4 s shell-out cost.
//    4. Any failure (CXX not found, compile error, dlopen error)
//       degrades to returning nullptr; the driver falls back to the
//       baked template path.
//
//  Correctness guarantee:
//    The JIT stub is an exact textual specialization of the template
//    body in dynasor_kernel.cpp.  Drift is prevented by keeping the
//    stub text alongside its original in this file -- any future edit
//    to the kernel MUST also update the string literal below
//    (jit_kernel_body).  The test harness (--verify with --jit) catches
//    drift: if the JIT output diverges numerically from the baked
//    kernel by more than 1e-5 relative Frobenius, the verify fails.
// ============================================================================
#include "dynasor_jit.h"
#include "dynasor_simd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <direct.h>
  #include <io.h>
  #include <sys/stat.h>
  #define DYN_JIT_DLLEXT ".dll"
  #define DYN_JIT_PATHSEP '\\'
#else
  #include <dlfcn.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
  #define DYN_JIT_DLLEXT ".so"
  #define DYN_JIT_PATHSEP '/'
#endif

namespace dynasor {

// ---------------------------------------------------------------------------
//  Env + force flag.
// ---------------------------------------------------------------------------
static std::mutex  g_jit_mu;
static int         g_jit_force = -1;   // -1 = env, 0 = off, 1 = on

void dyn_jit_force_enable(bool on) {
    std::lock_guard<std::mutex> lk(g_jit_mu);
    g_jit_force = on ? 1 : 0;
}

bool dyn_jit_enabled() {
    static const bool from_env = [] {
        const char* s = std::getenv("DYN_JIT");
        return s && s[0] && s[0] != '0';
    }();
    std::lock_guard<std::mutex> lk(g_jit_mu);
    if (g_jit_force >= 0) return g_jit_force != 0;
    return from_env;
}

// ---------------------------------------------------------------------------
//  Stats.
// ---------------------------------------------------------------------------
static JitStats g_stats;
JitStats dyn_jit_stats() {
    std::lock_guard<std::mutex> lk(g_jit_mu);
    return g_stats;
}

// ---------------------------------------------------------------------------
//  Cache directory.  Created on first need next to the driver binary, or
//  overridden via DYN_JIT_CACHE_DIR.
// ---------------------------------------------------------------------------
static std::string jit_cache_dir() {
    const char* env = std::getenv("DYN_JIT_CACHE_DIR");
    std::string d = (env && *env) ? env : ".dyn_jit_cache";
#if defined(_WIN32)
    _mkdir(d.c_str());
#else
    ::mkdir(d.c_str(), 0755);
#endif
    return d;
}

// ---------------------------------------------------------------------------
//  ISA token -- matches the baked binary's DYN_SIMD_NAME but slugified
//  (no spaces / punctuation) so it is safe to embed in a filename.
// ---------------------------------------------------------------------------
static std::string isa_slug() {
    std::string s = DYN_SIMD_NAME;
    for (char& c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9')))
            c = '_';
    return s;
}

// Toolchain token.  We don't need full determinism -- a stable short
// signature of `$(CXX) --version` is enough to invalidate the cache on
// GCC upgrades.
static std::string cxx_tag() {
    const char* env = std::getenv("DYN_JIT_CXX");
    std::string cxx = (env && *env) ? env : "g++";
    std::string cmd = cxx + " -dumpfullversion -dumpversion 2>&1";
#if defined(_WIN32)
    FILE* p = _popen(cmd.c_str(), "r");
#else
    FILE* p = ::popen(cmd.c_str(), "r");
#endif
    std::string ver;
    if (p) {
        char buf[256];
        while (std::fgets(buf, sizeof(buf), p)) ver += buf;
#if defined(_WIN32)
        _pclose(p);
#else
        ::pclose(p);
#endif
    }
    // Trim whitespace; reduce to first line only.
    for (char& c : ver) if (c == '\r' || c == '\n' || c == ' ') c = '_';
    if (ver.empty()) ver = "unknown";
    return ver.substr(0, 32);
}

// ---------------------------------------------------------------------------
//  Cache key + filename.
// ---------------------------------------------------------------------------
struct JitKey {
    int  N;
    int  Rp;
    int  TGT;
    bool pf_far;
    bool operator==(const JitKey& o) const {
        return N == o.N && Rp == o.Rp && TGT == o.TGT && pf_far == o.pf_far;
    }
};
struct JitKeyHash {
    size_t operator()(const JitKey& k) const {
        return ((size_t)k.N * 131 + (size_t)k.Rp) * 131 +
               ((size_t)k.TGT * 2 + (size_t)k.pf_far);
    }
};

static std::string key_stem(const JitKey& k) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "fiber_csr__N%d_Rp%d_TGT%d_pf%d__%s__%s",
                  k.N, k.Rp, k.TGT, (int)k.pf_far,
                  isa_slug().c_str(),
                  cxx_tag().c_str());
    return buf;
}

// ---------------------------------------------------------------------------
//  The JIT stub source.  Hard-codes N_TMPL, RP_TMPL, TGT_TMPL, PF_FAR
//  as constexpr constants.  The body is a near-textual copy of
//  process_range_fiber_csr_T from dynasor_kernel.cpp, with the runtime
//  branches on `pf_far_on` replaced by `if constexpr (kPfFar)` so the
//  compiler DCE's the far-prefetch block entirely when OFF.  The entry
//  symbol is `extern "C"` to avoid C++ name mangling.
// ---------------------------------------------------------------------------
static std::string make_jit_source(const JitKey& k) {
    std::string s;
    s.reserve(8 * 1024);

    auto add = [&](const char* t) { s += t; };
    char ln[512];

    add("// AUTO-GENERATED by dyn_jit_get_fiber_csr(). DO NOT EDIT.\n");
    add("#include \"dynasor_common.h\"\n");
    add("#include \"dynasor_simd.h\"\n");
    add("#include <cstdint>\n");
    add("\n");
    add("using namespace dynasor;\n");
    add("\n");
    std::snprintf(ln, sizeof(ln),
                  "static constexpr int  kN      = %d;\n"
                  "static constexpr int  kRp     = %d;\n"
                  "static constexpr int  kTGT    = %d;\n"
                  "static constexpr bool kPfFar  = %s;\n"
                  "static constexpr int  kPfDist      = 4;\n"
                  "static constexpr int  kPfDistFar   = 16;\n"
                  "static constexpr int  kMaxVecChunks = 64;\n"
                  "static constexpr int  kSIMD   = DYN_SIMD_WIDTH;\n"
                  "static constexpr int  kVC    = kRp / kSIMD;\n"
                  "static constexpr int  kW0    = (kTGT == 0) ? 1 : 0;\n",
                  k.N, k.Rp, k.TGT, k.pf_far ? "true" : "false");
    add(ln);
    add("\n");
    // w_last as a compile-time constexpr.
    add("static constexpr int compute_w_last() {\n"
        "    int w_last = -1;\n"
        "    for (int w = kN - 1; w >= 0; --w) {\n"
        "        if (w != kTGT && w != kW0) { w_last = w; break; }\n"
        "    }\n"
        "    return w_last;\n"
        "}\n"
        "static constexpr int kWlast = compute_w_last();\n"
        "static constexpr int kPfLines = (kRp * (int)sizeof(value_t) + 63) / 64;\n"
        "\n");

    // The kernel body itself.  Mirror of process_range_fiber_csr_T with
    // every template parameter replaced by its constexpr counterpart.
    add(R"(extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void dyn_jit_fiber_csr(
    const value_t* DYN_RESTRICT vals,
    const idx_t*   const*       idx,
    const uint64_t* DYN_RESTRICT rowptr,
    idx_t row_b, idx_t row_e,
    const value_t* const* DYN_RESTRICT Y,
    value_t* DYN_RESTRICT Yhat_n)
{
    constexpr int N  = kN;
    constexpr int Rp = kRp;
    constexpr int VC = kVC;
    constexpr int n  = kTGT;
    constexpr int w0 = kW0;
    constexpr int w_last = kWlast;
    constexpr int pf_lines = kPfLines;
    const uint64_t e = rowptr[row_e];

    for (idx_t row = row_b; row < row_e; ++row) {
        const uint64_t i_beg = rowptr[row];
        const uint64_t i_end = rowptr[row + 1];
        if (i_beg == i_end) continue;

        if constexpr (kPfFar) {
            if (row + 1 < row_e) {
                value_t* const pf_next =
                    Yhat_n + (size_t)(row + 1) * (size_t)Rp;
                for (int k = 0; k < pf_lines; ++k)
                    dyn_prefetch_l2(pf_next + k * (64 / (int)sizeof(value_t)));
            }
        }

        value_t* const outRow = Yhat_n + (size_t)row * Rp;
        dyn_vec_t acc[kMaxVecChunks];
        for (int c = 0; c < VC; ++c)
            acc[c] = dyn_vload(outRow + c * DYN_SIMD_WIDTH);

        for (uint64_t k = i_beg; k < i_end; ++k) {
            if (DYN_LIKELY(k + kPfDist < e)) {
                const uint64_t pk = k + kPfDist;
                for (int w = 0; w < N; ++w) {
                    if (w == n) continue;
                    const value_t* pf = Y[w] + (size_t)idx[w][pk] * Rp;
                    for (int l = 0; l < pf_lines; ++l)
                        dyn_prefetch(pf + l * (64 / (int)sizeof(value_t)));
                }
            }
            if constexpr (kPfFar) {
                if (DYN_LIKELY(k + kPfDistFar < e)) {
                    const uint64_t pk = k + kPfDistFar;
                    for (int w = 0; w < N; ++w) {
                        if (w == n) continue;
                        const value_t* pf = Y[w] + (size_t)idx[w][pk] * Rp;
                        for (int l = 0; l < pf_lines; ++l)
                            dyn_prefetch_l2(pf + l * (64 / (int)sizeof(value_t)));
                    }
                }
            }

            const value_t   val  = vals[k];
            const dyn_vec_t vval = dyn_vset1(val);

            dyn_vec_t prod[kMaxVecChunks];
            {
                const value_t* rowY = Y[w0] + (size_t)idx[w0][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rowY + c * DYN_SIMD_WIDTH);
                    prod[c] = dyn_vmul(vval, y);
                }
            }
            for (int w = 0; w < N; ++w) {
                if (w == n || w == w0 || w == w_last) continue;
                const value_t* rY = Y[w] + (size_t)idx[w][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                    prod[c] = dyn_vmul(prod[c], y);
                }
            }

            if constexpr (w_last >= 0) {
                const value_t* rY = Y[w_last] + (size_t)idx[w_last][k] * Rp;
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rY + c * DYN_SIMD_WIDTH);
                    acc[c] = dyn_vfma(prod[c], y, acc[c]);
                }
            } else {
                for (int c = 0; c < VC; ++c)
                    acc[c] = dyn_vadd(acc[c], prod[c]);
            }
        }

        for (int c = 0; c < VC; ++c)
            dyn_vstore(outRow + c * DYN_SIMD_WIDTH, acc[c]);
    }
}
)");

    return s;
}

// ---------------------------------------------------------------------------
//  File helpers.
// ---------------------------------------------------------------------------
static bool file_exists(const std::string& path) {
#if defined(_WIN32)
    return _access(path.c_str(), 0) == 0;
#else
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
#endif
}

static bool write_all(const std::string& path, const std::string& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const size_t w = std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
    return w == data.size();
}

// ---------------------------------------------------------------------------
//  Shell-out compile.  Returns true iff the shared library was
//  produced.  stdout/stderr of the compiler is captured to a .log file
//  next to the .cpp (useful for post-mortem).
// ---------------------------------------------------------------------------
static bool compile_to_dll(const std::string& cpp_path,
                           const std::string& dll_path,
                           const std::string& include_dir)
{
    const char* cxx_env = std::getenv("DYN_JIT_CXX");
    const std::string cxx = (cxx_env && *cxx_env) ? cxx_env : "g++";

    const char* extra_env = std::getenv("DYN_JIT_CXXFLAGS");
    const std::string extra = (extra_env && *extra_env) ? extra_env : "";

    // -march=native is the safe default: the JIT runs on the machine that
    // will execute the kernel, so the JIT build target matches the host.
    // -O3 -funroll-loops mirrors the main Makefile; LTO is intentionally
    // NOT enabled (per-kernel LTO gives ~0% win here and doubles compile
    // time).  -fno-semantic-interposition keeps the compiler free to
    // inline across the extern "C" boundary even with -fPIC on Linux.
    //
    // On MinGW/Windows g++ needs -shared with no -fPIC.  On POSIX we add
    // -fPIC explicitly so -shared produces a real PIC .so.  Both branches
    // pass the include search path for dynasor_common.h /_simd.h and the
    // same -fopenmp the binary was built with (the JIT kernel itself has
    // no #pragma omp, but #include <omp.h> chains via the headers on
    // some configurations).
    std::string flags =
        "-O3 -std=c++17 -funroll-loops -fopenmp "
#if !defined(_WIN32)
        "-fPIC "
#endif
        "-march=native -mtune=native ";
    if (!extra.empty()) { flags += extra; flags += ' '; }

    // Quote include path so Windows paths with spaces work.
    std::string cmd = cxx;
    cmd += ' ';
    cmd += flags;
    cmd += " -I\"";
    cmd += include_dir;
    cmd += "\" -shared \"";
    cmd += cpp_path;
    cmd += "\" -o \"";
    cmd += dll_path;
    cmd += "\" 2> \"";
    cmd += cpp_path;
    cmd += ".log\"";

    const int rc = std::system(cmd.c_str());
    return rc == 0 && file_exists(dll_path);
}

// ---------------------------------------------------------------------------
//  Dynamic loader abstraction.
// ---------------------------------------------------------------------------
static void* jit_dlopen(const std::string& path) {
#if defined(_WIN32)
    return (void*)LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void* jit_dlsym(void* h, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)h, name);
#else
    return dlsym(h, name);
#endif
}

// ---------------------------------------------------------------------------
//  Resolver.  Single global cache; inserts are serialized on the mutex,
//  reads after the first hit are lock-free through the pointer cache.
// ---------------------------------------------------------------------------
static std::unordered_map<JitKey, jit_fiber_csr_fn, JitKeyHash> g_cache;

jit_fiber_csr_fn dyn_jit_get_fiber_csr(int N, int Rp, int TGT, bool pf_far) {
    if (!dyn_jit_enabled()) return nullptr;

    // Only specialize for the same grid the baked kernel covers
    // productively.  Outside that, the JIT still works but offers no
    // advantage AND makes compile times user-visible; silently return
    // nullptr so the baked runtime-fallback path takes over.
    if (N < 2 || N > 8) return nullptr;
    if (Rp <= 0 || (Rp % 4) != 0) return nullptr;
    if (TGT < 0 || TGT >= N) return nullptr;

    const JitKey key{N, Rp, TGT, pf_far};

    {
        std::lock_guard<std::mutex> lk(g_jit_mu);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) return it->second;
    }

    // Slow path: build or load the .so/.dll.  We do this inside the
    // mutex so duplicate concurrent requests don't race on the same
    // cache file.
    std::lock_guard<std::mutex> lk(g_jit_mu);

    // Someone else may have populated while we were waiting.
    if (auto it = g_cache.find(key); it != g_cache.end()) return it->second;

    const std::string dir   = jit_cache_dir();
    const std::string stem  = key_stem(key);
    const std::string cpp   = dir + DYN_JIT_PATHSEP + stem + ".cpp";
    const std::string dll   = dir + DYN_JIT_PATHSEP + stem + DYN_JIT_DLLEXT;

    const char* inc_env = std::getenv("DYN_JIT_INCLUDE");
    const std::string inc = (inc_env && *inc_env) ? inc_env : "include";

    jit_fiber_csr_fn fn = nullptr;
    bool from_cache = false;

    if (!file_exists(dll)) {
        const std::string src = make_jit_source(key);
        if (!write_all(cpp, src)) {
            g_stats.failures++;
            g_cache[key] = nullptr;
            return nullptr;
        }
        if (!compile_to_dll(cpp, dll, inc)) {
            std::fprintf(stderr,
                "[dynasor][jit] compile failed for %s (see %s.log); "
                "falling back to baked kernel\n",
                stem.c_str(), cpp.c_str());
            g_stats.failures++;
            g_cache[key] = nullptr;
            return nullptr;
        }
        g_stats.compiled++;
    } else {
        from_cache = true;
        g_stats.cache_hits++;
    }

    void* h = jit_dlopen(dll);
    if (!h) {
        std::fprintf(stderr,
            "[dynasor][jit] dlopen('%s') failed; falling back to baked kernel\n",
            dll.c_str());
        g_stats.failures++;
        g_cache[key] = nullptr;
        return nullptr;
    }
    fn = reinterpret_cast<jit_fiber_csr_fn>(
            jit_dlsym(h, "dyn_jit_fiber_csr"));
    if (!fn) {
        std::fprintf(stderr,
            "[dynasor][jit] symbol 'dyn_jit_fiber_csr' missing in %s; "
            "falling back to baked kernel\n",
            dll.c_str());
        g_stats.failures++;
        g_cache[key] = nullptr;
        return nullptr;
    }

    std::printf("[dynasor][jit] %s (N=%d Rp=%d TGT=%d pf_far=%d) [%s]\n",
                from_cache ? "cache-hit" : "compiled",
                key.N, key.Rp, key.TGT, (int)key.pf_far,
                dll.c_str());

    g_cache[key] = fn;
    return fn;
}

// ===========================================================================
//  Stage 4 -- JIT'd all-modes Morton kernel.
//
//  Bakes in (N, Rp, masks[0..N-1]) as constexpr constants so the inner
//  _pext_u64 calls compile to immediate-mask BMI2 ops and the per-mode
//  loop inside the kernel fully unrolls.  Structurally mirrors
//  dyn_jit_get_fiber_csr above: same cache dir, same toolchain tag,
//  same compile-and-dlopen shell-out.
// ===========================================================================

struct JitAmMortonKey {
    int      N;
    int      Rp;
    uint64_t mask[DYN_MAX_MODES];
    bool operator==(const JitAmMortonKey& o) const {
        if (N != o.N || Rp != o.Rp) return false;
        for (int i = 0; i < N; ++i)
            if (mask[i] != o.mask[i]) return false;
        return true;
    }
};
struct JitAmMortonKeyHash {
    size_t operator()(const JitAmMortonKey& k) const {
        size_t h = (size_t)k.N * 131 + (size_t)k.Rp;
        for (int i = 0; i < k.N; ++i) {
            h = h * 131 + (size_t)(k.mask[i] ^ (k.mask[i] >> 32));
        }
        return h;
    }
};

static std::string key_stem_am_morton(const JitAmMortonKey& k) {
    // Short cryptographic-ish digest of the masks so the filename stays
    // under any filesystem's path-length limit even for N=8 masks.
    uint64_t digest = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < k.N; ++i) {
        digest ^= k.mask[i] + 0x9E3779B97F4A7C15ULL
                + (digest << 6) + (digest >> 2);
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "am_morton__N%d_Rp%d_m%016llx__%s__%s",
                  k.N, k.Rp, (unsigned long long)digest,
                  isa_slug().c_str(), cxx_tag().c_str());
    return buf;
}

static std::string make_jit_source_am_morton(const JitAmMortonKey& k) {
    std::string s;
    s.reserve(12 * 1024);
    auto add = [&](const char* t) { s += t; };
    char ln[512];

    add("// AUTO-GENERATED by dyn_jit_get_all_modes_morton(). DO NOT EDIT.\n");
    add("#include \"dynasor_common.h\"\n");
    add("#include \"dynasor_simd.h\"\n");
    add("#include <cstdint>\n");
    add("#include <cstdlib>\n");
    add("#if defined(__BMI2__)\n#include <immintrin.h>\n#endif\n");
    add("\n");
    add("using namespace dynasor;\n");
    add("\n");
    std::snprintf(ln, sizeof(ln),
                  "static constexpr int  kN   = %d;\n"
                  "static constexpr int  kRp  = %d;\n"
                  "static constexpr int  kVC  = kRp / DYN_SIMD_WIDTH;\n"
                  "static constexpr int  kPfLines = (kRp * (int)sizeof(value_t) + 63) / 64;\n"
                  "static constexpr int  kPfDist    = 4;\n"
                  "static constexpr int  kPfDistFar = 16;\n"
                  "static constexpr int  kMaxVecChunks = 64;\n",
                  k.N, k.Rp);
    add(ln);
    // Emit the masks as a static constexpr array indexed 0..N-1.
    add("static constexpr uint64_t kMask[kN] = {");
    for (int i = 0; i < k.N; ++i) {
        std::snprintf(ln, sizeof(ln),
                      "%s0x%016llxULL",
                      i ? ", " : "",
                      (unsigned long long)k.mask[i]);
        add(ln);
    }
    add("};\n\n");

    // Inline pext shim -- BMI2 fast path + scalar fallback.  The fallback
    // only activates when the JIT binary is targeted at a non-BMI2 ISA;
    // in practice the JIT compile uses -march=native so BMI2 is taken.
    add("static inline uint64_t m_pext(uint64_t x, uint64_t mask) {\n"
        "#if defined(__BMI2__)\n"
        "    return _pext_u64(x, mask);\n"
        "#else\n"
        "    uint64_t r = 0, b = 1;\n"
        "    while (mask) {\n"
        "        const uint64_t lo = mask & (uint64_t)(-(int64_t)mask);\n"
        "        if (x & lo) r |= b;\n"
        "        mask ^= lo; b <<= 1;\n"
        "    }\n"
        "    return r;\n"
        "#endif\n"
        "}\n\n");

    // Kernel body.  Structurally identical to the N=3 / N=4 / general
    // branches of process_range_all_modes_morton_T in dynasor_morton.cpp,
    // but with the kMask[] lookups folded to compile-time constants.
    add(R"(extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void dyn_jit_all_modes_morton(
    const value_t*         mvals,
    const uint64_t*        mkeys,
    uint64_t b, uint64_t e,
    const value_t* const*  Y,
    value_t* const*        Yout)
{
    constexpr int N  = kN;
    constexpr int Rp = kRp;
    constexpr int VC = kVC;
    constexpr int pf_lines = kPfLines;

    const bool pf_far_on = [] {
        const char* s = std::getenv("DYN_PF_FAR");
        return s && s[0] && s[0] != '0';
    }();

    for (uint64_t i = b; i < e; ++i) {
        if (DYN_LIKELY(i + kPfDist < e)) {
            const uint64_t kpf = mkeys[i + kPfDist];
            for (int w = 0; w < N; ++w) {
                const idx_t r = (idx_t)m_pext(kpf, kMask[w]);
                const value_t* pf_in = Y[w]    + (size_t)r * Rp;
                const value_t* pf_ot = Yout[w] + (size_t)r * Rp;
                for (int k = 0; k < pf_lines; ++k) {
                    dyn_prefetch(pf_in + k * (64 / (int)sizeof(value_t)));
                    dyn_prefetch(pf_ot + k * (64 / (int)sizeof(value_t)));
                }
            }
        }
        if (pf_far_on && DYN_LIKELY(i + kPfDistFar < e)) {
            const uint64_t kpf = mkeys[i + kPfDistFar];
            for (int w = 0; w < N; ++w) {
                const idx_t r = (idx_t)m_pext(kpf, kMask[w]);
                const value_t* pf_in = Y[w]    + (size_t)r * Rp;
                const value_t* pf_ot = Yout[w] + (size_t)r * Rp;
                for (int k = 0; k < pf_lines; ++k) {
                    dyn_prefetch_l2(pf_in + k * (64 / (int)sizeof(value_t)));
                    dyn_prefetch_l2(pf_ot + k * (64 / (int)sizeof(value_t)));
                }
            }
        }

        const uint64_t key = mkeys[i];
        idx_t idx[DYN_MAX_MODES];
        for (int w = 0; w < N; ++w) idx[w] = (idx_t)m_pext(key, kMask[w]);

        const value_t   val  = mvals[i];
        const dyn_vec_t vval = dyn_vset1(val);

        if constexpr (N == 3) {
            const value_t* DYN_RESTRICT rY0 = Y[0] + (size_t)idx[0] * Rp;
            const value_t* DYN_RESTRICT rY1 = Y[1] + (size_t)idx[1] * Rp;
            const value_t* DYN_RESTRICT rY2 = Y[2] + (size_t)idx[2] * Rp;
            value_t* DYN_RESTRICT oY0 = Yout[0] + (size_t)idx[0] * Rp;
            value_t* DYN_RESTRICT oY1 = Yout[1] + (size_t)idx[1] * Rp;
            value_t* DYN_RESTRICT oY2 = Yout[2] + (size_t)idx[2] * Rp;
            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t y0 = dyn_vload(rY0 + off);
                dyn_vec_t y1 = dyn_vload(rY1 + off);
                dyn_vec_t y2 = dyn_vload(rY2 + off);
                dyn_vec_t vy0 = dyn_vmul(vval, y0);
                dyn_vec_t vy1 = dyn_vmul(vval, y1);
                dyn_vec_t o0 = dyn_vload(oY0 + off);
                o0 = dyn_vfma(vy1, y2, o0);
                dyn_vstore(oY0 + off, o0);
                dyn_vec_t o1 = dyn_vload(oY1 + off);
                o1 = dyn_vfma(vy0, y2, o1);
                dyn_vstore(oY1 + off, o1);
                dyn_vec_t o2 = dyn_vload(oY2 + off);
                o2 = dyn_vfma(vy0, y1, o2);
                dyn_vstore(oY2 + off, o2);
            }
            continue;
        }

        if constexpr (N == 4) {
            const value_t* DYN_RESTRICT rY0 = Y[0] + (size_t)idx[0] * Rp;
            const value_t* DYN_RESTRICT rY1 = Y[1] + (size_t)idx[1] * Rp;
            const value_t* DYN_RESTRICT rY2 = Y[2] + (size_t)idx[2] * Rp;
            const value_t* DYN_RESTRICT rY3 = Y[3] + (size_t)idx[3] * Rp;
            value_t* DYN_RESTRICT oY0 = Yout[0] + (size_t)idx[0] * Rp;
            value_t* DYN_RESTRICT oY1 = Yout[1] + (size_t)idx[1] * Rp;
            value_t* DYN_RESTRICT oY2 = Yout[2] + (size_t)idx[2] * Rp;
            value_t* DYN_RESTRICT oY3 = Yout[3] + (size_t)idx[3] * Rp;
            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t y0 = dyn_vload(rY0 + off);
                dyn_vec_t y1 = dyn_vload(rY1 + off);
                dyn_vec_t y2 = dyn_vload(rY2 + off);
                dyn_vec_t y3 = dyn_vload(rY3 + off);
                dyn_vec_t p1 = dyn_vmul(vval, y0);
                dyn_vec_t p2 = dyn_vmul(p1, y1);
                dyn_vec_t p3 = dyn_vmul(p2, y2);
                dyn_vec_t s1 = dyn_vmul(y2, y3);
                dyn_vec_t s0 = dyn_vmul(y1, s1);
                dyn_vec_t o0 = dyn_vload(oY0 + off);
                o0 = dyn_vfma(vval, s0, o0);
                dyn_vstore(oY0 + off, o0);
                dyn_vec_t o1 = dyn_vload(oY1 + off);
                o1 = dyn_vfma(p1, s1, o1);
                dyn_vstore(oY1 + off, o1);
                dyn_vec_t o2 = dyn_vload(oY2 + off);
                o2 = dyn_vfma(p2, y3, o2);
                dyn_vstore(oY2 + off, o2);
                dyn_vec_t o3 = dyn_vload(oY3 + off);
                o3 = dyn_vadd(o3, p3);
                dyn_vstore(oY3 + off, o3);
            }
            continue;
        }

        // General-N prefix-suffix fallback.
        dyn_vec_t pref[DYN_MAX_MODES][kMaxVecChunks];
        const value_t* rPtr[DYN_MAX_MODES];
        value_t*       oPtr[DYN_MAX_MODES];
        for (int w = 0; w < N; ++w) {
            rPtr[w] = Y[w]    + (size_t)idx[w] * Rp;
            oPtr[w] = Yout[w] + (size_t)idx[w] * Rp;
        }
        for (int c = 0; c < VC; ++c) pref[0][c] = vval;
        for (int k = 1; k < N; ++k) {
            for (int c = 0; c < VC; ++c) {
                dyn_vec_t y = dyn_vload(rPtr[k - 1] + c * DYN_SIMD_WIDTH);
                pref[k][c] = dyn_vmul(pref[k - 1][c], y);
            }
        }
        dyn_vec_t suf[kMaxVecChunks];
        for (int c = 0; c < VC; ++c) suf[c] = dyn_vset1(1.0f);
        for (int k = N - 1; k >= 0; --k) {
            for (int c = 0; c < VC; ++c) {
                const int off = c * DYN_SIMD_WIDTH;
                dyn_vec_t o = dyn_vload(oPtr[k] + off);
                o = dyn_vfma(pref[k][c], suf[c], o);
                dyn_vstore(oPtr[k] + off, o);
            }
            if (k > 0) {
                for (int c = 0; c < VC; ++c) {
                    dyn_vec_t y = dyn_vload(rPtr[k] + c * DYN_SIMD_WIDTH);
                    suf[c] = dyn_vmul(suf[c], y);
                }
            }
        }
    }
}
)");

    return s;
}

static std::unordered_map<JitAmMortonKey, jit_all_modes_morton_fn,
                          JitAmMortonKeyHash> g_am_morton_cache;

jit_all_modes_morton_fn dyn_jit_get_all_modes_morton(
    int N, int Rp, const uint64_t* masks)
{
    if (!dyn_jit_enabled()) return nullptr;
    if (N < 2 || N > DYN_MAX_MODES) return nullptr;
    if (Rp <= 0 || (Rp % DYN_SIMD_WIDTH) != 0) return nullptr;

    JitAmMortonKey key{};
    key.N = N;
    key.Rp = Rp;
    for (int i = 0; i < N; ++i) key.mask[i] = masks[i];

    {
        std::lock_guard<std::mutex> lk(g_jit_mu);
        auto it = g_am_morton_cache.find(key);
        if (it != g_am_morton_cache.end()) return it->second;
    }

    std::lock_guard<std::mutex> lk(g_jit_mu);
    if (auto it = g_am_morton_cache.find(key); it != g_am_morton_cache.end())
        return it->second;

    const std::string dir  = jit_cache_dir();
    const std::string stem = key_stem_am_morton(key);
    const std::string cpp  = dir + DYN_JIT_PATHSEP + stem + ".cpp";
    const std::string dll  = dir + DYN_JIT_PATHSEP + stem + DYN_JIT_DLLEXT;

    const char* inc_env = std::getenv("DYN_JIT_INCLUDE");
    const std::string inc = (inc_env && *inc_env) ? inc_env : "include";

    bool from_cache = false;
    if (!file_exists(dll)) {
        const std::string src = make_jit_source_am_morton(key);
        if (!write_all(cpp, src)) {
            g_stats.failures++;
            g_am_morton_cache[key] = nullptr;
            return nullptr;
        }
        if (!compile_to_dll(cpp, dll, inc)) {
            std::fprintf(stderr,
                "[dynasor][jit] am_morton compile failed for %s (see %s.log); "
                "falling back to baked kernel\n",
                stem.c_str(), cpp.c_str());
            g_stats.failures++;
            g_am_morton_cache[key] = nullptr;
            return nullptr;
        }
        g_stats.compiled++;
    } else {
        from_cache = true;
        g_stats.cache_hits++;
    }

    void* h = jit_dlopen(dll);
    if (!h) {
        std::fprintf(stderr,
            "[dynasor][jit] am_morton dlopen('%s') failed; baked fallback\n",
            dll.c_str());
        g_stats.failures++;
        g_am_morton_cache[key] = nullptr;
        return nullptr;
    }
    auto fn = reinterpret_cast<jit_all_modes_morton_fn>(
        jit_dlsym(h, "dyn_jit_all_modes_morton"));
    if (!fn) {
        std::fprintf(stderr,
            "[dynasor][jit] symbol 'dyn_jit_all_modes_morton' missing in %s\n",
            dll.c_str());
        g_stats.failures++;
        g_am_morton_cache[key] = nullptr;
        return nullptr;
    }

    std::printf("[dynasor][jit] am_morton %s (N=%d Rp=%d) [%s]\n",
                from_cache ? "cache-hit" : "compiled",
                key.N, key.Rp, dll.c_str());

    g_am_morton_cache[key] = fn;
    return fn;
}

} // namespace dynasor
