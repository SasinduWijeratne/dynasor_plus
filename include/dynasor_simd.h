// ============================================================================
//  Portable SIMD wrappers for Dynasor+.
//
//  Detects the best available ISA at compile time and defines:
//
//      DYN_SIMD_WIDTH      : lane count of a float vector
//      DYN_SIMD_NAME       : textual back-end name
//      dyn_vec_t           : SIMD register type
//      dyn_vset1(v)        : broadcast scalar -> vector
//      dyn_vzero()         : zero vector
//      dyn_vload (p)       : aligned load  (p must be DYN_ALIGN-aligned)
//      dyn_vloadu(p)       : unaligned load
//      dyn_vstore (p,x)    : aligned store
//      dyn_vstoreu(p,x)    : unaligned store
//      dyn_vmul(a,b)       : elementwise multiply
//      dyn_vfma(a,b,c)     : a*b + c (fused if available)
//
//  Supported back-ends (in order of preference):
//      AVX-512F  (x86, 16 lanes)
//      AVX2+FMA  (x86,  8 lanes)
//      SSE2      (x86,  4 lanes)
//      NEON      (ARMv7/AArch64, 4 lanes)
//      SVE       (AArch64, VLA -- currently compiled but used through the
//                 generic chunk loop with a 512-bit logical lane count)
//      scalar    (4-wide emulated for portability)
// ============================================================================
#ifndef DYNASOR_SIMD_H
#define DYNASOR_SIMD_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------- alignment macro ----------
#ifndef DYN_ALIGN
#define DYN_ALIGN 64
#endif

#if defined(_MSC_VER)
  #define DYN_ALIGNED(x) __declspec(align(x))
  #define DYN_RESTRICT   __restrict
#else
  #define DYN_ALIGNED(x) __attribute__((aligned(x)))
  #define DYN_RESTRICT   __restrict__
#endif

// ---------- ISA selection ----------
#if defined(__AVX512F__)
  #define DYN_HAS_AVX512 1
#endif
#if defined(__AVX2__)
  #define DYN_HAS_AVX2 1
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  #define DYN_HAS_SSE2 1
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
  #define DYN_HAS_NEON 1
#endif

// ============================================================================
//  AVX-512 back-end  -- 16 x float
// ============================================================================
#if defined(DYN_HAS_AVX512)
#include <immintrin.h>

#define DYN_SIMD_WIDTH 16
#define DYN_SIMD_NAME  "AVX-512F"
using dyn_vec_t = __m512;

static inline dyn_vec_t dyn_vzero()                      { return _mm512_setzero_ps(); }
static inline dyn_vec_t dyn_vset1(float v)               { return _mm512_set1_ps(v); }
static inline dyn_vec_t dyn_vload (const float* p)       { return _mm512_load_ps(p); }
static inline dyn_vec_t dyn_vloadu(const float* p)       { return _mm512_loadu_ps(p); }
static inline void      dyn_vstore (float* p, dyn_vec_t x){ _mm512_store_ps(p, x); }
static inline void      dyn_vstoreu(float* p, dyn_vec_t x){ _mm512_storeu_ps(p, x); }
static inline dyn_vec_t dyn_vmul(dyn_vec_t a, dyn_vec_t b){ return _mm512_mul_ps(a, b); }
static inline dyn_vec_t dyn_vadd(dyn_vec_t a, dyn_vec_t b){ return _mm512_add_ps(a, b); }
static inline dyn_vec_t dyn_vfma(dyn_vec_t a, dyn_vec_t b, dyn_vec_t c){
    return _mm512_fmadd_ps(a, b, c);
}

// ============================================================================
//  AVX2 + FMA back-end  -- 8 x float
// ============================================================================
#elif defined(DYN_HAS_AVX2)
#include <immintrin.h>

#define DYN_SIMD_WIDTH 8
#define DYN_SIMD_NAME  "AVX2+FMA"
using dyn_vec_t = __m256;

static inline dyn_vec_t dyn_vzero()                      { return _mm256_setzero_ps(); }
static inline dyn_vec_t dyn_vset1(float v)               { return _mm256_set1_ps(v); }
static inline dyn_vec_t dyn_vload (const float* p)       { return _mm256_load_ps(p); }
static inline dyn_vec_t dyn_vloadu(const float* p)       { return _mm256_loadu_ps(p); }
static inline void      dyn_vstore (float* p, dyn_vec_t x){ _mm256_store_ps(p, x); }
static inline void      dyn_vstoreu(float* p, dyn_vec_t x){ _mm256_storeu_ps(p, x); }
static inline dyn_vec_t dyn_vmul(dyn_vec_t a, dyn_vec_t b){ return _mm256_mul_ps(a, b); }
static inline dyn_vec_t dyn_vadd(dyn_vec_t a, dyn_vec_t b){ return _mm256_add_ps(a, b); }
static inline dyn_vec_t dyn_vfma(dyn_vec_t a, dyn_vec_t b, dyn_vec_t c){
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

// ============================================================================
//  ARM NEON back-end (AArch64 / ARMv7)  -- 4 x float
// ============================================================================
#elif defined(DYN_HAS_NEON)
#include <arm_neon.h>

#define DYN_SIMD_WIDTH 4
#define DYN_SIMD_NAME  "ARM-NEON"
using dyn_vec_t = float32x4_t;

static inline dyn_vec_t dyn_vzero()                      { return vdupq_n_f32(0.0f); }
static inline dyn_vec_t dyn_vset1(float v)               { return vdupq_n_f32(v); }
static inline dyn_vec_t dyn_vload (const float* p)       { return vld1q_f32(p); }
static inline dyn_vec_t dyn_vloadu(const float* p)       { return vld1q_f32(p); }
static inline void      dyn_vstore (float* p, dyn_vec_t x){ vst1q_f32(p, x); }
static inline void      dyn_vstoreu(float* p, dyn_vec_t x){ vst1q_f32(p, x); }
static inline dyn_vec_t dyn_vmul(dyn_vec_t a, dyn_vec_t b){ return vmulq_f32(a, b); }
static inline dyn_vec_t dyn_vadd(dyn_vec_t a, dyn_vec_t b){ return vaddq_f32(a, b); }
static inline dyn_vec_t dyn_vfma(dyn_vec_t a, dyn_vec_t b, dyn_vec_t c){
    return vfmaq_f32(c, a, b);   // c += a*b
}

// ============================================================================
//  SSE2 back-end  -- 4 x float
// ============================================================================
#elif defined(DYN_HAS_SSE2)
#include <emmintrin.h>

#define DYN_SIMD_WIDTH 4
#define DYN_SIMD_NAME  "SSE2"
using dyn_vec_t = __m128;

static inline dyn_vec_t dyn_vzero()                      { return _mm_setzero_ps(); }
static inline dyn_vec_t dyn_vset1(float v)               { return _mm_set1_ps(v); }
static inline dyn_vec_t dyn_vload (const float* p)       { return _mm_load_ps(p); }
static inline dyn_vec_t dyn_vloadu(const float* p)       { return _mm_loadu_ps(p); }
static inline void      dyn_vstore (float* p, dyn_vec_t x){ _mm_store_ps(p, x); }
static inline void      dyn_vstoreu(float* p, dyn_vec_t x){ _mm_storeu_ps(p, x); }
static inline dyn_vec_t dyn_vmul(dyn_vec_t a, dyn_vec_t b){ return _mm_mul_ps(a, b); }
static inline dyn_vec_t dyn_vadd(dyn_vec_t a, dyn_vec_t b){ return _mm_add_ps(a, b); }
static inline dyn_vec_t dyn_vfma(dyn_vec_t a, dyn_vec_t b, dyn_vec_t c){
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}

// ============================================================================
//  Portable scalar fallback  -- implemented as a 4-wide struct so the rest
//  of the pipeline looks identical.
// ============================================================================
#else

#define DYN_SIMD_WIDTH 4
#define DYN_SIMD_NAME  "scalar-x4"

struct dyn_vec_t {
    float v[4];
};
static inline dyn_vec_t dyn_vzero() {
    dyn_vec_t r; r.v[0]=r.v[1]=r.v[2]=r.v[3]=0.0f; return r;
}
static inline dyn_vec_t dyn_vset1(float x) {
    dyn_vec_t r; r.v[0]=r.v[1]=r.v[2]=r.v[3]=x; return r;
}
static inline dyn_vec_t dyn_vload (const float* p) {
    dyn_vec_t r; std::memcpy(r.v, p, sizeof(r.v)); return r;
}
static inline dyn_vec_t dyn_vloadu(const float* p) { return dyn_vload(p); }
static inline void dyn_vstore (float* p, dyn_vec_t x){ std::memcpy(p, x.v, sizeof(x.v)); }
static inline void dyn_vstoreu(float* p, dyn_vec_t x){ std::memcpy(p, x.v, sizeof(x.v)); }
static inline dyn_vec_t dyn_vmul(dyn_vec_t a, dyn_vec_t b) {
    dyn_vec_t r;
    for (int i=0;i<4;++i) r.v[i] = a.v[i]*b.v[i];
    return r;
}
static inline dyn_vec_t dyn_vadd(dyn_vec_t a, dyn_vec_t b) {
    dyn_vec_t r;
    for (int i=0;i<4;++i) r.v[i] = a.v[i]+b.v[i];
    return r;
}
static inline dyn_vec_t dyn_vfma(dyn_vec_t a, dyn_vec_t b, dyn_vec_t c) {
    dyn_vec_t r;
    for (int i=0;i<4;++i) r.v[i] = a.v[i]*b.v[i] + c.v[i];
    return r;
}
#endif

// ---------- software prefetch ----------
//
// Two-tier portable prefetch:
//
//   dyn_prefetch     -- "bring into L1, high temporal locality" (T0/PLDL1KEEP).
//                       Used at short distance (~4 nnz ahead) to cover the
//                       L2->L1 transfer in the steady state.
//   dyn_prefetch_l2  -- "bring into L2 only, moderate locality"  (T2/PLDL2KEEP).
//                       Used at long distance (~16 nnz ahead) so the DRAM->L2
//                       fill overlaps with many FMA chains.  On random
//                       factor-matrix access (Y[w][idx[w][i]]) this is the
//                       only way to hide full DRAM latency -- hardware
//                       stream prefetchers don't trigger on non-sequential
//                       reads.  On any unknown back-end we degrade to a
//                       no-op so correctness is never affected.
#if defined(DYN_HAS_AVX512) || defined(DYN_HAS_AVX2) || defined(DYN_HAS_SSE2)
  #include <xmmintrin.h>
  static inline void dyn_prefetch   (const void* p) {
      _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
  }
  static inline void dyn_prefetch_l2(const void* p) {
      _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T2);
  }
#elif defined(__GNUC__) || defined(__clang__)
  static inline void dyn_prefetch   (const void* p) {
      __builtin_prefetch(p, /*rw=*/0, /*locality=*/3);
  }
  static inline void dyn_prefetch_l2(const void* p) {
      __builtin_prefetch(p, /*rw=*/0, /*locality=*/2);
  }
#else
  static inline void dyn_prefetch   (const void* /*p*/) { /* no-op */ }
  static inline void dyn_prefetch_l2(const void* /*p*/) { /* no-op */ }
#endif

// ---------- branch-likelihood hints ----------
#if defined(__GNUC__) || defined(__clang__)
  #define DYN_LIKELY(x)   (__builtin_expect(!!(x), 1))
  #define DYN_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
  #define DYN_LIKELY(x)   (x)
  #define DYN_UNLIKELY(x) (x)
#endif

// ---------- non-temporal (streaming) scalar stores ----------
//
// Target use-case: the remap/scatter output buffers (`scr_vals`, `scr_idx`)
// in Dynasor are WRITE-ONCE per mode and not read back until the next mode
// starts consuming them (by which point every cache line that held the
// write is long evicted).  An ordinary store incurs a Read-For-Ownership
// that loads the old line into L1 before writing -- effectively doubling
// the write bandwidth we pay.  An NT scalar store skips the RFO and goes
// via the write-combining buffers straight to memory.
//
// Correctness requires an sfence/DMB before any subsequent reader
// re-interprets the stored bytes.  Callers issue that fence *after* their
// OpenMP parallel region (which is itself an implicit barrier).
//
// NEON has no scalar NT store; we fall back to an ordinary store, which
// still benefits from the ARMv8 write-combining buffers.  The fence is
// still emitted so the ARM DMB flushes any pending WC writes.
#if defined(DYN_HAS_AVX512) || defined(DYN_HAS_AVX2) || defined(DYN_HAS_SSE2)
  static inline void dyn_stream_f32(float* p, float v) {
      int32_t bits;
      std::memcpy(&bits, &v, sizeof(bits));
      _mm_stream_si32(reinterpret_cast<int*>(p), bits);
  }
  static inline void dyn_stream_u32(uint32_t* p, uint32_t v) {
      _mm_stream_si32(reinterpret_cast<int*>(p), (int)v);
  }
  static inline void dyn_sfence_stores() { _mm_sfence(); }
#elif defined(DYN_HAS_NEON)
  static inline void dyn_stream_f32(float* p, float v)      { *p = v; }
  static inline void dyn_stream_u32(uint32_t* p, uint32_t v){ *p = v; }
  static inline void dyn_sfence_stores() {
    #if defined(__aarch64__) || defined(__arm__)
      __asm__ __volatile__("dmb ishst" ::: "memory");
    #endif
  }
#else
  static inline void dyn_stream_f32(float* p, float v)      { *p = v; }
  static inline void dyn_stream_u32(uint32_t* p, uint32_t v){ *p = v; }
  static inline void dyn_sfence_stores() {}
#endif

// ---------- cache-line NT zero ----------
//
// Helper used by the "NT-store Yhat zero" path at sweep start (see
// zero_output_bulk in dynasor.cpp).  When Yhat is bigger than LLC, the
// plain-memset zero fills cache lines that will be evicted before the
// kernel ever loads them, wasting write bandwidth on RFO.  Replacing the
// memset with write-combining NT stores skips the RFO entirely: the
// bytes go straight to memory, and the kernel's first read miss picks
// them up from DRAM.  On L3-resident workloads the reverse is true (we
// want the zeros in cache), so the caller gates the NT path on a size
// threshold; if the threshold is not met it falls back to memset.
//
// `dyn_nt_zero_cacheline` writes 64 bytes of zeros to a 64-B aligned
// destination and is the primitive that callers chain in a loop.
//
//   * x86 SSE2+:  one _mm_stream_si128 per 16 B => 4 stores per line.
//   * AVX-512:    one _mm512_stream_si512 per line (single store).
//   * AVX2:       two _mm256_stream_si256 per line.
//   * Others:     ordinary 64 B memset (no NT path available).
#if defined(DYN_HAS_AVX512)
  static inline void dyn_nt_zero_cacheline(void* p) {
      __m512i z = _mm512_setzero_si512();
      _mm512_stream_si512(reinterpret_cast<__m512i*>(p), z);
  }
#elif defined(DYN_HAS_AVX2)
  static inline void dyn_nt_zero_cacheline(void* p) {
      __m256i z = _mm256_setzero_si256();
      _mm256_stream_si256(reinterpret_cast<__m256i*>(p) + 0, z);
      _mm256_stream_si256(reinterpret_cast<__m256i*>(p) + 1, z);
  }
#elif defined(DYN_HAS_SSE2)
  static inline void dyn_nt_zero_cacheline(void* p) {
      __m128i z = _mm_setzero_si128();
      _mm_stream_si128(reinterpret_cast<__m128i*>(p) + 0, z);
      _mm_stream_si128(reinterpret_cast<__m128i*>(p) + 1, z);
      _mm_stream_si128(reinterpret_cast<__m128i*>(p) + 2, z);
      _mm_stream_si128(reinterpret_cast<__m128i*>(p) + 3, z);
  }
#else
  static inline void dyn_nt_zero_cacheline(void* p) {
      std::memset(p, 0, 64);
  }
#endif

// ---------- tiny helpers shared by all back-ends ----------

// Round `x` up to the next multiple of `m` (m must be a power of two).
static inline int dyn_round_up(int x, int m) { return (x + m - 1) & ~(m - 1); }

// ---------- OS hints for server-class Linux targets ----------
//
// Two things matter when running large-tensor spMTTKRP on a 64-core Zen 5
// or Neoverse V2 box under Ubuntu:
//
//   1. Transparent huge pages.  Default 4 KiB pages put several million
//      DTLB entries in play for a 1 GiB tensor; a single TLB miss can
//      cost 50+ cycles.  `madvise(MADV_HUGEPAGE)` asks the kernel to
//      back the region with 2 MiB pages where possible -- a ~5-15% win
//      on large-nnz tensors with no downside when not available.
//
//   2. NUMA first-touch.  Linux assigns a physical page to whichever
//      NUMA node the FIRST writing thread is running on.  If a large
//      buffer is filled single-threaded (as during tensor load) every
//      page lands on socket 0 and remote accesses from socket 1 pay
//      50%+ bandwidth penalty.  `dyn_parallel_touch` walks the buffer
//      in parallel (one page per thread, static schedule) so the kernel
//      later re-finds the same threads owning the same pages.
//
// Both functions are no-ops on non-Linux targets.
#if defined(__linux__)
  #include <sys/mman.h>
  #include <sys/syscall.h>
  #include <unistd.h>
  #include <cctype>
  static inline void dyn_advise_huge(void* p, size_t bytes) {
      if (!p || !bytes) return;
      // ignore errors; this is a hint
      (void)madvise(p, bytes, MADV_HUGEPAGE);
  }

  // ---------- NUMA interleave (for Y factor matrices) ----------
  //
  // Y is read with random index patterns during MTTKRP (Y[w][idx[w][i]]).
  // On a 2-socket server the default first-touch policy lands every page
  // on socket 0 (the loader's socket), so socket-1 threads pay 50%+ more
  // latency on every Y fetch.  Asking the kernel to interleave Y's pages
  // across NUMA nodes evens that cost out; it's not as good as a full
  // per-node replica (which the `DYN_NUMA_REPLICATE` roadmap adds), but
  // it's a one-syscall change that wins on every multi-NUMA Linux box
  // with zero code-path changes elsewhere.
  //
  // Implemented directly on top of the mbind syscall so we don't drag in
  // libnuma.  On non-Linux or single-node systems this is a no-op and
  // never fails.
  //
  // Yhat is intentionally NOT interleaved -- the driver's parallel
  // zero/write patterns give it a coherent owner-thread, so first-touch
  // keeps those pages local to the writer.
  static inline int dyn_numa_max_node() {
      FILE* fp = std::fopen("/sys/devices/system/node/online", "r");
      if (!fp) return 0;
      char buf[64] = {0};
      size_t nb = std::fread(buf, 1, sizeof(buf) - 1, fp);
      std::fclose(fp);
      long maxn = 0;
      for (size_t i = 0; i < nb; ) {
          while (i < nb && !std::isdigit((unsigned char)buf[i])) ++i;
          long v = 0; bool any = false;
          while (i < nb && std::isdigit((unsigned char)buf[i])) {
              v = v * 10 + (buf[i] - '0'); ++i; any = true;
          }
          if (any && v > maxn) maxn = v;
      }
      return (int)maxn;
  }

  static inline void dyn_interleave_pages(void* p, size_t bytes) {
      if (!p || !bytes) return;
      const int maxn = dyn_numa_max_node();
      if (maxn <= 0) return;   // single-node box -- nothing to spread.
      const unsigned long mask =
          (maxn + 1 >= 64) ? ~0UL : ((1UL << (maxn + 1)) - 1UL);
      // MPOL_INTERLEAVE == 3 in every Linux kernel since 2.6.x.
      (void)syscall(SYS_mbind, p, bytes, 3 /*MPOL_INTERLEAVE*/,
                    &mask, (unsigned long)(maxn + 2), 0);
  }
#else
  static inline void dyn_advise_huge(void* /*p*/, size_t /*bytes*/) {}
  static inline int  dyn_numa_max_node() { return 0; }
  static inline void dyn_interleave_pages(void* /*p*/, size_t /*bytes*/) {}
#endif

// One-shot startup check.  Reads /sys/kernel/mm/transparent_hugepage/enabled
// and prints a warning if THP is disabled, because MADV_HUGEPAGE is a no-op
// in that state.  Also reports whether explicit huge pages (vm.nr_hugepages)
// have any pool reserved, which is required for DYN_USE_HUGETLB.  Silent and
// empty on non-Linux targets so Windows / macOS never see stray output.
static inline void dyn_report_hugepages() {
#if defined(__linux__)
    FILE* fp = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (fp) {
        char buf[128] = {0};
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, fp);
        std::fclose(fp);
        if (n > 0) {
            // Format is e.g. "always [madvise] never".  The bracketed token
            // is the active mode.
            if (std::strstr(buf, "[never]")) {
                std::fprintf(stderr,
                    "[warn] transparent_hugepage=never -- MADV_HUGEPAGE hints "
                    "will be ignored.  Expect reduced performance on large "
                    "tensors.  Fix: echo madvise | sudo tee "
                    "/sys/kernel/mm/transparent_hugepage/enabled\n");
            }
        }
    }
    // Explicit huge-page pool (only relevant when DYN_USE_HUGETLB was set).
    fp = std::fopen("/proc/sys/vm/nr_hugepages", "r");
    if (fp) {
        long nh = 0;
        if (std::fscanf(fp, "%ld", &nh) == 1) {
            std::printf("[info] THP mode hinted via MADV_HUGEPAGE; "
                        "vm.nr_hugepages = %ld\n", nh);
        }
        std::fclose(fp);
    }
#endif
}

// Parallel first-touch: write a single byte per 4 KiB page so the OS
// faults each page on the thread that will later access it.  The caller
// is expected to invoke this from serial code (we spawn our own parallel
// region).  Harmless if called on a warmed-up allocation.
static inline void dyn_parallel_touch(void* p, size_t bytes) {
    if (!p || !bytes) return;
    constexpr size_t kPage = 4096;
    char*  cp      = static_cast<char*>(p);
    size_t n_pages = (bytes + kPage - 1) / kPage;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n_pages; ++i) {
        cp[(size_t)i * kPage] = 0;
    }
}

// Aligned allocate / free (portable). Size is in *bytes*.
//
// We roll our own over-allocating wrapper so the code compiles on every
// tool-chain we care about (MSVC, MinGW, glibc, musl, Apple libc, Android,
// AArch64 Linux).  Store the original pointer in the word immediately below
// the aligned address to recover it on free.
#include <cstdlib>
#include <cstdint>
// ---------- aligned allocation ----------
//
// Two-tier policy:
//
//   * bytes <  DYN_HUGE_THRESHOLD   -> DYN_ALIGN (64 B, cache line)
//   * bytes >= DYN_HUGE_THRESHOLD   -> 2 MiB, i.e. the Linux huge page size
//
// Aligning multi-MiB buffers to 2 MiB lets khugepaged fold the region into
// transparent huge pages without having to "split" pages at our boundaries.
// On non-Linux targets the extra alignment is free (at worst a 2 MiB tail
// of padding on one allocation per run).
//
// The Linux branch (DYN_USE_HUGETLB=1 at compile time) first tries an
// explicit MAP_HUGETLB mmap for the large path and transparently falls
// back to std::malloc when the kernel has no huge pages reserved.  Opt-in
// because MAP_HUGETLB fails outright without sysctl vm.nr_hugepages set.

#ifndef DYN_HUGE_THRESHOLD
  #define DYN_HUGE_THRESHOLD (2u * 1024u * 1024u)  // 2 MiB
#endif
#ifndef DYN_HUGE_ALIGN
  #define DYN_HUGE_ALIGN (2u * 1024u * 1024u)      // 2 MiB
#endif

// Tag byte: were we allocated via mmap(MAP_HUGETLB)?  Stored just below
// the user pointer together with the raw pointer, so free() can decide.
struct dyn_alloc_hdr {
    void*  raw;     // pointer to free / munmap base
    size_t bytes;   // mapping size (only valid for mmap path)
    int    kind;    // 0 = malloc, 1 = mmap
};

#if defined(__linux__) && defined(DYN_USE_HUGETLB)
  #include <sys/mman.h>
#endif

// Windows large-page opt-in.  Gated on env `DYN_WIN_LARGE_PAGES=1` and
// succeeds only when the process holds `SeLockMemoryPrivilege` (i.e. the
// user is in the "Lock pages in memory" policy and the process was
// started with the privilege enabled).  Silent fallback to std::malloc
// on every failure mode -- typical non-admin users will get the
// fallback, which matches pre-existing behaviour.
//
// kind == 2 in dyn_alloc_hdr denotes VirtualAlloc, freed with VirtualFree.
#if defined(_WIN32)
  #include <windows.h>

  static inline bool dyn_win_large_pages_enabled() {
      const char* s = std::getenv("DYN_WIN_LARGE_PAGES");
      return s && s[0] && s[0] != '0';
  }

  // One-shot privilege enable.  Best-effort; a failure just means we fall
  // through to std::malloc on the large path.  Idempotent: repeated calls
  // are harmless (AdjustTokenPrivileges returns TRUE even if the priv is
  // already enabled).
  static inline bool dyn_win_enable_lock_memory_privilege() {
      HANDLE tok;
      if (!OpenProcessToken(GetCurrentProcess(),
                            TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok))
          return false;
      TOKEN_PRIVILEGES tp = {};
      tp.PrivilegeCount = 1;
      tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
      if (!LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege",
                                 &tp.Privileges[0].Luid)) {
          CloseHandle(tok);
          return false;
      }
      const BOOL ok = AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
      const DWORD le = GetLastError();
      CloseHandle(tok);
      return ok && le == ERROR_SUCCESS;
  }
#endif

static inline void* dyn_aligned_alloc(size_t bytes) {
    const size_t align =
        (bytes >= (size_t)DYN_HUGE_THRESHOLD) ? (size_t)DYN_HUGE_ALIGN
                                              : (size_t)DYN_ALIGN;

#if defined(__linux__) && defined(DYN_USE_HUGETLB)
    // Try an explicit huge-page mapping for large allocations.
    if (bytes >= (size_t)DYN_HUGE_THRESHOLD) {
        // Round up to a 2 MiB multiple, then reserve one extra huge page
        // at the front for our dyn_alloc_hdr bookkeeping.
        const size_t page  = (size_t)DYN_HUGE_ALIGN;
        const size_t rnd   = (bytes + page - 1) & ~(page - 1);
        const size_t total = rnd + page;
        void* raw = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (raw != MAP_FAILED) {
            char* user_base = (char*)raw + page;    // 2 MiB aligned
            dyn_alloc_hdr* h = ((dyn_alloc_hdr*)user_base) - 1;
            h->raw = raw; h->bytes = total; h->kind = 1;
            return user_base;
        }
        // Fall through to the malloc-based path on failure (no huge pages
        // reserved -- this is the common non-root case and must not abort).
    }
#endif

#if defined(_WIN32)
    // Opt-in large-page mapping for large allocations.  Only enters the
    // VirtualAlloc path when:
    //   (a) bytes >= DYN_HUGE_THRESHOLD (otherwise large pages are waste),
    //   (b) env DYN_WIN_LARGE_PAGES=1 (off by default), and
    //   (c) SeLockMemoryPrivilege can be enabled on this token.
    // The minimum large-page size on Windows is typically 2 MiB
    // (GetLargePageMinimum()); round up to that granularity so the call
    // succeeds.  kind == 2 => VirtualFree on free.
    if (bytes >= (size_t)DYN_HUGE_THRESHOLD && dyn_win_large_pages_enabled()) {
        static bool priv_tried = false;
        static bool priv_ok    = false;
        if (!priv_tried) {
            priv_ok    = dyn_win_enable_lock_memory_privilege();
            priv_tried = true;
        }
        if (priv_ok) {
            const SIZE_T page = GetLargePageMinimum();
            if (page) {
                const size_t rnd   = (bytes + page - 1) & ~(page - 1);
                // Reserve one extra large page for the dyn_alloc_hdr.
                const size_t total = rnd + page;
                void* raw = VirtualAlloc(nullptr, total,
                                         MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                         PAGE_READWRITE);
                if (raw) {
                    char* user_base = (char*)raw + page;
                    dyn_alloc_hdr* h = ((dyn_alloc_hdr*)user_base) - 1;
                    h->raw = raw; h->bytes = total; h->kind = 2;
                    return user_base;
                }
                // Fall through to malloc on allocator failure (not
                // enough contiguous large pages, etc.).
            }
        }
    }
#endif

    const size_t padded = bytes + align + sizeof(dyn_alloc_hdr);
    void* raw = std::malloc(padded);
    if (!raw) return nullptr;
    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw) + sizeof(dyn_alloc_hdr);
    uintptr_t aligned  = (raw_addr + align - 1) & ~(uintptr_t)(align - 1);
    dyn_alloc_hdr* h = ((dyn_alloc_hdr*)aligned) - 1;
    h->raw = raw; h->bytes = 0; h->kind = 0;
    return reinterpret_cast<void*>(aligned);
}

static inline void dyn_aligned_free(void* p) {
    if (!p) return;
    dyn_alloc_hdr* h = ((dyn_alloc_hdr*)p) - 1;
#if defined(__linux__) && defined(DYN_USE_HUGETLB)
    if (h->kind == 1) {
        munmap(h->raw, h->bytes);
        return;
    }
#endif
#if defined(_WIN32)
    if (h->kind == 2) {
        VirtualFree(h->raw, 0, MEM_RELEASE);
        return;
    }
#endif
    std::free(h->raw);
}

#endif // DYNASOR_SIMD_H
