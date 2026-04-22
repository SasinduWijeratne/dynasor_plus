// ============================================================================
//  main.cpp : CLI driver for Dynasor+
//
//  Usage:
//      dynasor_plus <tensor.tns> [--rank R] [--threads T] [--shard-size g]
//                                [--verify] [--dump OUT_PREFIX]
//
//  Example:
//      ./dynasor_plus data/3d_7.tns --rank 16 --threads 4 --verify
//
//  Runs both the reference single-thread implementation and the parallel
//  Dynasor implementation, then reports timing and the maximum relative
//  error between the two factor-matrix sets.
// ============================================================================
#include "dynasor_common.h"
#include "dynasor_cpals.h"
#include "dynasor_dnp.h"
#include "dynasor_jit.h"
#include "dynasor_morton.h"
#include "dynasor_ooc.h"
#include "dynasor_simd.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace dynasor;

struct Args {
    std::string tensor_path;
    int         rank       = 16;
    int         threads    = 0;        // 0 == use OMP default
    int         shard_size = 16;
    bool        verify     = false;
    std::string dump_prefix;
    bool        dump_binary = false;
    bool        no_cache   = false;
    uint64_t    seed       = 42;
    std::string layout     = "auto";    // auto | pingpong | ncopy | inplace
    // Fiber-sort pass for NCopy:
    //   "auto" -> default ON in preprocess (matches internal default).
    //   "on"   -> force DYN_NCOPY_FIBER_SORT=1.
    //   "off"  -> force DYN_NCOPY_FIBER_SORT=0 (element kernel for every
    //             mode under NCopy; skip the one-time sort).
    std::string ncopy_fiber_sort = "auto";

    // CSR row-pointer build for NCopy + DenseFiber modes (Stage 2A):
    //   "auto" -> default ON (matches internal default when fiber-sort
    //              also runs).
    //   "on"   -> force DYN_NCOPY_CSR=1.
    //   "off"  -> force DYN_NCOPY_CSR=0 (COO fiber kernel path).
    std::string ncopy_csr = "auto";

    // CSR slab compaction for NCopy DenseFiber modes (Stage 2B):
    //   "auto" -> default ON except when --verify is set, where it
    //              auto-disables because compaction frees the primary
    //              and scratch slabs that the reference kernel reads.
    //   "on"   -> force DYN_NCOPY_CSR_COMPACT=1.
    //   "off"  -> force DYN_NCOPY_CSR_COMPACT=0 (keep ix[k] in slab).
    std::string ncopy_csr_compact = "auto";

    // CP-ALS extensions.
    std::string decompose;             // "" (none) or "cpals"
    int         cp_iters   = 50;
    double      cp_tol     = 1e-4;

    // CP-ALS MTTKRP strategy: "auto" (default, planner picks the best
    // kernel based on N, R, tensor footprint and memory budget), or an
    // explicit value -- "gauss-seidel" (classical per-mode update) or
    // "jacobi" (all-modes single-pass; each iter does one tensor pass
    // but with decoupled sub-problem updates).  Also overridable via
    // DYN_CP_KERNEL={auto,gs,jacobi}.  See CpKernel in dynasor_cpals.h.
    //
    //  Auto-selection rule (Stage 1 planner):
    //    Pick Jacobi when the all-modes kernel would NOT auto-fall-back
    //    (N != 3 || Rp < 64) && Rp < 128, AND its per-thread ofibs
    //    buffer fits comfortably in RAM, AND MTTKRP dominates the iter
    //    (nnz >= ~1e6).  Otherwise pick Gauss-Seidel.  OOC tensors
    //    always force Jacobi (only OOC driver).
    std::string cp_kernel  = "auto";
    double      cp_ridge   = 1e-10;
    std::string cp_dump;                // prefix for factor output
    std::string cp_dump_init;           // prefix for *initial* factor dump

    // Stage-3 JIT: when true, specialize the fiber_csr kernel for the
    // exact (N, Rp, target_mode, pf_far) of this run by shelling out to
    // $(CXX) at first dispatch.  See include/dynasor_jit.h.  Env var
    // DYN_JIT=1 is equivalent.
    bool        jit        = false;

    // All-modes single-pass spMTTKRP driver.  Streams the tensor once
    // per iteration instead of once per target mode, updating every
    // Yhat[n] in a single kernel pass.  Targets the medium-tensor
    // regime (10M-100M nnz) where ALTO otherwise wins.  Env
    // DYN_ALL_MODES=1 is equivalent.
    bool        all_modes  = false;

    // Out-of-core mode.  The tensor stays on disk (as its .dnb cache) and
    // is streamed through RAM one chunk at a time during every CP-ALS
    // iteration.  Value meanings:
    //   "auto" (default) -- use OOC only when the in-RAM layout decision
    //                       would fail (projected_need > available).
    //   "on"             -- force OOC even if the tensor fits in RAM.
    //                       Useful for benchmarking the streaming path.
    //   "off"            -- never use OOC; abort when the tensor does not
    //                       fit in available RAM (legacy behaviour).
    // Overridable by DYN_OOC={auto,on,off}.  Only --decompose cpals honors
    // this; the standalone MTTKRP benchmark path has no OOC driver.
    std::string ooc        = "auto";

    // Streaming chunk size in nonzeros (only consulted when OOC is active).
    //   0  -> auto-pick (~256 MiB per chunk, see ooc_default_chunk_nnz).
    // Env override: DYN_OOC_CHUNK (nnz) or DYN_OOC_CHUNK_BYTES (bytes).
    uint64_t    ooc_chunk  = 0;

    // Dynasor+ framework: processed-tensor I/O.
    //
    //   --save-processed PATH
    //       After preprocessing completes, serialize the fully-prepared
    //       tensor (selected layout + all FLYCOO / Morton / CSR state)
    //       to <PATH>.  Subsequent runs can skip preprocessing entirely
    //       by loading the .dnp file with --runtime.  Ignored for OOC
    //       tensors (the .dnb cache already serves as their runtime-
    //       only format).
    //
    //   --runtime
    //       Treat <tensor_path> as a .dnp file produced by a prior
    //       --save-processed invocation.  Skips build_flycoo /
    //       build_morton_layout / NCopy sort and jumps straight to
    //       CP-ALS.  The thread count baked into the .dnp (schedule
    //       is thread-specific) MUST match --threads, otherwise the
    //       load is rejected with an actionable error.
    std::string save_processed;
    bool        runtime_only = false;
};

static void usage(const char* arg0) {
    std::printf(
        "Usage: %s <tensor.tns> [options]\n"
        "Options:\n"
        "  --rank R         factor-matrix rank (default 16)\n"
        "  --threads T      number of OpenMP threads (default: system max)\n"
        "  --shard-size g   FLYCOO shard size (default 16)\n"
        "  --verify         also run reference + compute max rel. error\n"
        "  --dump PREFIX    dump Yhat_n to <PREFIX>.fm<n>_out.txt\n"
        "  --dump-bin       use raw float32 format (.fm<n>_out.bin) instead\n"
        "  --no-cache       do not read or write the .dnb binary tensor cache\n"
        "  --seed S         RNG seed (default 42)\n"
        "  --layout L       tensor storage layout:\n"
        "                     auto|pingpong|ncopy|inplace|morton|ooc\n"
        "                   (default auto; also overridable via DYN_LAYOUT env).\n"
        "                   The planner (dyn_plan_storage) evaluates ALL four\n"
        "                   candidates each run and prints a unified banner\n"
        "                   ([plan] pre-load when a .dnb cache exists, or\n"
        "                   [layout] post-load otherwise).\n"
        "                     auto     : NCopy when it fits in RAM, else\n"
        "                                PingPong, else InPlace, else OOC.\n"
        "                                Recommended.\n"
        "                     pingpong : 2x nnz footprint, per-iter remap (legacy).\n"
        "                     ncopy    : N pre-sorted physical copies, no remap.\n"
        "                                Fastest when the tensor fits, OOM if not.\n"
        "                     inplace  : 1x nnz footprint (minimum in-core).\n"
        "                                Re-sorts the primary slab in place before\n"
        "                                each mode's MTTKRP.  ~0-5%% slower than\n"
        "                                PingPong when PingPong also fits.\n"
        "                     morton   : ALTO-style linearized layout (Stage 2).\n"
        "                                12 B / nnz; one Morton sort, no per-mode\n"
        "                                re-sort.  Requires sum(ceil(log2(I_n)))<=64.\n"
        "                                Auto-promotes --cp-kernel to jacobi.\n"
        "                     ooc      : out-of-core streaming (requires .dnb\n"
        "                                cache + --decompose cpals).  Equivalent\n"
        "                                to --ooc on; tensor lives on disk,\n"
        "                                only factors + ofibs + 1 chunk in RAM.\n"
        "  --ncopy-fiber-sort M  enable/disable one-time fiber-sort pass for\n"
        "                        NCopy's DenseFiber modes.  auto|on|off (default\n"
        "                        auto = on).  Also overridable via env\n"
        "                        DYN_NCOPY_FIBER_SORT=0|1.  Off: fiber kernel\n"
        "                        falls back to element kernel for affected modes\n"
        "                        (useful for short ALS runs where the preprocess\n"
        "                        sort cost outweighs the per-iter kernel win).\n"
        "  --ncopy-csr M         enable/disable CSR row-pointer build + kernel\n"
        "                        dispatch for NCopy DenseFiber modes. auto|on|off\n"
        "                        (default auto = on when fiber-sort is on).\n"
        "                        Env override DYN_NCOPY_CSR=0|1. No effect under\n"
        "                        PingPong (CSR is incompatible with per-iter\n"
        "                        remap). Off: falls back to COO fiber kernel.\n"
        "  --ncopy-csr-compact M enable/disable CSR slab compaction (drops\n"
        "                        ix[k] column from copy k, reducing tensor\n"
        "                        memory by ~N/(N+1)^2 per CSR mode). auto|on|off\n"
        "                        (default auto = on; off when --verify is set\n"
        "                        to preserve the primary/scratch slabs the\n"
        "                        reference kernel reads). Env override\n"
        "                        DYN_NCOPY_CSR_COMPACT=0|1.\n"
        "  --jit            shell out to $(CXX) at runtime to compile a\n"
        "                   fiber_csr kernel specialized for this tensor's\n"
        "                   exact (N, rank_padded, target_mode, pf_far)\n"
        "                   tuple.  Opt-in, default off.  Env DYN_JIT=1 is\n"
        "                   equivalent.  Compiled .so/.dll are cached under\n"
        "                   .dyn_jit_cache/ keyed on ISA + toolchain, so\n"
        "                   only first-ever compiles pay the 2-4 s cost.\n"
        "  --all-modes      use the all-modes single-pass MTTKRP driver:\n"
        "                   stream the tensor once per iteration and update\n"
        "                   every Yhat[n] in one kernel pass, using per-\n"
        "                   thread private output buffers (ofibs).  Cuts\n"
        "                   tensor-bandwidth by N (3-4x on medium tensors).\n"
        "                   Falls back to the per-mode driver if ofibs do\n"
        "                   not fit.  Only applies to the standalone MTTKRP\n"
        "                   benchmark (not CP-ALS -- that needs Gauss-Seidel\n"
        "                   factor updates between modes).  Env DYN_ALL_MODES=1\n"
        "                   is equivalent.\n"
        "\n"
        "Decomposition:\n"
        "  --decompose cpals        run CP-ALS after preprocessing (skips\n"
        "                           the standalone MTTKRP benchmark path)\n"
        "  --cp-iters N             CP-ALS max iterations (default 50)\n"
        "  --cp-tol TOL             fit-change convergence threshold (default 1e-4)\n"
        "  --cp-kernel K            MTTKRP strategy inside each ALS iteration:\n"
        "                           auto (default): planner picks jacobi when\n"
        "                                  the all-modes kernel would win (N<=4,\n"
        "                                  Rp not in the scatter-heavy regime,\n"
        "                                  ofibs fit in RAM) and MTTKRP dominates;\n"
        "                                  otherwise falls back to gauss-seidel.\n"
        "                           gauss-seidel: classical per-mode update;\n"
        "                                  one tensor pass per mode per iter\n"
        "                                  (N passes), freshest factors for each\n"
        "                                  sub-problem (fast convergence).\n"
        "                           jacobi: all-modes single-pass MTTKRP; one\n"
        "                                  tensor pass per iter with per-thread\n"
        "                                  private ofibs, updates decoupled.\n"
        "                                  Slower per-iter convergence but each\n"
        "                                  iter is ~N/1 cheaper in MTTKRP work.\n"
        "                           Env override: DYN_CP_KERNEL={auto,gs,jacobi}.\n"
        "  --ooc MODE               out-of-core streaming mode (CP-ALS only):\n"
        "                           auto (default): OOC only if the tensor is\n"
        "                                too large for any in-RAM layout.\n"
        "                           on:   force OOC even if it would fit.\n"
        "                           off:  never OOC; abort on OOM.\n"
        "                           Requires an existing <tensor>.dnb cache.\n"
        "                           Forces --cp-kernel jacobi (the only\n"
        "                           single-pass OOC driver).  Env DYN_OOC=.\n"
        "  --ooc-chunk NNZ          streaming window size in nonzeros\n"
        "                           (default auto ~256 MiB per chunk).  Env\n"
        "                           DYN_OOC_CHUNK_BYTES overrides byte-target.\n"
        "  --cp-ridge R             relative Cholesky ridge (default 1e-10)\n"
        "  --cp-dump PREFIX         after CP-ALS, write Y[n] to\n"
        "                           <PREFIX>.cp_fm<n>.bin (+ lambda.bin)\n"
        "  --cp-dump-init PREFIX    write the random initial factors before\n"
        "                           the first ALS sweep to <PREFIX>.cp_init_fm<n>.bin\n"
        "                           (lets external validators reproduce the run)\n"
        "\n"
        "Dynasor+ framework:\n"
        "  --save-processed PATH    after preprocessing, write the fully-\n"
        "                           prepared tensor (layout + FLYCOO /\n"
        "                           Morton / CSR state + schedule) to\n"
        "                           <PATH>.  Enables subsequent --runtime\n"
        "                           loads that skip preprocessing entirely.\n"
        "                           Not applicable to OOC tensors (.dnb\n"
        "                           already serves as their runtime-only\n"
        "                           format).\n"
        "  --runtime                treat <tensor_path> as a .dnp file from\n"
        "                           a previous --save-processed run.  Skips\n"
        "                           build_flycoo / build_morton_layout /\n"
        "                           NCopy sort and jumps straight to CP-ALS.\n"
        "                           --threads MUST match the thread count\n"
        "                           baked in at save time (the schedule is\n"
        "                           thread-specific).\n",
        arg0);
}

static bool parse(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.tensor_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* n) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", n); std::exit(1);
            }
            return argv[++i];
        };
        if      (k == "--rank")       a.rank = std::atoi(need("--rank"));
        else if (k == "--threads")    a.threads = std::atoi(need("--threads"));
        else if (k == "--shard-size") a.shard_size = std::atoi(need("--shard-size"));
        else if (k == "--seed")       a.seed = (uint64_t)std::atoll(need("--seed"));
        else if (k == "--verify")     a.verify = true;
        else if (k == "--dump")       a.dump_prefix = need("--dump");
        else if (k == "--dump-bin")   a.dump_binary = true;
        else if (k == "--no-cache")   a.no_cache = true;
        else if (k == "--decompose")  a.decompose = need("--decompose");
        else if (k == "--cp-iters")   a.cp_iters  = std::atoi(need("--cp-iters"));
        else if (k == "--cp-tol")     a.cp_tol    = std::atof(need("--cp-tol"));
        else if (k == "--cp-kernel")  a.cp_kernel = need("--cp-kernel");
        else if (k == "--cp-ridge")   a.cp_ridge  = std::atof(need("--cp-ridge"));
        else if (k == "--cp-dump")      a.cp_dump      = need("--cp-dump");
        else if (k == "--cp-dump-init") a.cp_dump_init = need("--cp-dump-init");
        else if (k == "--layout")       a.layout       = need("--layout");
        else if (k == "--ncopy-fiber-sort") a.ncopy_fiber_sort = need("--ncopy-fiber-sort");
        else if (k == "--ncopy-csr")        a.ncopy_csr        = need("--ncopy-csr");
        else if (k == "--ncopy-csr-compact") a.ncopy_csr_compact = need("--ncopy-csr-compact");
        else if (k == "--jit")        a.jit = true;
        else if (k == "--all-modes")  a.all_modes = true;
        else if (k == "--ooc")        a.ooc   = need("--ooc");
        else if (k == "--ooc-chunk")  a.ooc_chunk = (uint64_t)std::atoll(need("--ooc-chunk"));
        else if (k == "--save-processed") a.save_processed = need("--save-processed");
        else if (k == "--runtime")        a.runtime_only   = true;
        else if (k == "-h" || k == "--help") return false;
        else {
            std::fprintf(stderr, "unknown option: %s\n", k.c_str());
            return false;
        }
    }
    return true;
}

// Fill a factor matrix with uniform random values on [0,1).
static void random_init(value_t* Y, idx_t rows, int rank, int rank_padded,
                        std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    for (idx_t i = 0; i < rows; ++i) {
        for (int r = 0; r < rank; ++r) {
            Y[(size_t)i * rank_padded + r] = U(rng);
        }
        for (int r = rank; r < rank_padded; ++r)
            Y[(size_t)i * rank_padded + r] = 0.0f;
    }
}

int main(int argc, char** argv) {
    // Unbuffer stdout so any crash output up to the point of failure is seen.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Args args;
    if (!parse(argc, argv, args)) { usage(argv[0]); return 1; }

    // Stage-3 JIT: flip the global toggle BEFORE SpmttkrpSweep queries
    // dyn_jit_enabled() inside its constructor.  --jit on the CLI is
    // equivalent to DYN_JIT=1 in the environment; CLI wins when both
    // are present.
    if (args.jit) dyn_jit_force_enable(true);

    std::printf("=== Dynasor+  (backend: %s) ===\n", simd_backend_name());
    dyn_report_hugepages();

    // Allocate the tensor on the heap.  Large tensors do not belong on the
    // stack, and on MinGW-w64 this also dodges a libstdc++/OpenMP codegen
    // bug for stream objects declared inside very large frames.
    std::unique_ptr<Tensor> Tp(new Tensor());
    Tensor& T = *Tp;

    // ======================================================================
    //  Dynasor+ runtime-only mode.
    //
    //  When --runtime is set, `args.tensor_path` points at a .dnp file
    //  produced by a previous --save-processed invocation.  We skip the
    //  entire preprocessing pipeline (no .tns / .dnb load, no .dnb pre-
    //  plan, no build_flycoo, no build_morton_layout) and load the fully-
    //  prepared tensor state directly.  The thread count MUST match what
    //  was baked in at save time; the loader emits an actionable error
    //  otherwise.
    //
    //  Runtime-only mode is incompatible with --verify (the reference
    //  kernel depends on SoA buffers that Morton and CSR-compact NCopy
    //  have freed) and with --ooc (OOC already has its own runtime-only
    //  path via .dnb).
    // ======================================================================
    bool runtime_skip_preprocess = false;
    if (args.runtime_only) {
        if (args.verify) {
            std::fprintf(stderr,
                "[main] --runtime is incompatible with --verify "
                "(the reference kernel reads SoA slabs that some layouts "
                "discard after preprocessing).\n");
            return 7;
        }
        if (args.ooc == "on") {
            std::fprintf(stderr,
                "[main] --runtime and --ooc on are mutually exclusive; "
                "OOC tensors already have a runtime-only path via .dnb.\n");
            return 7;
        }

        // Resolve thread count (needed up-front -- the .dnp schedule is
        // thread-specific and we must pass the exact value to the loader).
        int nthreads = args.threads;
#ifdef _OPENMP
        if (nthreads <= 0) nthreads = omp_get_max_threads();
#else
        nthreads = 1;
#endif
        if (nthreads < 1) nthreads = 1;

        Tensor hdr;
        if (!load_dnp_header(args.tensor_path, hdr)) {
            std::fprintf(stderr,
                "[main] --runtime: cannot read .dnp header from '%s'.\n",
                args.tensor_path.c_str());
            return 2;
        }
        const char* hdr_layout_str =
            (hdr.layout == Layout::NCopy)    ? "NCopy"    :
            (hdr.layout == Layout::PingPong) ? "PingPong" :
            (hdr.layout == Layout::InPlace)  ? "InPlace"  :
            (hdr.layout == Layout::Morton)   ? "Morton"   : "?";
        std::printf("[runtime] loading .dnp: %s\n"
                    "          modes=%d  nnz=%llu  layout=%s  threads=%d\n",
                    args.tensor_path.c_str(),
                    hdr.num_modes, (unsigned long long)hdr.nnz,
                    hdr_layout_str, nthreads);

        if (!load_dnp_tensor(args.tensor_path, T, nthreads)) {
            return 2;
        }

        args.threads              = nthreads;
        runtime_skip_preprocess   = true;
    }

    // ----------------------------------------------------------------------
    //  Unified storage-layout decision (pre-load).
    //
    //  When a .dnb cache is present we can plan the layout BEFORE touching
    //  any NNZ: peek the header (nnz, num_modes, mode_sizes), then call
    //  dyn_plan_storage() -- the same function decide_and_populate_layout
    //  uses post-load -- to pick among {PingPong, NCopy, InPlace,
    //  OutOfCore}.  If the plan returns OutOfCore, the in-core loader is
    //  skipped entirely and the CP-ALS driver streams from disk.
    //
    //  OOC is only honored under --decompose cpals; the standalone MTTKRP
    //  benchmark has no streaming driver.  For that path we forcibly
    //  re-resolve the plan with ooc="off" so a misconfigured --ooc on /
    //  --layout ooc doesn't crash at runtime.
    // ----------------------------------------------------------------------
    bool use_ooc = false;
    std::string dnb_path = args.tensor_path + ".dnb";
    if (!runtime_skip_preprocess) {
        std::string ooc_cli    = args.ooc;
        std::string layout_cli = args.layout;
        // Standalone MTTKRP bench has no OOC driver -- clamp to off.
        if (args.decompose != "cpals") {
            if (ooc_cli == "on" || layout_cli == "ooc") {
                std::fprintf(stderr,
                    "[main] --ooc / --layout ooc ignored: OOC streaming only "
                    "applies to --decompose cpals\n");
            }
            ooc_cli    = "off";
            if (layout_cli == "ooc") layout_cli = "auto";
        }

        // Peek the .dnb header.  If absent, skip the pre-load plan: the
        // legacy loader will pick up from here and call
        // decide_and_populate_layout post-load.
        Tensor hdr;
        const bool have_dnb = ooc_load_header(dnb_path, hdr);
        // Mirror the override keywords dyn_plan_storage accepts for --layout.
        auto is_layout_ooc = [](const std::string& s) {
            std::string v = s;
            for (auto& c : v) c = (char)std::tolower((unsigned char)c);
            return v == "ooc" || v == "outofcore" || v == "out-of-core";
        };
        const bool ooc_forced = (ooc_cli == "on") || is_layout_ooc(layout_cli);

        if (have_dnb) {
            // Pre-load: we do not yet know the per-mode kernel_class[],
            // so assume the worst-case of any DenseFiber mode existing.
            // This adds a sort-scratch term (~NNZ * per_nnz bytes) to
            // each in-core layout's projected_need and keeps the pre-load
            // planner conservative -- otherwise a tensor like 335M nnz
            // would pick InPlace pre-load (6.3 GiB fits budget) and then
            // FATAL post-load because decide_layout's more accurate
            // accounting rejects it.  Erring on the side of OOC is the
            // right trade-off: worst case we run slightly slower on disk.
            StoragePlan plan = dyn_plan_storage(hdr.nnz, hdr.num_modes,
                                                hdr.mode_size,
                                                /*rank_padded_est=*/args.rank > 0
                                                    ? args.rank : 16,
                                                /*has_dense_fiber_hint=*/true,
                                                layout_cli, ooc_cli);
            dyn_print_plan(plan, "[plan]");

            use_ooc = (plan.choice == Layout::OutOfCore);
            if (use_ooc) {
                T.num_modes = hdr.num_modes;
                T.nnz       = hdr.nnz;
                for (int n = 0; n < hdr.num_modes; ++n)
                    T.mode_size[n] = hdr.mode_size[n];
                T.ooc_header_bytes = hdr.ooc_header_bytes;
            }
        } else if (ooc_forced) {
            std::fprintf(stderr,
                "[main] OOC requested but .dnb cache is missing: %s\n"
                "       Run a small in-core load first (any rank) to "
                "build the cache, or pre-generate it offline.\n",
                dnb_path.c_str());
            return 2;
        }
        // No .dnb + not forcing OOC -> normal text-load path below.
    }

    if (runtime_skip_preprocess) {
        // Runtime-only mode: T is fully populated from the .dnp file.
        // Nothing to load, nothing to build.  Fall through to the
        // thread-count resolution + CP-ALS block below.
    } else if (use_ooc) {
        std::printf("[main] OUT-OF-CORE mode engaged\n"
                    "       tensor:   %s\n"
                    "       nnz:      %llu  modes: %d\n"
                    "       dnb file: %s\n",
                    args.tensor_path.c_str(),
                    (unsigned long long)T.nnz, T.num_modes,
                    dnb_path.c_str());

        T.layout        = Layout::OutOfCore;
        T.ooc_enabled   = true;
        T.ooc_path      = dnb_path;
        T.ooc_chunk_nnz = args.ooc_chunk ? args.ooc_chunk
                                         : ooc_default_chunk_nnz(T);

        double norm_pass_s = 0.0;
#ifdef _OPENMP
        const double t_norm0 = omp_get_wtime();
#else
        const auto t_norm0 = std::chrono::steady_clock::now();
#endif
        if (!ooc_precompute_norm(T)) {
            std::fprintf(stderr, "[main] OOC: ||T||^2 precompute failed\n");
            return 2;
        }
#ifdef _OPENMP
        norm_pass_s = omp_get_wtime() - t_norm0;
#else
        norm_pass_s = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t_norm0)
                          .count();
#endif
        std::printf("       chunk:    %llu nnz (~%.1f MiB)\n"
                    "       ||T||:    %.6e  (streaming pass %.3f s)\n",
                    (unsigned long long)T.ooc_chunk_nnz,
                    (double)(T.ooc_chunk_nnz *
                             (sizeof(value_t) + T.num_modes * sizeof(idx_t)))
                        / (double)(1ULL << 20),
                    std::sqrt(T.ooc_tnorm_sq),
                    norm_pass_s);
    } else {
        if (!load_coo_tensor(args.tensor_path, T,
                             /*cache_on_success=*/!args.no_cache))
            return 2;
    }

    // ---- resolve thread count up front so build_flycoo and the parallel
    //      algorithm agree on a single value ----
    int nthreads = args.threads;
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#else
    nthreads = 1;
#endif
    if (nthreads < 1) nthreads = 1;
    args.threads = nthreads;                // feed the resolved value back

    // Map --layout onto the DYN_LAYOUT env var so build_flycoo's
    // decide_layout() picks it up.  "auto" keeps the heuristic in charge.
    //
    // --layout morton is a post-processing conversion: the preprocessor
    // first picks InPlace (smallest SoA footprint), and after it runs we
    // call build_morton_layout() to transform SoA -> Morton and free the
    // SoA slabs.  So map "morton" -> "inplace" for the preprocessor and
    // remember the intent in `wants_morton`.
    bool wants_morton = false;
    {
        std::string layout_lc; layout_lc.reserve(args.layout.size());
        for (char c : args.layout)
            layout_lc.push_back((char)std::tolower((unsigned char)c));
        if (layout_lc == "morton") {
            wants_morton = true;
            // Tell the preprocessor to use InPlace (most compact SoA).
#ifdef _WIN32
            _putenv_s("DYN_LAYOUT", "inplace");
#else
            setenv("DYN_LAYOUT", "inplace", /*overwrite=*/1);
#endif
        } else if (!args.layout.empty() && layout_lc != "auto") {
#ifdef _WIN32
            _putenv_s("DYN_LAYOUT", args.layout.c_str());
#else
            setenv("DYN_LAYOUT", args.layout.c_str(), /*overwrite=*/1);
#endif
        }
    }

    // --ncopy-fiber-sort -> DYN_NCOPY_FIBER_SORT=0/1.  "auto" leaves the
    // env unset so the preprocess default (= ON) takes effect.  The env
    // var itself only cares about the "0" literal for the opt-out path.
    if (args.ncopy_fiber_sort == "on") {
#ifdef _WIN32
        _putenv_s("DYN_NCOPY_FIBER_SORT", "1");
#else
        setenv("DYN_NCOPY_FIBER_SORT", "1", 1);
#endif
    } else if (args.ncopy_fiber_sort == "off") {
#ifdef _WIN32
        _putenv_s("DYN_NCOPY_FIBER_SORT", "0");
#else
        setenv("DYN_NCOPY_FIBER_SORT", "0", 1);
#endif
    } else if (args.ncopy_fiber_sort != "auto") {
        std::fprintf(stderr,
            "warning: --ncopy-fiber-sort: unknown value '%s' (expected "
            "auto|on|off); ignoring.\n",
            args.ncopy_fiber_sort.c_str());
    }

    // --ncopy-csr -> DYN_NCOPY_CSR=0/1.  Same convention as fiber-sort:
    // "auto" leaves env unset (preprocess default = ON), explicit on/off
    // force the env var, anything else warns.
    if (args.ncopy_csr == "on") {
#ifdef _WIN32
        _putenv_s("DYN_NCOPY_CSR", "1");
#else
        setenv("DYN_NCOPY_CSR", "1", 1);
#endif
    } else if (args.ncopy_csr == "off") {
#ifdef _WIN32
        _putenv_s("DYN_NCOPY_CSR", "0");
#else
        setenv("DYN_NCOPY_CSR", "0", 1);
#endif
    } else if (args.ncopy_csr != "auto") {
        std::fprintf(stderr,
            "warning: --ncopy-csr: unknown value '%s' (expected "
            "auto|on|off); ignoring.\n",
            args.ncopy_csr.c_str());
    }

    // --ncopy-csr-compact -> DYN_NCOPY_CSR_COMPACT=0/1.  Same convention
    // as fiber-sort / csr.  "auto" + --verify -> force 0; "auto" otherwise
    // -> leave env unset (default = ON in preprocess).
    if (args.ncopy_csr_compact == "on") {
#ifdef _WIN32
        _putenv_s("DYN_NCOPY_CSR_COMPACT", "1");
#else
        setenv("DYN_NCOPY_CSR_COMPACT", "1", 1);
#endif
    } else if (args.ncopy_csr_compact == "off") {
#ifdef _WIN32
        _putenv_s("DYN_NCOPY_CSR_COMPACT", "0");
#else
        setenv("DYN_NCOPY_CSR_COMPACT", "0", 1);
#endif
    } else if (args.ncopy_csr_compact == "auto") {
        // All-modes driver streams T.vals / T.idx_buf[*] (all N columns
        // of the primary slab) in one pass.  CSR compaction frees the
        // primary under NCopy, which would null T.vals.  Disable compact
        // whenever we know compact would break the kernel:
        //   (a) --verify: reference reads T.vals directly.
        //   (b) --all-modes (or DYN_ALL_MODES=1): same reason.
        bool am_cli = args.all_modes;
        if (!am_cli) {
            const char* am_env = std::getenv("DYN_ALL_MODES");
            if (am_env && am_env[0] && am_env[0] != '0') am_cli = true;
        }
        if (args.verify) {
            std::printf("[main] --verify active  -> auto-disabling CSR "
                        "slab compaction (preserves primary/scratch slabs "
                        "for reference kernel).\n");
#ifdef _WIN32
            _putenv_s("DYN_NCOPY_CSR_COMPACT", "0");
#else
            setenv("DYN_NCOPY_CSR_COMPACT", "0", 1);
#endif
        } else if (am_cli) {
            std::printf("[main] --all-modes active  -> auto-disabling CSR "
                        "slab compaction (primary slab must stay intact "
                        "so the all-modes kernel can stream all N idx "
                        "columns from a single view).\n");
#ifdef _WIN32
            _putenv_s("DYN_NCOPY_CSR_COMPACT", "0");
#else
            setenv("DYN_NCOPY_CSR_COMPACT", "0", 1);
#endif
        }
    } else {
        std::fprintf(stderr,
            "warning: --ncopy-csr-compact: unknown value '%s' (expected "
            "auto|on|off); ignoring.\n",
            args.ncopy_csr_compact.c_str());
    }

    // ---- build FLYCOO ----
    //
    //  OOC skips all preprocessing: there is no in-RAM slab to FLYCOO-sort
    //  or physically re-copy (that's the whole point of OOC).  We only
    //  need T.num_modes / T.nnz / T.mode_size[] populated, which the
    //  header read above already did.
    //
    //  Runtime-only mode (--runtime) also skips this entirely: the .dnp
    //  loader has already populated every FLYCOO / Morton / CSR field the
    //  kernels will read.
    if (!use_ooc && !runtime_skip_preprocess) {
        build_flycoo(T, nthreads, nullptr, (idx_t)args.shard_size);

        // Stage 2: if the user asked for --layout morton, convert the
        // SoA InPlace slab into the 12 B/nnz Morton layout.  build_morton_layout
        // packs keys (via BMI2 pdep), sorts by key ascending (Z-order), and
        // frees the SoA slabs.  Only the Jacobi CP-ALS path has a Morton
        // driver -- cpals() promotes GS -> Jacobi automatically if needed.
        if (wants_morton) {
            if (args.verify) {
                std::fprintf(stderr,
                    "[main] --layout morton is incompatible with --verify "
                    "(reference kernel reads the SoA slab that Morton "
                    "frees).  Disable one or the other.\n");
                return 7;
            }
            if (!build_morton_layout(T, nthreads)) {
                std::fprintf(stderr,
                    "[main] --layout morton: build_morton_layout failed; "
                    "staying on InPlace.\n");
            }
        }

        // ----------------------------------------------------------------
        //  Dynasor+ framework: persist the fully-preprocessed tensor when
        //  the user passed --save-processed.  This produces a .dnp file
        //  that subsequent runs can load via --runtime to skip the entire
        //  preprocessing pipeline.  OOC tensors are excluded (their .dnb
        //  cache is already the runtime-only format).
        // ----------------------------------------------------------------
        if (!args.save_processed.empty()) {
#ifdef _OPENMP
            const double tsav0 = omp_get_wtime();
#else
            const auto tsav0 = std::chrono::steady_clock::now();
#endif
            if (!save_dnp_tensor(args.save_processed, T, nthreads)) {
                std::fprintf(stderr,
                    "[main] --save-processed: failed to write '%s'.\n",
                    args.save_processed.c_str());
                return 8;
            }
#ifdef _OPENMP
            const double tsav_s = omp_get_wtime() - tsav0;
#else
            const double tsav_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tsav0).count();
#endif
            std::printf("[dnp] wrote processed tensor -> %s  (%.3f s)\n",
                        args.save_processed.c_str(), tsav_s);
        }
    } else if (!args.save_processed.empty()) {
        std::fprintf(stderr,
            "[main] --save-processed ignored (%s).\n",
            use_ooc ? "OOC tensor -- .dnb already serves as the runtime-only format"
                    : "tensor was loaded via --runtime");
    }

    // ======================================================================
    //  Mode A: CP-ALS decomposition
    // ======================================================================
    if (args.decompose == "cpals") {
        FactorMatrices F;  F.allocate(T.num_modes, T.mode_size, args.rank);

        CpOptions opt;
        opt.max_iters   = args.cp_iters;
        opt.tol         = args.cp_tol;
        opt.seed        = args.seed;
        opt.num_threads = args.threads;
        opt.ridge            = (value_t)args.cp_ridge;
        opt.verbose          = true;
        opt.init_dump_prefix = args.cp_dump_init;

        // --cp-kernel (or DYN_CP_KERNEL env) picks the MTTKRP strategy.
        // Accepted aliases:
        //   "auto" (default)                                 -> planner picks
        //   "gauss-seidel" | "gs" | "classical" | "per-mode" -> GaussSeidel
        //   "jacobi"       | "js" | "all-modes"              -> Jacobi
        //
        //  Stage 4: the entire decision tree lives inside
        //  dyn_build_compute_plan.  Here we just pass the CLI/env strings
        //  through and consume the kernel_enum output.  The banner shows
        //  the full plan (layout + kernel + per-mode + JIT) via
        //  dyn_print_compute_plan so behaviour is transparent.
        {
            int nt = opt.num_threads;
            if (nt <= 0) {
#ifdef _OPENMP
                nt = omp_get_max_threads();
#else
                nt = 1;
#endif
            }

            ComputePlan cp = dyn_build_compute_plan(
                T, F.rank_padded, nt,
                args.layout, args.ooc, args.cp_kernel);

            // In runtime-only mode the layout is baked into the .dnp
            // file -- the planner's hypothetical "choice" is misleading.
            // Rewrite the banner fields to reflect the frozen decision.
            if (runtime_skip_preprocess) {
                cp.storage.choice          = T.layout;
                cp.storage.reason          = "loaded from .dnp";
                cp.storage.override_source = "dnp-runtime";
            }

            // Apply the plan's recommendations BEFORE printing so the
            // banner reflects the actual runtime state the kernels see.
            if (cp.pf_far) {
#if defined(_WIN32)
                _putenv_s("DYN_PF_FAR", "1");
#else
                setenv("DYN_PF_FAR", "1", 1);
#endif
            }
            if (cp.use_jit) dyn_jit_force_enable(true);

            // Re-evaluate the "effective" flags for the banner: --jit /
            // DYN_JIT may have turned JIT on even if the plan heuristics
            // said the current shape wouldn't benefit much.  The user's
            // explicit opt-in always wins, so the banner reports the
            // global runtime state rather than just the plan advice.
            cp.use_jit = dyn_jit_enabled();

            dyn_print_compute_plan(cp, "[plan]");

            opt.kernel = (cp.kernel_enum == 1) ? CpKernel::Jacobi
                                               : CpKernel::GaussSeidel;
            std::printf("[cpals] kernel -> %s  (reason=%s; src=%s)\n",
                        opt.kernel == CpKernel::Jacobi
                            ? "jacobi (all-modes single-pass)"
                            : "gauss-seidel (classical per-mode)",
                        cp.kernel_reason, cp.kernel_source);
        }

        CpResult res = cpals(T, F, opt);

        if (!args.cp_dump_init.empty()) {
            std::printf("[io] wrote initial factors to %s.cp_init_fm<n>.bin\n",
                        args.cp_dump_init.c_str());
        }

        // Optional factor + lambda dump -- raw little-endian float32, so
        // downstream tools (numpy.fromfile) can reshape to (I_n, rank).
        if (!args.cp_dump.empty()) {
            for (int n = 0; n < T.num_modes; ++n) {
                std::string p = args.cp_dump + ".cp_fm" + std::to_string(n) + ".bin";
                if (!dump_factor_matrix_bin(p, F.Y[n], T.mode_size[n],
                                            F.rank, F.rank_padded))
                    std::fprintf(stderr, "[io] WARN: failed to write %s\n",
                                 p.c_str());
            }
            std::string lpath = args.cp_dump + ".cp_lambda.bin";
            if (FILE* fp = std::fopen(lpath.c_str(), "wb")) {
                std::fwrite(res.lambda.data(), sizeof(value_t),
                            (size_t)args.rank, fp);
                std::fclose(fp);
            }
            std::printf("[io] wrote CP factors to %s.cp_fm<n>.bin and lambda to %s\n",
                        args.cp_dump.c_str(), lpath.c_str());
        }

        std::printf("[summary] cpals iters=%d  final_fit=%.6f  residual=%.3e  "
                    "total=%.3f s  spMTTKRP=%.3f s  linalg=%.3f s\n",
                    res.iters_done, res.final_fit, res.final_residual,
                    res.total_time_s, res.mttkrp_time_s, res.linalg_time_s);

        F.free_all();
        return (res.iters_done > 0) ? 0 : 4;
    }
    if (!args.decompose.empty()) {
        std::fprintf(stderr, "[main] unknown --decompose algorithm: %s "
                             "(supported: cpals)\n", args.decompose.c_str());
        return 5;
    }

    // OOC only supports CP-ALS (no --decompose or unsupported --decompose
    // is already rejected above).  If we reach the standalone MTTKRP
    // benchmark below with use_ooc true, T.vals / T.idx_buf are nullptr
    // and the reference + per-mode drivers would segfault.  Guard it.
    if (use_ooc) {
        std::fprintf(stderr,
            "[main] --ooc requires --decompose cpals; the standalone MTTKRP "
            "benchmark has no streaming driver.\n");
        return 6;
    }

    // ======================================================================
    //  Mode B: single-sweep MTTKRP benchmark (original behaviour)
    // ======================================================================
    FactorMatrices F_dyn;  F_dyn .allocate(T.num_modes, T.mode_size, args.rank);
    FactorMatrices F_ref;  F_ref .allocate(T.num_modes, T.mode_size, args.rank);

    // identical random init in Y (the inputs)
    std::mt19937_64 rng(args.seed);
    for (int n = 0; n < T.num_modes; ++n) {
        random_init(F_dyn.Y[n], T.mode_size[n], args.rank, F_dyn.rank_padded, rng);
    }
    // copy Y to the reference set
    for (int n = 0; n < T.num_modes; ++n) {
        std::memcpy(F_ref.Y[n], F_dyn.Y[n],
                    (size_t)T.mode_size[n] * F_dyn.rank_padded * sizeof(value_t));
    }

    // ---- run reference first (so Dynasor can mutate T.elements) ----
    if (args.verify) {
        auto t0 = std::chrono::steady_clock::now();
        spmttkrp_reference(T, F_ref);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[reference] elapsed = %.6f s\n", secs);
    }

    // ---- run Dynasor ----
    //
    // --all-modes (or DYN_ALL_MODES=1) switches from the per-target-mode
    // sweep (N tensor passes per iter) to the all-modes single-pass
    // driver (1 tensor pass per iter with per-thread ofibs).  On medium
    // tensors this is the optimization that lets Dynasor beat ALTO.
    bool use_all_modes = args.all_modes;
    if (!use_all_modes) {
        const char* am_env = std::getenv("DYN_ALL_MODES");
        if (am_env && am_env[0] && am_env[0] != '0') use_all_modes = true;
    }
    double dyn_secs = use_all_modes
        ? spmttkrp_all_modes_dynasor(T, F_dyn, args.threads)
        : spmttkrp_dynasor          (T, F_dyn, args.threads);
    (void)use_all_modes;

    // ---- dump factor matrices if requested ----
    if (!args.dump_prefix.empty()) {
        const char* ext = args.dump_binary ? ".bin" : ".txt";
        for (int n = 0; n < T.num_modes; ++n) {
            std::string p = args.dump_prefix + ".fm" + std::to_string(n) + "_out" + ext;
            bool ok = args.dump_binary
                ? dump_factor_matrix_bin(p, F_dyn.Yhat[n], T.mode_size[n],
                                         F_dyn.rank, F_dyn.rank_padded)
                : dump_factor_matrix    (p, F_dyn.Yhat[n], T.mode_size[n],
                                         F_dyn.rank, F_dyn.rank_padded);
            if (!ok) std::fprintf(stderr, "[io] WARN: failed to write %s\n", p.c_str());
        }
        std::printf("[io] wrote Yhat[*] to %s.fm<n>_out%s\n",
                    args.dump_prefix.c_str(), ext);
    }

    // ---- verification ----
    int rc = 0;
    if (args.verify) {
        double err = compare_factors_detail(F_dyn, F_ref, T.mode_size);
        const double tol = 1e-4;
        if (err > tol) {
            std::printf("FAIL: Frobenius relative error %.3e exceeds %.1e\n",
                        err, tol);
            rc = 3;
        } else {
            std::printf("PASS: Frobenius relative error %.3e within %.1e\n",
                        err, tol);
        }
    }

    std::printf("[summary] dynasor = %.6f s (%.3f Mnnz/s)\n",
                dyn_secs,
                (double)T.nnz / 1e6 / dyn_secs);

    F_dyn.free_all();
    F_ref.free_all();
    return rc;
}
