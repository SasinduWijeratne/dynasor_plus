# Dynasor+ : CP Tensor Decomposition Framework

A portable, multi-core-CPU framework for **CP (CANDECOMP/PARAFAC) tensor
decomposition** on large sparse tensors. Built on top of the FLYCOO layout
and greedy LPT scheduling introduced in

> S. Wijeratne, R. Kannan, V. Prasanna.
> *Dynasor: A Dynamic Memory Layout for Accelerating Sparse MTTKRP for Tensor
> Decomposition on Multi-core CPU*. arXiv:2309.09131, 2023.

Dynasor+ extends the original Dynasor kernel into a complete framework with:

- **Four in-core layouts** chosen automatically from tensor size + RAM
  budget: `PingPong`, `NCopy`, `InPlace`, `Morton`.
- **Out-of-core streaming** (`OOC`) for tensors that exceed RAM.
- **A unified `ComputePlan`** that picks the layout, the CP-ALS inner kernel
  (Gauss-Seidel vs all-modes Jacobi), and JIT specialization.
- **Two execution modes** (the focus of this framework):
  1. **Complete processing + runtime**  --  load raw `.tns`/`.dnb`, run the
     full preprocessing pipeline, then CP-ALS. Optionally persist the
     fully-processed state via `--save-processed`.
  2. **Runtime only**  --  load a previously saved `.dnp` file via
     `--runtime` and jump straight into CP-ALS, skipping every preprocessing
     step.

A portable SIMD layer targets AVX-512F / AVX2+FMA / SSE2 / NEON / SVE with a
scalar fallback. The compiled backend is printed at startup.

---

## 1. Build

### POSIX (Linux / macOS / WSL)

```bash
make                        # native
make ISA=znver5             # AMD Zen 5
make ISA=neoverse-v2        # Arm Neoverse V2 (Graviton4 / Grace)
make ISA=avx512
make ISA=avx2
make ISA=neon
make LTO=1 ISA=znver5       # whole-program LTO (+3-8%)
make clean
```

Or via the wrapper: `./scripts/build.sh [isa|clean]`.

Requirements: `g++` or `clang++` with OpenMP (`-fopenmp`), C++17.

### Windows (MSYS2 / MinGW-w64)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1              # native
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -ISA avx2
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Clean
```

The binary lands in `build/dynasor_plus[.exe]`.

---

## 2. Two execution modes

### Mode 1 -- Complete processing + runtime (default)

Point the binary at a raw `.tns` (or cached `.dnb`). Dynasor+ parses it,
plans a layout, builds FLYCOO / Morton / NCopy-CSR metadata, and runs
CP-ALS.

```bash
# auto-planned layout (recommended)
./build/dynasor_plus data/synth_cp_3d_r10_noisy.tns \
    --decompose cpals --rank 16 --threads 16 --cp-iters 50

# force a specific layout
./build/dynasor_plus data/my_tensor.tns \
    --layout inplace \
    --decompose cpals --rank 32 --threads 32 --cp-iters 50
```

Add `--save-processed PATH.dnp` to serialize the fully-preprocessed tensor
state (layout buffers + FLYCOO schedule + per-thread cursors + Morton keys
+ NCopy CSR rowptrs) so subsequent runs can skip preprocessing:

```bash
./build/dynasor_plus data/my_tensor.tns \
    --layout inplace --threads 32 \
    --save-processed data/my_tensor.inplace.dnp \
    --decompose cpals --rank 32 --cp-iters 50
```

Valid `--layout` values: `auto` (default), `morton`, `ncopy`, `pingpong`,
`inplace`, `ooc`. `NCopy` must be saved with `--ncopy-csr-compact off`
(compacted copies free the slabs needed for reload).

### Mode 2 -- Runtime only

Load an existing `.dnp` with `--runtime`. No parsing, no FLYCOO rebuild, no
Morton key generation -- just CP-ALS on the already-resident layout.

```bash
./build/dynasor_plus data/my_tensor.inplace.dnp --runtime \
    --decompose cpals --rank 32 --threads 32 --cp-iters 50
```

The thread count baked into the `.dnp` at save time is part of the FLYCOO
schedule. You must load it with the same `--threads` value; mismatches are
rejected with an actionable message.

### Mode selector

| Goal                                     | Flags                                     |
|------------------------------------------|-------------------------------------------|
| One-off decomposition                    | *(none of the framework flags)*           |
| Save processed state for deployment      | `--save-processed PATH.dnp`               |
| Re-run a previously saved tensor fast    | `--runtime` (pass the `.dnp` as input)    |
| Tensor bigger than RAM                   | `--ooc on` (streams from `.dnb` cache)    |

---

## 3. Binary formats

| Ext    | What it stores                                         | Used by              |
|--------|--------------------------------------------------------|----------------------|
| `.tns` | FROSTT text (1-based indices + value)                  | any loader           |
| `.dnb` | Raw COO binary cache (magic `DNB1`)                    | fast re-parse; OOC   |
| `.dnp` | **Fully-processed** tensor state (magic `DNP1`)        | `--runtime` mode     |

A `.dnb` sits alongside the source `.tns` and is auto-built on first parse
(~50-100x faster on re-load). A `.dnp` is created explicitly via
`--save-processed` and contains everything needed to skip preprocessing for
the specific layout + thread count it was saved with.

See `include/dynasor_dnp.h` for the `.dnp` on-disk layout.

---

## 4. Key CLI options

```
Usage: dynasor_plus <tensor.tns | tensor.dnb | tensor.dnp> [options]

Core:
  --rank R                 factor-matrix rank (default 16)
  --threads T              OpenMP threads (default: system max)
  --verify                 run reference kernel + compare (MTTKRP path)
  --decompose cpals        run CP-ALS (vs standalone MTTKRP benchmark)
  --cp-iters N             max ALS iterations (default 50)
  --cp-tol TOL             fit-change convergence threshold (default 1e-4)

Layout / kernel:
  --layout L               auto | morton | ncopy | pingpong | inplace | ooc
  --cp-kernel K            auto | gauss-seidel | jacobi
  --ooc MODE               auto | on | off           (CP-ALS only)
  --ooc-chunk NNZ          streaming window size
  --ncopy-csr-compact M    auto | on | off           (must be off to save .dnp)
  --jit                    JIT-specialize the fiber_csr kernel (cached)

Framework:
  --save-processed PATH    after preprocessing, persist state to PATH (.dnp)
  --runtime                treat <tensor_path> as a .dnp from a previous save
```

Full help: `./build/dynasor_plus --help`.

---

## 5. Repository layout

```
Dynasor_plus/
  include/           public headers (dynasor_common, _simd, _dnp, _cpals, ...)
  src/               core implementation (IO, preprocess, kernels, CP-ALS,
                     Morton, OOC, JIT, DNP serializer, main)
  scripts/           build wrappers + test harness + synthetic tensor gen
  tools/             offline helpers (dnb <-> tns, Python validators)
  competitor/        optional third-party reference implementations used
                     for benchmarking (not part of the framework build, not
                     vendored in this repo -- clone upstream separately)
  data/              small smoke-test tensors (most benchmark tensors are
                     .gitignore'd -- bring your own FROSTT data)
  Makefile           portable build with ISA auto-detect
  README.md
```

---

## 6. Correctness

```bash
# smoke test a tensor
./build/dynasor_plus data/synth_cp_r5.tns --rank 8 --threads 4 --verify

# bundled harness: 4 tensors x {rank,threads} sweep
./scripts/run_tests.sh                                    # POSIX
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1   # Windows
```

CP-ALS fit / residual on a `.dnp` runtime load matches the full-pipeline
run to within float32 round-off (per-iteration thread-order reductions may
differ by 1-2 ulp).

---

## 7. References

- S. Wijeratne, R. Kannan, V. Prasanna. *Dynasor: A Dynamic Memory Layout
  for Accelerating Sparse MTTKRP for Tensor Decomposition on Multi-core
  CPU*. arXiv:2309.09131, 2023.
- FROSTT: The Formidable Repository of Open Sparse Tensors & Tools,
  <http://frostt.io/>
