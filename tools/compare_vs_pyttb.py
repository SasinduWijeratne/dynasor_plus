"""
compare_vs_pyttb.py
-------------------

Cross-validate Dynasor+'s CP-ALS against Kolda's Python Tensor Toolbox
(`pyttb`), which is the official Sandia port of the MATLAB Tensor
Toolbox `cp_als`.

Pipeline
~~~~~~~~
1. Load a FROSTT `.tns` sparse tensor into a `pyttb.sptensor`.
2. Run `pyttb.cp_als` with matching rank / max-iters / tolerance.
3. Run the Dynasor+ C++ binary on the same tensor with matching
   parameters, dumping factor matrices + lambda.
4. Compare:
     * final fit (scalar, meaningful regardless of CP ambiguities)
     * ``‖T − M‖_F``
     * optionally: factor alignment via column-matching cosine similarity
       (Kruskal ambiguity: permutation + sign + per-column scale)

Running
~~~~~~~
The Dynasor+ side is invoked directly with `subprocess`, so only the path
to the C++ binary needs to be supplied.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyttb as ttb


# ----------------------------------------------------------------------
#  Tensor I/O
# ----------------------------------------------------------------------
def load_tns_as_sptensor(path: Path) -> ttb.sptensor:
    """Read a FROSTT COO .tns file into a pyttb.sptensor (1-indexed -> 0).

    Uses np.loadtxt for the heavy lifting instead of Python lists, so 30 M
    nnz takes ~2-3 GiB peak instead of ~10-15 GiB."""
    try:
        raw = np.loadtxt(str(path), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None, :]
        N = raw.shape[1] - 1
        subs_np = raw[:, :N].astype(np.int64) - 1
        vals_np = raw[:, N].astype(np.float64).reshape(-1, 1)
        shape = tuple(int(subs_np[:, k].max()) + 1 for k in range(N))
        return ttb.sptensor(subs_np, vals_np, shape)
    except Exception:
        # Fallback for malformed files.
        pass
    subs: list[list[int]] = []
    vals: list[float] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # last token is the value, first N are indices (1-indexed)
            idx = [int(x) - 1 for x in parts[:-1]]
            val = float(parts[-1])
            subs.append(idx)
            vals.append(val)
    subs_np = np.asarray(subs, dtype=np.int64)
    vals_np = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
    shape = tuple(int(subs_np[:, k].max()) + 1 for k in range(subs_np.shape[1]))
    return ttb.sptensor(subs_np, vals_np, shape)


def load_cp_dump(prefix: str, num_modes: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Load C++ CP-ALS dumped factors + lambda.

    The dump format is a 24-byte binary header (see
    `dump_factor_matrix_bin` in src/dynasor_io.cpp) followed by
    row-major float32 data of shape (rows, rank_padded).  We only read
    the first `rank` columns -- trailing padding columns are zeroed.
    """
    # Header (24 bytes total):
    #   magic[4]='DYFM', version u32, value_bytes u32, rank u32, rows u64
    # Data: row-major, `rank` columns (padding stripped on the C++ side).
    def _read_factor(path: Path) -> np.ndarray:
        with path.open("rb") as fh:
            header = fh.read(24)
            if header[:4] != b"DYFM":
                raise RuntimeError(f"bad magic in {path}: {header[:4]!r}")
            _version, value_bytes, rank = \
                np.frombuffer(header[4:16], dtype=np.uint32)
            rows = int(np.frombuffer(header[16:24], dtype=np.uint64)[0])
            if value_bytes != 4:
                raise RuntimeError(
                    f"expected float32 factors, got value_bytes={value_bytes}")
            raw = np.frombuffer(fh.read(), dtype=np.float32)
        expected = rows * int(rank)
        if raw.size != expected:
            raise RuntimeError(
                f"{path}: expected {expected} floats, got {raw.size}")
        return raw.reshape(rows, int(rank)).astype(np.float64)

    factors = [_read_factor(Path(f"{prefix}.cp_fm{n}.bin"))
               for n in range(num_modes)]
    # Lambda is dumped as raw float32[rank] (no header -- see main.cpp).
    lam = np.fromfile(Path(f"{prefix}.cp_lambda.bin"),
                      dtype=np.float32).astype(np.float64)
    return factors, lam


# ----------------------------------------------------------------------
#  Factor alignment (Kruskal ambiguity aware)
# ----------------------------------------------------------------------
def align_columns(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy permutation + sign alignment of columns of B onto A.

    Returns (perm, signs) such that B[:, perm] * signs ≈ A (modulo scale).
    Uses absolute cosine similarity as the matching score and a greedy
    selection (sufficient for R <= 256 and enough to surface outright
    recovery failures; a full Hungarian match would be identical here
    when factors are well-separated).
    """
    R = A.shape[1]
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-30)
    Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
    sim = An.T @ Bn                       # (R, R), signed cosine
    abs_sim = np.abs(sim)

    perm = -np.ones(R, dtype=np.int64)
    signs = np.ones(R, dtype=np.float64)
    used = np.zeros(R, dtype=bool)
    for r in range(R):
        # match the A-column with the best remaining candidate
        idx = np.argsort(abs_sim[r])[::-1]
        for j in idx:
            if not used[j]:
                perm[r] = j
                signs[r] = np.sign(sim[r, j]) if sim[r, j] != 0 else 1.0
                used[j] = True
                break
    return perm, signs


def factor_similarity(truth_factors: list[np.ndarray],
                      est_factors: list[np.ndarray]) -> float:
    """Mean absolute cosine similarity across modes after greedy column
    alignment.  1.0 iff the two CP decompositions are identical modulo
    permutation + sign + column scale.
    """
    N = len(truth_factors)
    sims: list[float] = []
    perm, signs = align_columns(truth_factors[0], est_factors[0])
    for n in range(N):
        A = truth_factors[n]
        B = est_factors[n][:, perm] * signs[np.newaxis, :]
        An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-30)
        Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
        sims.append(float(np.mean(np.abs(np.einsum("ir,ir->r", An, Bn)))))
    return float(np.mean(sims))


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", required=True, type=Path)
    p.add_argument("--binary", default="build/dynasor_plus.exe",
                   help="Path to dynasor_plus binary (Windows .exe or ELF)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dump-prefix", default="run_xcheck")
    p.add_argument("--keep-dump", action="store_true",
                   help="Keep the dumped C++ factors after the run.")
    args = p.parse_args()

    print("=" * 78)
    print(f"Cross-check: {args.tensor.name}  rank={args.rank}  "
          f"iters<={args.iters}  tol={args.tol:.1e}  seed={args.seed}")
    print("=" * 78)

    # --- Load tensor ------------------------------------------------------
    print(f"[pyttb] loading {args.tensor} ...")
    t0 = time.perf_counter()
    T_sp = load_tns_as_sptensor(args.tensor)
    print(f"[pyttb] loaded sptensor shape={T_sp.shape}  "
          f"nnz={T_sp.vals.size}  load_time={time.perf_counter() - t0:.3f}s")

    # --- Dynasor+ path FIRST: generates both the solution AND the initial
    #     factors, so pyttb can start from the exact same point.  CP-ALS is
    #     non-convex, so comparing two runs with *different* random inits
    #     measures the optimization landscape, not algorithmic equivalence.
    prefix = args.dump_prefix
    for p_ in Path(".").glob(f"{prefix}.cp_*"):
        p_.unlink()

    cmd = [
        str(Path(args.binary).resolve()),
        str(args.tensor),
        "--decompose", "cpals",
        "--rank", str(args.rank),
        "--cp-iters", str(args.iters),
        "--cp-tol", str(args.tol),
        "--seed", str(args.seed),
        "--cp-dump", prefix,
        "--cp-dump-init", prefix,
    ]
    print(f"[dynasor] launching: {' '.join(cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t_dyn = time.perf_counter() - t0
    if r.returncode != 0:
        print("[dynasor] STDOUT:\n" + r.stdout)
        print("[dynasor] STDERR:\n" + r.stderr)
        return 1

    fit_dyn = None
    resid_dyn = None
    iters_dyn = None
    for line in r.stdout.splitlines():
        if line.startswith("[summary] cpals"):
            for tok in line.split():
                if tok.startswith("iters="):
                    iters_dyn = int(tok.split("=", 1)[1])
                elif tok.startswith("final_fit="):
                    fit_dyn = float(tok.split("=", 1)[1])
                elif tok.startswith("residual="):
                    resid_dyn = float(tok.split("=", 1)[1])
    if fit_dyn is None:
        print("[dynasor] failed to parse summary line; raw output:\n" + r.stdout)
        return 1
    print(f"[dynasor] done   iters={iters_dyn}  fit={fit_dyn:.8f}  "
          f"residual={resid_dyn:.6e}  wall={t_dyn:.3f}s")

    dyn_factors, dyn_lambda = load_cp_dump(prefix, T_sp.ndims)

    # Load the initial factors that Dynasor+ started from and hand them
    # to pyttb as a ktensor init, so both runs trace the same trajectory.
    init_factors: list[np.ndarray] = []
    for n in range(T_sp.ndims):
        p_init = Path(f"{prefix}.cp_init_fm{n}.bin")
        with p_init.open("rb") as fh:
            header = fh.read(24)
            if header[:4] != b"DYFM":
                raise RuntimeError(f"bad magic in {p_init}: {header[:4]!r}")
            _ver, _vb, rank_u = np.frombuffer(header[4:16], dtype=np.uint32)
            rows_u = int(np.frombuffer(header[16:24], dtype=np.uint64)[0])
            raw = np.frombuffer(fh.read(), dtype=np.float32)
        init_factors.append(
            raw.reshape(rows_u, int(rank_u)).astype(np.float64)
        )
    M_init = ttb.ktensor(init_factors, np.ones(args.rank, dtype=np.float64))

    # --- Reference path: Kolda's pyttb with matched init ------------------
    t0 = time.perf_counter()
    M_ref, _Minit, info = ttb.cp_als(
        T_sp, args.rank,
        stoptol=args.tol, maxiters=args.iters,
        init=M_init, printitn=0,
    )
    t_ref = time.perf_counter() - t0
    fit_ref = float(info["fit"])
    resid_ref = float(info["normresidual"])
    iters_ref = int(info["iters"])

    ref_factors = [np.asarray(M_ref.factor_matrices[n], dtype=np.float64)
                   for n in range(T_sp.ndims)]
    ref_lambda = np.asarray(M_ref.weights, dtype=np.float64)
    for n in range(T_sp.ndims):
        colnorms = np.linalg.norm(ref_factors[n], axis=0) + 1e-30
        ref_factors[n] = ref_factors[n] / colnorms
        ref_lambda = ref_lambda * colnorms

    print(f"[pyttb] done   iters={iters_ref}  fit={fit_ref:.8f}  "
          f"residual={resid_ref:.6e}  wall={t_ref:.3f}s  (init=Dynasor's)")

    # --- Comparison --------------------------------------------------------
    dfit = fit_dyn - fit_ref
    dres = resid_dyn - resid_ref
    sim = factor_similarity(ref_factors, dyn_factors)

    print("-" * 78)
    print(f"{'metric':<28}{'pyttb (Kolda)':>20}{'Dynasor+':>20}{'delta':>10}")
    print(f"{'iterations':<28}{iters_ref:>20d}{iters_dyn:>20d}"
          f"{iters_dyn - iters_ref:>+10d}")
    print(f"{'final fit':<28}{fit_ref:>20.8f}{fit_dyn:>20.8f}"
          f"{dfit:>+10.2e}")
    print(f"{'residual ||T - M||_F':<28}{resid_ref:>20.6e}{resid_dyn:>20.6e}"
          f"{dres:>+10.2e}")
    print(f"{'wall time (s)':<28}{t_ref:>20.3f}{t_dyn:>20.3f}"
          f"{t_dyn - t_ref:>+10.3f}")
    print(f"{'factor cos-sim (aligned)':<28}{sim:>20.6f}")
    print("-" * 78)

    # Pass/fail thresholds: fit within 1e-3 absolute is well inside
    # CP-ALS's own non-determinism from independent random inits.
    passed = abs(dfit) < 1e-3 or abs(dfit) < 0.01 * max(abs(fit_ref), 1e-6)
    if passed:
        print("[PASS] Dynasor+ matches Kolda's pyttb within tolerance.")
    else:
        print("[WARN] fit delta exceeds tolerance; inspect factor similarity.")

    if not args.keep_dump:
        for p_ in Path(".").glob(f"{prefix}.cp_*"):
            p_.unlink()

    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
