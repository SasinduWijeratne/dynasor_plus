"""
Independent check of CP-ALS output.

Given (1) a tensor .tns file and (2) a CP factor dump prefix written by
dynasor_plus' --cp-dump option, reconstruct the model

    M[i_0, ..., i_{N-1}] = sum_r lambda[r] * prod_n U_n[i_n, r]

and report the *true* residual ||T - M||_F / ||T||_F, computed in double
precision end-to-end.

Memory notes
------------
The previous version materialized the *dense* tensor T and the *dense* model
M.  For a 300x300x300x100 tensor that is 21 GiB per copy, so the checker
silently OOMs on large tensors.

The sparse path (default) only touches the nonzero entries of T:

    ||T||^2 = sum_k T[k]^2                                  -- O(nnz)
    <T, M>  = sum_k T[k] * sum_r lambda[r] * prod_n U_n[i_n(k), r]
                                                            -- O(nnz * R * N)
    ||M||^2 = lambda' * (H_0 * H_1 * ... * H_{N-1}) * lambda
              H_n = U_n' U_n                                -- O(R^2 * (sum I_n))

This needs O(R*N + nnz) extra floats and runs on tensors with billions of
nonzeros without OOM.

Use --dense to switch to the old code path (useful for cross-checking).
"""
import argparse
import os
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------
def load_tns_sparse(path):
    """Stream a FROSTT .tns and return (subs [nnz x N, int32], vals [nnz, f64],
    dims [N]).  Uses np.loadtxt for a single-pass parse."""
    # Discover N from the first non-empty line.
    with open(path) as fp:
        first = ""
        for line in fp:
            line = line.strip()
            if line:
                first = line
                break
    if not first:
        raise RuntimeError(f"{path} is empty")
    N = len(first.split()) - 1

    t0 = time.time()
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    subs = raw[:, :N].astype(np.int64) - 1  # 1-based -> 0-based
    vals = raw[:, N].astype(np.float64)
    dims = (subs.max(axis=0) + 1).astype(np.int64)
    t1 = time.time()
    print(f"[check] loaded {len(vals):,} nnz in {t1 - t0:.2f} s  "
          f"dims={list(dims)}  N={N}")
    return subs, vals, list(int(d) for d in dims)


def load_tns_dense(path):
    """Load FROSTT .tns into a dense numpy array (small tensors only)."""
    subs, vals, dims = load_tns_sparse(path)
    vol_bytes = np.prod(dims) * 8
    if vol_bytes > (4 << 30):
        print(f"[check] WARNING: dense allocation = {vol_bytes/(1<<30):.1f} GiB",
              file=sys.stderr)
    T = np.zeros(tuple(dims), dtype=np.float64)
    T[tuple(subs[:, n] for n in range(len(dims)))] = vals
    return T, dims


def load_cp_dump(prefix, dims, rank):
    """Read factor matrices + lambda written by Dynasor+."""
    N = len(dims)
    U = []
    for n in range(N):
        p = f"{prefix}.cp_fm{n}.bin"
        with open(p, "rb") as fp:
            header = fp.read(24)
            assert header[:4] == b"DYFM", f"bad magic in {p}: {header[:4]!r}"
            raw = fp.read()
        a = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
        a = a.reshape(dims[n], rank)
        U.append(a)
    lam = np.fromfile(f"{prefix}.cp_lambda.bin",
                      dtype=np.float32).astype(np.float64)
    return U, lam


# ---------------------------------------------------------------------------
#  Sparse residual: ||T - M||_F without ever materializing M
# ---------------------------------------------------------------------------
def sparse_residual(subs, vals, U, lam):
    """Return (Tnorm2, Mnorm2, inner, resid2) in double precision.

    M[i_0,...] = sum_r lam[r] * prod_n U_n[i_n, r]; this gathers rows of each
    U_n and multiplies column-wise, then reduces over r.  Peak memory is
    O(nnz * R) for the stacked row buffers, which for 30 M nnz and R=32 is
    30e6 * 32 * 8 = 7.7 GiB -- tight but feasible.  For larger cases we
    chunk.
    """
    N = len(U)
    R = lam.shape[0]
    nnz = vals.shape[0]

    Tnorm2 = float(np.dot(vals, vals))

    # Chunk to cap peak memory at ~256 MiB per chunk (nnz_per_chunk * R * 8).
    bytes_per_row = R * 8
    chunk = max(1, int((256 << 20) // max(bytes_per_row, 1)))
    inner = 0.0
    for s in range(0, nnz, chunk):
        e = min(s + chunk, nnz)
        # accum[k, r] = prod_n U_n[subs[k, n], r]
        accum = U[0][subs[s:e, 0], :].copy()
        for n in range(1, N):
            accum *= U[n][subs[s:e, n], :]
        # sum_r lam[r] * accum[k, r]
        Mk = accum @ lam
        inner += float(np.dot(vals[s:e], Mk))

    # ||M||^2 via the Gram / Khatri-Rao trick (exact, cheap).
    H = np.ones((R, R), dtype=np.float64)
    for Un in U:
        H *= (Un.T @ Un)
    Mnorm2 = float(lam @ H @ lam)

    resid2 = Tnorm2 - 2.0 * inner + Mnorm2
    return Tnorm2, Mnorm2, inner, resid2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True, help="FROSTT .tns path")
    ap.add_argument("--cp-dump", required=True,
                    help="prefix used with --cp-dump in dynasor_plus")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--dense", action="store_true",
                    help="Use the old dense code path (only for small tensors).")
    args = ap.parse_args()

    if args.dense:
        T, dims = load_tns_dense(args.tensor)
        U, lam = load_cp_dump(args.cp_dump, dims, args.rank)
        Tnorm2 = float((T * T).sum())
        N = len(U)
        letters = "abcdefgh"[:N]
        spec = f"{','.join(f'{L}r' for L in letters)},r->{letters}"
        M = np.einsum(spec, *U, lam)
        Mnorm2 = float((M * M).sum())
        inner = float((T * M).sum())
        resid2 = float(((T - M) ** 2).sum())
    else:
        subs, vals, dims = load_tns_sparse(args.tensor)
        U, lam = load_cp_dump(args.cp_dump, dims, args.rank)
        Tnorm2, Mnorm2, inner, resid2 = sparse_residual(subs, vals, U, lam)

    Tnorm = np.sqrt(max(Tnorm2, 0.0))
    Mnorm = np.sqrt(max(Mnorm2, 0.0))
    resid = np.sqrt(max(resid2, 0.0))
    fit = 1.0 - resid / max(Tnorm, 1e-30)

    print(f"[check] dims={dims}  rank={args.rank}  N={len(dims)}  "
          f"nnz/dense-mode={'sparse' if not args.dense else 'dense'}")
    print(f"[check] ||T||_F     = {Tnorm:.6e}    ||T||^2  = {Tnorm2:.6e}")
    print(f"[check] ||M||_F     = {Mnorm:.6e}    ||M||^2  = {Mnorm2:.6e}")
    print(f"[check] <T, M>      = {inner:.6e}    <=||T||*||M|| = {Tnorm*Mnorm:.6e}")
    print(f"[check] ||T - M||_F = {resid:.6e}    ||T-M||^2 = {resid2:.6e}")
    print(f"[check] true fit    = {fit:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
