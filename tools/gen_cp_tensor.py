"""
Generate a low-rank CP-structured tensor for CP-ALS correctness testing.

For chosen dims (I_0, ..., I_{N-1}) and rank R, draws random factors U_n
(I_n x R) from a standard normal, weights lambda[r] from a uniform
distribution, and writes the full dense tensor

    T[i_0, ..., i_{N-1}] = sum_r lambda[r] * prod_n U_n[i_n, r]

to <out>.tns in FROSTT COO text format (1-based indices, float value).

A matching .truth.npz file stores the ground-truth factors + lambda so the
caller can measure recovery error after CP-ALS.

Usage:
    python tools/gen_cp_tensor.py --dims 20 25 30 --rank 5 \
                                  --noise 0.0 --seed 7 \
                                  --out data/synth_cp_r5.tns
"""
import argparse
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", required=True,
                    help="tensor dimensions I_0 ... I_{N-1}")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="additive Gaussian noise std (fraction of ||T||)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    U = [rng.standard_normal(size=(I, args.rank)).astype(np.float32)
         for I in args.dims]
    lam = rng.uniform(0.5, 1.5, size=args.rank).astype(np.float32)

    # Build the dense tensor via einsum.  For CP structure,
    #    T[i_0, ..., i_{N-1}] = sum_r lambda[r] * prod_n U_n[i_n, r]
    # which is a sum of R rank-1 outer products weighted by lambda.
    N = len(args.dims)
    letters = "abcdefgh"[:N]
    # e.g. "ar,br,cr->abc"
    in_specs = ",".join(f"{L}r" for L in letters)
    spec = f"{in_specs},r->{letters}"
    T = np.einsum(spec, *U, lam).astype(np.float32)

    if args.noise > 0:
        Tnorm = float(np.linalg.norm(T))
        T = T + (args.noise * Tnorm / np.sqrt(T.size)) * \
            rng.standard_normal(T.shape).astype(np.float32)

    # Write FROSTT .tns: 1-based indices, one nonzero per line.
    # CP-structured tensors are generically dense, so we emit every cell.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        it = np.nditer(T, flags=["multi_index"])
        for v in it:
            idx = it.multi_index
            # skip exact zeros (unlikely but keeps file honest)
            if float(v) == 0.0:
                continue
            fp.write(" ".join(str(i + 1) for i in idx))
            fp.write(f" {float(v):.8g}\n")

    # Ground truth for post-hoc recovery check.
    truth = args.out + ".truth.npz"
    np.savez(truth, lam=lam, **{f"U{n}": U[n] for n in range(N)},
             tensor_frob=float(np.linalg.norm(T)))

    print(f"[gen] wrote tensor to {args.out}")
    print(f"[gen] dims={args.dims}  rank={args.rank}  nnz={int((T != 0).sum())}")
    print(f"[gen] ||T|| = {float(np.linalg.norm(T)):.6e}")
    print(f"[gen] ground truth saved to {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
