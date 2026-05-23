#!/usr/bin/env bash
# ============================================================================
#  Dynasor+ correctness test harness (POSIX)
# ============================================================================
set -u
BIN="${1:-$(dirname "$0")/../build/dynasor_plus}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

tensors=(3d_7.tns 3d-24.tns 4d_3_16.tns 3d_dense.tns)
ranks=(8 16 32 64)
threads=(1 2 4 8)

fail=0
ran=0
for t in "${tensors[@]}"; do
    tp="$ROOT/data/$t"
    if [ ! -f "$tp" ]; then echo "skip missing $tp"; continue; fi
    for R in "${ranks[@]}"; do
        for T in "${threads[@]}"; do
            ran=$((ran + 1))
            echo "=== $t  rank=$R  threads=$T ==="
            out=$("$BIN" "$tp" --rank "$R" --threads "$T" --verify 2>&1) || true
            echo "$out"
            if ! echo "$out" | grep -q '^PASS'; then
                echo "*** FAILURE ***"
                fail=$((fail + 1))
            fi
        done
    done
done

echo "============================================================"
if [ $fail -eq 0 ]; then
    echo "ALL $ran CONFIGURATIONS PASSED"
    exit 0
else
    echo "$fail / $ran CONFIGURATIONS FAILED"
    exit 1
fi
