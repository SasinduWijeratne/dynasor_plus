#!/usr/bin/env bash
# ============================================================================
#  Dynasor+  POSIX build script (Linux / macOS / WSL)
#
#  Usage:
#      ./scripts/build.sh                 # native ISA
#      ./scripts/build.sh avx2
#      ./scripts/build.sh neon            # AArch64
#      ./scripts/build.sh clean
# ============================================================================
set -euo pipefail

ISA=${1:-native}
CXX=${CXX:-g++}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
INC="$ROOT/include"
BUILD="$ROOT/build"
BIN="$BUILD/dynasor_plus"

if [ "$ISA" = "clean" ]; then
    rm -rf "$BUILD"
    echo "[build] cleaned $BUILD"
    exit 0
fi

mkdir -p "$BUILD"

case "$ISA" in
    native)      ARCH="-march=native -mtune=native" ;;
    znver5)      ARCH="-march=znver5 -mtune=znver5 -mprefer-vector-width=512" ;;
    znver4)      ARCH="-march=znver4 -mtune=znver4 -mprefer-vector-width=512" ;;
    neoverse-v2) ARCH="-mcpu=neoverse-v2 -msve-vector-bits=128" ;;
    neoverse-v1) ARCH="-mcpu=neoverse-v1 -msve-vector-bits=256" ;;
    neoverse-n2) ARCH="-mcpu=neoverse-n2 -msve-vector-bits=128" ;;
    avx512)      ARCH="-mavx512f -mavx512dq -mavx512vl -mfma -mprefer-vector-width=512" ;;
    avx2)        ARCH="-mavx2 -mfma" ;;
    sse2)        ARCH="-msse2" ;;
    neon)
        UN=$(uname -m 2>/dev/null || echo unknown)
        if [ "$UN" = "aarch64" ] || [ "$UN" = "arm64" ]; then
            ARCH="-march=armv8-a+simd"
        else
            ARCH="-mfpu=neon -mfloat-abi=hard"
        fi
        ;;
    sve2)   ARCH="-march=armv9-a+sve2" ;;
    sve)    ARCH="-march=armv8-a+sve" ;;
    scalar) ARCH="" ;;
    *) echo "unknown ISA: $ISA"; exit 1 ;;
esac

CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -fopenmp -funroll-loops $ARCH -I$INC"
LDFLAGS="-fopenmp"

# Optional whole-program LTO -- a consistent 3-8% win on gcc-15 for this
# kernel, at the cost of a longer link step.  Enable with:
#     DYN_LTO=1 ./scripts/build.sh znver5
if [ "${DYN_LTO:-0}" = "1" ]; then
    CXXFLAGS="$CXXFLAGS -flto=auto"
    LDFLAGS="$LDFLAGS -flto=auto -O3"
fi

# Optional: use explicit huge pages (mmap MAP_HUGETLB) for multi-MiB
# allocations.  Requires the admin to reserve a pool at boot, e.g.:
#     sudo sysctl -w vm.nr_hugepages=2048     # reserves 4 GiB of 2 MiB pages
# Without a pool the allocator silently falls back to std::malloc, so the
# binary still works on machines without huge pages configured.
if [ "${DYN_USE_HUGETLB:-0}" = "1" ]; then
    CXXFLAGS="$CXXFLAGS -DDYN_USE_HUGETLB=1"
fi

SRCS=(dynasor_io dynasor_preprocess dynasor_kernel dynasor dynasor_reference main)
OBJS=()
for s in "${SRCS[@]}"; do
    echo "[build] $CXX -c $s.cpp"
    $CXX $CXXFLAGS -c "$SRC/$s.cpp" -o "$BUILD/$s.o"
    OBJS+=("$BUILD/$s.o")
done

echo "[build] linking $BIN"
$CXX $LDFLAGS "${OBJS[@]}" -o "$BIN"
echo "--"
echo "[build] produced $BIN  (ISA=$ISA)"
