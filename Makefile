## ============================================================================
##  Dynasor+  Makefile
## ============================================================================
##  Auto-detects architecture and enables the best-available SIMD ISA.
##
##    make                         # native build (host ISA)
##    make ISA=znver5              # AMD Zen 5   (Ryzen 9000 / EPYC 9005)     [GCC >= 14.1]
##    make ISA=znver4              # AMD Zen 4   (Ryzen 7000 / EPYC 9004)     [GCC >= 12.1]
##    make ISA=graniterapids       # Intel Granite Rapids (6th Gen Xeon)      [GCC >= 14.1]
##    make ISA=sapphirerapids      # Intel Sapphire Rapids (4th Gen Xeon)     [GCC >= 11.1]
##    make ISA=neoverse-v2         # Arm Neoverse V2 (Graviton4 / Grace)      [GCC >= 12.1]
##    make ISA=neoverse-v1         # Arm Neoverse V1 (Graviton3)
##    make ISA=neoverse-n2         # Arm Neoverse N2
##    make ISA=avx512              # generic AVX-512F + FMA
##    make ISA=avx2                # generic AVX2 + FMA
##    make ISA=sse2                # SSE2
##    make ISA=neon                # ARM NEON
##    make ISA=sve2                # ARM SVE2 (generic)
##    make ISA=scalar              # disable SIMD entirely
##    make ISA=generic             # -O3 only, let the compiler auto-vectorize
##    make test                    # end-to-end correctness test
##    make clean
##
##  Requires: g++/clang++ with OpenMP support (-fopenmp).
## ============================================================================

CXX        ?= g++
AR         ?= ar

UNAME_S    := $(shell uname -s 2>/dev/null || echo Windows)
UNAME_M    := $(shell uname -m 2>/dev/null || echo unknown)

ISA        ?= native

ARCH_FLAGS :=
ifeq ($(ISA),native)
  ARCH_FLAGS := -march=native -mtune=native
else ifeq ($(ISA),znver5)
  # AMD Zen 5 (Ryzen 9000 / EPYC 9005 "Turin").  Zen 5 has native 512-bit
  # AVX-512 datapaths; -mprefer-vector-width=512 keeps GCC from splitting
  # to 256-bit registers.  Requires GCC >= 14.1.
  ARCH_FLAGS := -march=znver5 -mtune=znver5 -mprefer-vector-width=512
else ifeq ($(ISA),znver4)
  # AMD Zen 4 (Ryzen 7000 / EPYC 9004 "Genoa" / "Bergamo").  Zen 4 has
  # AVX-512 but runs it as two 256-bit micro-ops internally -- still a
  # throughput win for spMTTKRP because we issue fewer uops per nnz.
  # Requires GCC >= 12.1.
  ARCH_FLAGS := -march=znver4 -mtune=znver4 -mprefer-vector-width=512
else ifeq ($(ISA),graniterapids)
  # Intel Granite Rapids (6th Gen Xeon Scalable, "Xeon 6").  Full-rate
  # 512-bit AVX-512 datapaths, AVX-512 FP16/BF16, AMX, and a wider L2
  # (2 MiB private).  AMX is unused here (we're FP32 sparse, not GEMM),
  # but -march=graniterapids still enables every AVX-512 sub-feature
  # GCC knows about, so the compiler can emit vpermt2ps / vgatherdps
  # where profitable.  Requires GCC >= 14.1.
  ARCH_FLAGS := -march=graniterapids -mtune=graniterapids -mprefer-vector-width=512
else ifeq ($(ISA),graniterapids-d)
  # Granite Rapids-D (MCC / XCC dies with integrated I/O die, HBM SKUs).
  # Identical ISA to plain graniterapids but GCC tunes for the larger
  # LLC.  Use this when building for HBM-equipped Xeon 6 parts.
  ARCH_FLAGS := -march=graniterapids-d -mtune=graniterapids-d -mprefer-vector-width=512
else ifeq ($(ISA),sapphirerapids)
  # Intel Sapphire Rapids (4th Gen Xeon Scalable, Xeon Platinum 8xxx).
  # First Intel server part with full-rate 512-bit AVX-512 since Ice Lake
  # Server; adds AVX-512 BF16 + FP16, AMX, and CLWB/CLDEMOTE.  Requires
  # GCC >= 11.1 (11.1 added the -march token; 12+ recommended for
  # -mtune heuristics).
  ARCH_FLAGS := -march=sapphirerapids -mtune=sapphirerapids -mprefer-vector-width=512
else ifeq ($(ISA),emeraldrapids)
  # Intel Emerald Rapids (5th Gen Xeon Scalable).  Same ISA as Sapphire
  # Rapids, uarch refresh with larger L3.  GCC 13+ recognises
  # -march=emeraldrapids; on older GCC fall back to sapphirerapids.
  ARCH_FLAGS := -march=emeraldrapids -mtune=emeraldrapids -mprefer-vector-width=512
else ifeq ($(ISA),neoverse-v2)
  # Arm Neoverse V2 (AWS Graviton4, NVIDIA Grace).  SVE2 with fixed
  # 128-bit vector length.  Requires GCC >= 12.1.
  ARCH_FLAGS := -mcpu=neoverse-v2 -msve-vector-bits=128
else ifeq ($(ISA),neoverse-v1)
  ARCH_FLAGS := -mcpu=neoverse-v1 -msve-vector-bits=256
else ifeq ($(ISA),neoverse-n2)
  ARCH_FLAGS := -mcpu=neoverse-n2 -msve-vector-bits=128
else ifeq ($(ISA),avx512)
  ARCH_FLAGS := -mavx512f -mavx512dq -mavx512vl -mfma -mprefer-vector-width=512
else ifeq ($(ISA),avx2)
  ARCH_FLAGS := -mavx2 -mfma
else ifeq ($(ISA),sse2)
  ARCH_FLAGS := -msse2
else ifeq ($(ISA),neon)
  ifeq ($(UNAME_M),aarch64)
    ARCH_FLAGS := -march=armv8-a+simd
  else
    # 32-bit ARM
    ARCH_FLAGS := -mfpu=neon -mfloat-abi=hard
  endif
else ifeq ($(ISA),sve2)
  ARCH_FLAGS := -march=armv9-a+sve2
else ifeq ($(ISA),scalar)
  ARCH_FLAGS :=
else ifeq ($(ISA),generic)
  ARCH_FLAGS :=
endif

CXXFLAGS   := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
              -fopenmp -funroll-loops $(ARCH_FLAGS) -Iinclude
# Emit .d dependency files alongside each .o so a header edit invalidates
# every .cpp that includes it.  Without this, partial rebuilds can link
# TUs compiled against DIFFERENT struct layouts -- a silent ABI mismatch
# that manifests as random access-violations at runtime.
DEPFLAGS   := -MMD -MP
CXXFLAGS   += $(DEPFLAGS)
LDFLAGS    := -fopenmp
LIBS       :=
# dynasor_jit.cpp uses dlopen/dlsym on POSIX; link against -ldl.  On
# Windows (MinGW) LoadLibraryA lives in kernel32 which ld links
# automatically, so nothing extra is needed.
ifneq ($(OS),Windows_NT)
LIBS       += -ldl
endif

# Optional whole-program LTO -- a consistent 3-8% win with gcc-15 for this
# kernel.  Enable with:  make LTO=1 ISA=znver5
LTO        ?= 0
ifeq ($(LTO),1)
  CXXFLAGS += -flto=auto
  LDFLAGS  += -flto=auto -O3
endif

# Optional: use explicit huge pages (mmap MAP_HUGETLB) for multi-MiB
# allocations.  Requires the admin to reserve a pool at boot, e.g.:
#     sudo sysctl -w vm.nr_hugepages=2048   # 4 GiB of 2 MiB pages
# Without a pool the allocator silently falls back to std::malloc.
# Enable with:  make HUGETLB=1 ISA=znver5
HUGETLB    ?= 0
ifeq ($(HUGETLB),1)
  CXXFLAGS += -DDYN_USE_HUGETLB=1
endif

BUILD      := build
BIN        := $(BUILD)/dynasor_plus

SRCS := src/dynasor_io.cpp \
        src/dynasor_preprocess.cpp \
        src/dynasor_kernel.cpp \
        src/dynasor.cpp \
        src/dynasor_jit.cpp \
        src/dynasor_reference.cpp \
        src/dynasor_linalg.cpp \
        src/dynasor_cpals.cpp \
        src/dynasor_ooc.cpp \
        src/dynasor_morton.cpp \
        src/dynasor_dnp.cpp \
        src/main.cpp

OBJS := $(SRCS:src/%.cpp=$(BUILD)/%.o)
DEPS := $(OBJS:.o=.d)

-include $(DEPS)

.PHONY: all test clean print-config

all: $(BIN)

print-config:
	@echo "Compiler : $(CXX)"
	@echo "ISA      : $(ISA)"
	@echo "CXXFLAGS : $(CXXFLAGS)"

$(BUILD):
	@$(call MKDIR,$(BUILD))

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)
	@echo "--"
	@echo "Built: $(BIN)"

test: $(BIN)
	@echo "=== running 3d_7 correctness test ==="
	./$(BIN) data/3d_7.tns --rank 16 --threads 2 --verify

test-3d24: $(BIN)
	./$(BIN) data/3d-24.tns --rank 16 --threads 4 --verify

test-4d: $(BIN)
	./$(BIN) data/4d_3_16.tns --rank 16 --threads 2 --verify

test-cpals: $(BIN)
	@echo "=== running CP-ALS smoke test on dense3d.tns ==="
	./$(BIN) data/dense3d.tns --decompose cpals --rank 16 --threads 2 \
	         --cp-iters 30 --cp-tol 1e-5

# Portable shell helpers: Windows (cmd.exe) lacks `rm -rf` and `mkdir -p`,
# POSIX lacks `rmdir /s /q` and treats `mkdir build` as an error if the
# directory already exists.  We pick the idiom based on $(OS) so the
# Makefile works on native mingw32-make (which invokes cmd.exe) as well as
# on Linux / macOS.
ifeq ($(OS),Windows_NT)
MKDIR  = if not exist $(1) mkdir $(1)
RMTREE = if exist $(1) rmdir /s /q $(1)
else
MKDIR  = mkdir -p $(1)
RMTREE = rm -rf $(1)
endif

clean:
	$(call RMTREE,$(BUILD))
