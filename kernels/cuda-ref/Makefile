# Wave 1 CUDA-C secp256k1 reference scanner.
# ABSOLUTE nvcc path: apt shims are stale CUDA 12.0 and mishandle sm_120.
NVCC    := /usr/local/cuda/bin/nvcc
CCBIN   := clang-14
ARCH    := sm_120
NVFLAGS := -ccbin $(CCBIN) -O3 -arch=$(ARCH) -lstdc++

TARGET  := secp256k1_ref
SRC     := secp256k1_ref.cu
HDRS    := secp256k1_field.cuh secp256k1_ec.cuh hash.cuh

# Wave 3 fast-path variant (affine walk + batched Montgomery inversion). Separate binary so the
# Wave-1 baseline stays runnable for honest A/B. BATCH overridable: make BATCH=512.
FAST    := secp256k1_fast
FASTSRC := secp256k1_fast.cu
FASTHDR := secp256k1_field.cuh secp256k1_ec.cuh secp256k1_affine.cuh hash.cuh
BATCH   ?= 256

# Wave 3 Opt C variant (negation-map symmetry): same source, built with -DNEGMAP so each affine
# point hashes BOTH compressed parities (02||x and 03||x), covering keys k and n-k. Separate binary
# so the Opt A baseline stays byte-identical for honest A/B (the NEG branches DCE away without -DNEGMAP).
NEGMAP_BIN := secp256k1_negmap

.PHONY: all clean verify scan bench fast fast-verify fast-scan fast-bench \
        negmap negmap-verify negmap-scan negmap-bench

all: $(TARGET) $(FAST)

$(TARGET): $(SRC) $(HDRS)
	$(NVCC) $(NVFLAGS) -o $@ $(SRC)

fast: $(FAST)
$(FAST): $(FASTSRC) $(FASTHDR)
	$(NVCC) $(NVFLAGS) -DBATCH=$(BATCH) -o $@ $(FASTSRC)

negmap: $(NEGMAP_BIN)
$(NEGMAP_BIN): $(FASTSRC) $(FASTHDR)
	$(NVCC) $(NVFLAGS) -DBATCH=$(BATCH) -DNEGMAP -o $@ $(FASTSRC)

# Run from repo root so the default test-vector path resolves.
verify: $(TARGET)
	cd ../.. && kernels/cuda-ref/$(TARGET) verify

scan: $(TARGET)
	cd ../.. && kernels/cuda-ref/$(TARGET) scan

bench: $(TARGET)
	cd ../.. && kernels/cuda-ref/$(TARGET) bench

# Wave 3 fast-path run targets (from repo root so the default test-vector path resolves).
fast-verify: $(FAST)
	cd ../.. && kernels/cuda-ref/$(FAST) verify
fast-scan: $(FAST)
	cd ../.. && kernels/cuda-ref/$(FAST) scan
fast-bench: $(FAST)
	cd ../.. && kernels/cuda-ref/$(FAST) bench

# Wave 3 Opt C run targets. negmap-verify expects NEGFILE (a kat_oracle.py negmap dump).
NEGFILE ?= /tmp/negmap_1_1100.json
negmap-verify: $(NEGMAP_BIN)
	cd ../.. && kernels/cuda-ref/$(NEGMAP_BIN) negverify $(NEGFILE)
negmap-scan: $(NEGMAP_BIN)
	cd ../.. && kernels/cuda-ref/$(NEGMAP_BIN) scan
negmap-bench: $(NEGMAP_BIN)
	cd ../.. && kernels/cuda-ref/$(NEGMAP_BIN) bench

clean:
	rm -f $(TARGET) $(FAST) $(NEGMAP_BIN) $(FAST)_b*
