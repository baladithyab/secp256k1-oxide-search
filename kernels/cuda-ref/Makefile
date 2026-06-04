# Wave 1 CUDA-C secp256k1 reference scanner.
# ABSOLUTE nvcc path: apt shims are stale CUDA 12.0 and mishandle sm_120.
NVCC    := /usr/local/cuda/bin/nvcc
CCBIN   := clang-14
ARCH    := sm_120
NVFLAGS := -ccbin $(CCBIN) -O3 -arch=$(ARCH) -lstdc++

TARGET  := secp256k1_ref
SRC     := secp256k1_ref.cu
HDRS    := secp256k1_field.cuh secp256k1_ec.cuh hash.cuh

.PHONY: all clean verify scan bench

all: $(TARGET)

$(TARGET): $(SRC) $(HDRS)
	$(NVCC) $(NVFLAGS) -o $@ $(SRC)

# Run from repo root so the default test-vector path resolves.
verify: $(TARGET)
	cd ../.. && kernels/cuda-ref/$(TARGET) verify

scan: $(TARGET)
	cd ../.. && kernels/cuda-ref/$(TARGET) scan

bench: $(TARGET)
	cd ../.. && kernels/cuda-ref/$(TARGET) bench

clean:
	rm -f $(TARGET)
