#!/usr/bin/env bash
# Build the Wave 1 CUDA-C secp256k1 reference scanner.
# Uses ABSOLUTE nvcc path (apt shims are stale CUDA 12.0 and mishandle sm_120).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/secp256k1_ref}"
/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ \
    -o "$OUT" "$HERE/secp256k1_ref.cu"
echo "built: $OUT"
