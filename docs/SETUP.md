# Setup

Verified target environment (this dev box):

- NVIDIA RTX 5090, compute capability **sm_120** (consumer Blackwell)
- CUDA Toolkit **13.2** (V13.2.78), driver **596.21**
- WSL2 Ubuntu
- cuda-oxide **0.1.0** (NVlabs/cuda-oxide), Rust nightly, LLVM 21+

## cuda-oxide env (MUST export before any `cargo oxide` command)

```bash
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin
cargo oxide doctor   # every line must be ✓
```

Without these, cuda-oxide picks up a stale libNVVM shim → fake "Rust safety tax" + can't target
sm_120. (See rust-gpu-compute notes / libnvvm-shadow-bug.)

## CUDA C reference builds

Use **absolute paths** — the apt `/usr/bin/{nvcc,cuobjdump}` are stale CUDA 12.0 shims that
silently reject / mis-handle sm_120:

```bash
/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ -o ref ref.cu
/usr/local/cuda/bin/cuobjdump --dump-sass ref   # SASS analysis
```

## Python (orchestrator + correctness refs) — use uv

```bash
uv venv && source .venv/bin/activate
uv pip install coincurve ecdsa msgpack numpy
```

`coincurve` / `ecdsa` provide the bit-exact CPU reference for kernel correctness.

## Correctness discipline

All crypto kernels validated **bit-exact** against the Python reference — no float tolerance.
One wrong bit = wrong key. This is non-negotiable for this domain.
