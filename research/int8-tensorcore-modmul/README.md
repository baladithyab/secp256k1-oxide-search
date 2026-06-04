# Track C — INT8 Tensor-Core Modular Multiplication (research spike)

The **legitimate salvage** of the "use low-precision hardware" instinct. Not nvfp4, not
approximation — **exact INT8 integer** limb products routed through tensor cores.

## The idea (and why it's exact, not approximate)

A 256-bit × 256-bit multiply, with the operands split into `k` limbs of `w` bits, is a sum of
`k²` limb-products `a_i · b_j` placed at position `i+j`. That is **structurally a matmul** (outer
product of limb vectors), which is what tensor cores compute.

- Use **INT8 inputs with INT32 accumulation** (`mma.sync ... .s8.s8.s32` → `IMMA` on Blackwell).
  The accumulation is **lossless** for the small limb products — this is ordinary integer math, not
  a quantized approximation.
- Carry propagation + Montgomery/Barrett reduction happen *after* the tensor-core MAC, on the
  integer pipe.
- Prior art: `TensorCrypto` (Lee & Seo, lattice crypto on tensor cores), `gECC` (CGCL-codes/gECC,
  4–5× EC throughput via batched Montgomery + IMAD minimization). gECC's headline insight — modmul
  cost is dominated by IMAD count, minimized via predicate-register carry handling and IADD3
  substitution — is the *baseline we'd try to beat* with the tensor-core path.

## Why this is a spike, not a promise

- On **consumer Blackwell (sm_120)** the INT8 tensor-core path competes with a very well-tuned
  scalar IMAD/IADD3 integer path; the win (if any) is modest and shape-sensitive. gECC itself beats
  prior tensor-adjacent work using *scalar* integer optimization.
- cuda-oxide v0.1.0 has **no inline `asm!`**, so the hand-tuned `madc.hi/lo.cc` carry-chain that the
  C references rely on is *unreachable from Rust kernels today*. That makes Track C doubly
  interesting for the repo: it's a setting where the tensor-core route might let cuda-oxide
  *sidestep* its own missing-`asm!` limitation by expressing the bigint MAC as `mma` instead of a
  carry chain.

## Experiment ladder

1. **Scalar baseline (C + oxide):** 256-bit Montgomery modmul, scalar limbs. Measure modmul/s and
   SASS IMAD count. (This is also the Track A inner-loop primitive.)
2. **INT8 tensor-core modmul (C):** limb-decomposed MAC via `mma.sync .s8.s8.s32`, then reduce.
   Measure vs baseline. Establishes whether the route is viable on sm_120 *at all*.
3. **INT8 tensor-core modmul (oxide):** can cuda-oxide reach the `mma` path? (Per rust-gpu-compute
   skill, oxide's `mma.sync` support in v0.1.0 is absent/placeholder — this may be a *negative*
   result: "oxide can't express IMMA yet," which is itself a publishable finding.)
4. **Decision:** if (2) beats (1) by enough to matter and (3) is reachable, fold into Track A's
   scalar-mult inner loop. Otherwise document the negative result and keep Track A on scalar IMAD.

## Acceptance signals
- `IMMA.16832.S32.S8.S8` (or sm_120 equivalent) present in the tensor-core kernel's SASS.
- Lossless correctness vs a CPU bigint reference (Python `int`), bit-exact, no tolerance.
- modmul/s vs the scalar baseline on the same idle GPU, same session (thermal discipline).
