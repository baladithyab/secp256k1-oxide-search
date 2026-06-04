# ADR-0003: Alternative GPU Methods — The Full Solution-Space Matrix

- **Status:** Accepted
- **Date:** 2026-06
- **Context:** Codeseys asked to "explore other angles ... other methods that might be workable
  especially for GPUs." This ADR enumerates every method that's been tried or proposed for this
  problem class and states, for each, whether it changes the order of magnitude vs the ADR-0001
  baseline. The honest meta-finding up front:

> **There is no algorithm that makes the no-pubkey brute force (#71) sub-exponential.** It is a
> uniform search over an unstructured space; the information-theoretic floor is linear in keyspace.
> Every "method" below is either (a) a *constant-factor* engineering win on that brute force, or
> (b) a *different problem* (exposed-pubkey ECDLP) where √n algorithms apply. We pursue the biggest
> constant-factor wins (they're real and stack to ~10–50×) and implement the best √n method for the
> exposed-pubkey case. We do not chase a sub-exponential miracle for #71 — there isn't one.

## Method matrix

| Method | Applies to | Mechanism | Order-of-magnitude change? | Build? |
|---|---|---|---|---|
| **Naive double-and-add scalar mult** | both | baseline EC point mult per candidate | baseline | ✅ Track A baseline |
| **Batched Montgomery inversion (gECC)** | both | 1 modinv + 3n modmul amortizes the 500×-cost inversion across a batch; affine beats Jacobian at N>20 | **constant ~4–5×** (real, large) | ✅ core of Track A fast path |
| **IMAD-minimized modmul (gECC microarch)** | both | predicate-register carry, IADD3 substitution → fewer integer-MAC instructions | constant (stacks w/ above) | ✅ Track A inner loop |
| **Endomorphism (GLV / λ-decomposition)** | both | secp256k1's efficient endomorphism splits the scalar → ~2× fewer doublings | **constant ~2×** | ✅ cheap, high ROI |
| **Symmetry / negation map** | both | `−P` shares x-coord with `P` → ~√2 fewer distinct points to check; in kangaroo, the negation map gives √2 fewer jumps | constant ~1.4× | ✅ |
| **Baby-step giant-step (BSGS)** | exposed-pubkey only | meet-in-the-middle, O(√n) **time AND space** | **√n** (but memory-bound) | 🟡 Track D, memory permitting |
| **Pollard kangaroo (lambda)** | exposed-pubkey only | random walk on the group, O(√n) time, O(1) space, distinguished points | **√n** (the practical winner) | ✅ Track D primary |
| **RCKangaroo SOTA method** | exposed-pubkey only | improved jump distribution + DP density; c≈1.7 constant, near-optimal | √n with best constant | ✅ Track D target |
| **INT8 tensor-core modmul** | both | limb-decomposed bigint MAC as matmul on IMMA | constant, *maybe* (research) | 🟡 Track C spike |
| **Rho (Pollard rho)** | exposed-pubkey only | like kangaroo but for full-group DLP, not bounded interval | √n, worse fit than kangaroo for bounded ranges | ❌ kangaroo dominates here |
| **Index calculus** | ❌ not on secp256k1 | sub-exponential for *some* curves (GF(2^m) w/ structure), **NOT** prime-field secp256k1 | n/a | ❌ doesn't apply |
| **Rainbow tables / precompute** | ❌ | keyspace too large to precompute; no reuse across puzzles | n/a | ❌ |
| **ML "learn the keyspace"** | ❌ | PRF has no learnable structure (see ADR on nvfp4) | n/a | ❌ category error |
| **nvfp4 coarse→fine** | ❌ | no metric structure; FP4 approx = noise | n/a | ❌ refuted (root README) |

## The two regimes, crisply

### Regime 1 — no exposed pubkey (#71): constant-factor stacking only
Best achievable per-GPU rate = `baseline × (batched-inversion) × (endomorphism) × (negation) ×
(IMAD-min)` ≈ baseline × ~10–50×. That moves 7,500 GPU-years to maybe ~150–750 GPU-years on one
5090 — **still hopeless single-GPU, still negative-EV at cluster scale.** The honest framing: we
build the fast kernel because it's the right engineering and the benchmark headline, not because it
crosses a feasibility threshold.

### Regime 2 — exposed pubkey (#135): √n is the whole game
`c·√(2^134)` ≈ 1.7·2⁶⁷ group ops. This is genuinely cluster-attemptable (and being attempted by
pools). Kangaroo with the negation map + endomorphism + good DP density is the method. We implement
it (Track D) and validate on a small known-answer interval — never broadcasting a real solve.

## Decision

- Track A fast path = batched Montgomery + endomorphism + negation + IMAD-min modmul (all
  constant-factor, all real, all stack).
- Track C = INT8 tensor-core modmul spike (the one open research question).
- Track D = RCKangaroo-class kangaroo for the exposed-pubkey case, KAT-validated on a toy interval.
- Explicitly **decline**: index calculus (wrong curve), ML/coarse-fine (no structure), rainbow
  tables (space), and any "sub-exponential for #71" claim (impossible).

## Consequences

The repo's value proposition is: *the best honestly-achievable GPU constant factors, the correct
√n method for the tractable case, expressed and benchmarked across cuda-oxide vs CUDA-C* — with a
clear-eyed map of what is and isn't possible. That map (this ADR) is itself a deliverable.
