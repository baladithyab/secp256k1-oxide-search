# secp256k1-oxide-search

A **cuda-oxide (Rust → PTX)** research artifact for GPU-accelerated secp256k1 key search,
built as an honest engineering + cryptography study — **not** a get-rich scheme.

> **Read this first.** This repo exists to answer real technical questions:
> 1. How does a v0.1.0 Rust-to-PTX frontend (cuda-oxide) fare on the most
>    compute-bound, carry-chain-heavy kernel class there is — 256-bit elliptic-curve
>    modular arithmetic — versus a hand-tuned CUDA C reference?
> 2. Can the search be cleanly auto-scaled across **N GPUs** with a coordinator/worker
>    design and a principled (bandit / optimal-stopping) range-allocation policy?
> 3. Can the 256-bit modular multiply be routed through **INT8 tensor cores**
>    (limb-decomposed bigint MAC) and beat the scalar IMAD path on consumer Blackwell (sm_120)?
>
> It is aimed at the [Bitcoin puzzle](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)
> keyspaces as a *concrete, well-specified benchmark target*, while being completely
> clear-eyed about the economics (see **Reality Check** below).

## Reality Check (please don't skip)

The Bitcoin puzzle is a **deliberately reduced-keyspace brute force**, not a crypto break.

- **Puzzle #71** (lowest unsolved, no exposed pubkey): ~2⁷⁰ ≈ **1.18 × 10²¹** candidate keys.
  Pure brute force. At an optimistic ~5 × 10⁹ keys/s on one RTX 5090 → **~7,500 GPU-years**.
  A 1,000-GPU cluster → ~7.5 years, racing established pools.
- **Puzzle #135** (exposed pubkey): bounded-interval ECDLP → **Pollard's kangaroo**, ~2⁶⁷ group
  ops. Tractable-with-a-cluster *in principle*, still cluster-months-to-years, also contested.
- **Public broadcast = theft.** Puzzle #69's prize was stolen via a mempool-replacement attack
  after a public solve transaction. Solvers now mine privately. This repo does **not** ship a
  broadcast path.

**Expected value is negative.** Treat any "winnings" framing as a lottery ticket, and treat the
repo as what it is: a systems + cryptography research project with a crisp benchmark.

## Three Tracks

| Track | Directory | Question it answers |
|---|---|---|
| **A — cuda-oxide EC kernels** | `kernels/oxide-ec/`, `kernels/cuda-ref/` | Rust→PTX vs hand-tuned C on secp256k1 point math; SASS-level analysis |
| **B — N-GPU orchestration** | `orchestrator/` | Coordinator/worker, disjoint-range partitioning, distinguished-point collision server, bandit range allocation |
| **C — INT8 tensor-core modmul** | `research/int8-tensorcore-modmul/` | Limb-decomposed 256-bit MAC on tensor cores vs scalar IMAD on sm_120 |

## Why NOT nvfp4 / "coarse→fine low-precision search"

This was an explicit design question and the answer is a **principled no** — recorded here so it
isn't re-litigated:

- **EC key search has no metric structure.** Coarse→fine (image pyramids, gradient descent, ANN)
  works only when *nearby inputs → nearby outputs*. secp256k1 is a cryptographic PRF engineered for
  the opposite: key `k` and `k+1` give **uncorrelated** public keys (full avalanche). There is no
  "getting warmer" — so no coarse pass can guide a fine pass.
- **The arithmetic can't be approximated.** A match requires the *exact* 256-bit result. An FP4/FP8
  approximation of a point multiply is **noise**, not a coarse answer. One wrong bit → no match.
  No partial credit.
- **nvfp4 is a floating format for ML weight quantization** — wrong tool for exact bigint.

**The salvageable kernel of the idea** lives in Track C: tensor cores *can* accelerate the **exact
INT8 integer** limb-products that compose a 256-bit modmul (matmul-shaped), à la `TensorCrypto` /
`gECC`. That's lossless integer accumulation, not low-precision approximation. See
`research/int8-tensorcore-modmul/README.md`.

## Where stochastic methods actually help

- **Solving:** Pollard's kangaroo *is* the stochastic-process answer (random walk on the curve
  group; √n is its hitting time). SOTA gains (RCKangaroo-class) come from tuning the jump
  distribution + distinguished-point density — applied stochastic processes, which we implement
  directly. There is **no** extra calculus layer that beats √n for the no-pubkey (uniform) case;
  that's information-theoretic.
- **Orchestration:** Allocating finite GPU-hours across candidate ranges under uncertainty is a
  genuine **optimal-stopping / multi-armed-bandit** problem. The coordinator's allocation policy
  is where stochastic control earns its keep. See `docs/research/bandit-allocation.md`.

## Status

🚧 Scaffold. Track A kernel bring-up first (the load-bearing technical result), then B, then C.
See `docs/plans/` for the wave plan.

## Toolchain

- RTX 5090, sm_120, CUDA 13.2, driver 596.21 (WSL2 Ubuntu)
- cuda-oxide 0.1.0 (NVlabs/cuda-oxide), Rust nightly, LLVM 21+
- See `docs/SETUP.md`

## License

MIT — see `LICENSE`. Research/educational use.

## Acknowledgements & prior art

Hand-tuned references this work measures itself against: BitCrack, KeyHunt, JeanLucPons/Kangaroo,
RCKangaroo, gECC (CGCL-codes/gECC), TensorCrypto. All credited inline in the relevant track READMEs.
