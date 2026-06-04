# ADR-0001: Scope, Economics, and Non-Goals

- **Status:** Accepted
- **Date:** 2026-06
- **Context:** Codeseys asked whether cuda-oxide can build something to solve the Bitcoin puzzle,
  auto-scale to N GPUs, use stochastic methods to accelerate, exploit nvfp4 low precision, and what
  Amazon Braket (quantum) would add.

## Decision

This repo is a **research artifact**: a cross-frontend GPU systems + applied-cryptography study that
uses the [Bitcoin puzzle](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) keyspaces as a concrete,
well-specified benchmark target. It is **not** a money-making tool and ships **no transaction
broadcast path**.

## Economics (the honest baseline every other ADR inherits)

- **#71** (lowest unsolved, no exposed pubkey): ~2⁷⁰ ≈ 1.18×10²¹ keys, pure brute force. At an
  optimistic 5×10⁹ keys/s on one RTX 5090 → **~7,500 GPU-years**. 1,000 GPUs → ~7.5 years, racing
  established pools.
- **#135** (exposed pubkey): bounded-interval ECDLP → Pollard kangaroo, ~2⁶⁷ group ops.
  Cluster-months-to-years, contested.
- **Broadcast = theft.** #69's prize was stolen via mempool replacement after a public solve. The
  repo deliberately has no broadcast code.

**Expected value is negative.** We build it for the engineering and the cryptography, with eyes open.

## Non-Goals

1. No transaction construction/broadcast/signing-for-spend.
2. No claim that any method makes #71 brute force tractable.
3. No private-key exfiltration tooling beyond what's needed to *demonstrate* a solve on a small
   known-answer toy interval.

## Consequences

Every downstream ADR must state its result *relative to this negative-EV baseline* — i.e. "does this
change the order of magnitude?" not "does this help at all?" Most don't.
