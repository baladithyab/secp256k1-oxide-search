# ADR-0004: N-GPU Orchestration + Stochastic Allocation

- **Status:** Accepted
- **Date:** 2026-06
- **Context:** Codeseys asked for auto-scale-to-N-GPUs and "a bit of stochastic calculus to help
  accelerate the orchestration and/or the solving." This ADR records the architecture decision and,
  importantly, *where the stochastic math legitimately helps and where it doesn't*.

## Decision

**Coordinator/worker, share-nothing on the hot path.** Full design in `orchestrator/README.md`.
- Brute-force mode: coordinator leases disjoint fixed-size keyspace blocks; workers sweep on-GPU;
  only lease/ack/heartbeat/FOUND cross the wire. Linear scaling, crash-tolerant via lease TTL.
- Kangaroo mode: workers run herds in the shared interval; emit distinguished points (DPs) to a
  central collision store; collision → key. Bandwidth tuned by DP threshold `d`.
- Transport: msgpack over length-prefixed TCP (v1). Heterogeneous/elastic by design.

## Where stochastic methods help (and where they're decoration)

| Layer | Stochastic content | Real benefit? |
|---|---|---|
| **Solving — kangaroo** | The algorithm *is* a random walk on the curve group; √n is its hitting time. SOTA = tuning jump distribution + DP density (RCKangaroo). | ✅ Yes — but this is "implement a good kangaroo," which we do anyway. No *extra* calculus layer beats √n. |
| **Solving — brute force** | None. Uniform search, memoryless, constant per-key hit probability while unswept. | ❌ No learning signal exists. By design (PRF). |
| **Orchestration — allocation** | Allocating finite GPU-hours across candidate ranges/puzzles under uncertainty + a contention hazard = multi-armed bandit / optimal-stopping. | ✅ **This is the legitimate hook.** Full spec in `docs/research/bandit-allocation.md`. |

## The allocator (the honest "stochastic calculus" deliverable)

`orchestrator/coordinator/allocator.py`, runs once per epoch (off the GPU hot path):
- **Brute-force arms:** no exploration (no learning signal). Rank by
  `prize_i / R_i · shrinkage(contention_i)`; stop an arm only when hazard-adjusted EV < 0.
- **Kangaroo arms:** Bayesian/Gittins-index bandit; posterior on time-to-collision updated from
  observed DP rate + herd overlap each epoch.
- **Contention hazard:** exponential `h_i(t)` (someone solves first), estimated from on-chain
  monitoring of the target address. Folds into `EV_i` with the private-mining mitigation factor.

**What it explicitly does NOT claim:** it does not beat √n for any instance, does not make brute
force tractable. It's an *allocation* optimizer + a *stop-loss* — spend a fixed budget more wisely
across options. That's the honest scope of "stochastic calculus accelerating orchestration."

## Consequences

The orchestration layer is genuinely sound and scales. The stochastic-control layer is real but
modest: it optimizes *where you point* a fixed budget, not the cost of any single search. Stated
this way so no one mistakes the allocator for an asymptotic improvement.
