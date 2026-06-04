# Bandit / Optimal-Stopping Range Allocation

The one place where "add some stochastic math" is a *real* improvement rather than decoration.

## The problem

A coordinator owns a finite budget of GPU-hours (and a power/$ cost rate). It can point workers at
several candidate sub-ranges or puzzles, each with its own:

- range size `2^b` (→ prior collision/hit probability per unit work),
- exposed-pubkey status (brute force vs kangaroo — different work model),
- contention estimate (are pools already grinding it?),
- partial progress (fraction of range swept / distinguished points collected).

**Decision, repeated each epoch:** where do I send the next block of GPU-hours to maximize
expected value per watt — and *when do I stop* a range that's looking bad?

This is NOT going to change the asymptotics of any single search. It optimizes *allocation across*
searches under uncertainty, which is a legitimately different, tractable problem.

## Model

### Brute-force ranges (no pubkey)
Hit is a Poisson process in swept keys. With range `R = 2^b` keys and sweep rate `λ` keys/s, the
probability the key lies in the not-yet-swept portion is uniform; expected remaining work is
memoryless given fraction swept `f`: `E[keys to hit | not yet found] = R·(1−f)/2` in expectation
over the unknown key, but the *marginal* hit-probability per additional key is constant `1/R` while
unswept. → A brute-force arm has **constant reward rate** `1/R` per key until exhausted; there is
**no learning signal** mid-sweep (consistent with "no metric structure"). So for brute-force arms
the allocation reduces to: prefer smallest `R`, account for contention, and there's nothing to
"explore" — it's pure exploitation of the narrowest range. Optimal-stopping only matters when a
range is shared with a pool that may solve it first (a hazard term).

### Kangaroo ranges (exposed pubkey)
Expected group-ops to collision ≈ `c·√(2^b)` (c ≈ 1.7 for SOTA 3-kangaroo). Distinguished-point
(DP) collection gives a **real-time progress signal**: observed DP rate and herd overlap let you
update a posterior on time-to-collision. This arm *does* support learning → a Bayesian/Gittins-index
treatment is meaningful.

### Contention hazard
Model "someone else solves it first" as an exponential hazard `h_i(t)` per range (estimated from
on-chain monitoring of the target address + known pool chatter). Expected reward of arm `i`:

```
EV_i(allocate Δt) = P(we solve in Δt) · prize_i · P(not front-run | we solve)
                  − cost(Δt)
```

`P(not front-run)` folds in both the hazard and the private-mining mitigation.

## Policy

1. **Brute-force arms:** rank by `prize_i / R_i · S(contention_i)` (shrinkage on contended ranges).
   No exploration term. Stop an arm only if hazard-adjusted EV goes negative.
2. **Kangaroo arms:** Gittins-index / Bayesian-bandit over arms, posterior on time-to-collision
   updated from DP statistics each epoch. This is the part with genuine exploration value.
3. **Cross-arm:** allocate next GPU-block to `argmax` index; re-solve each epoch (epoch = DP-upload
   interval, e.g. 60 s). Optimal-stopping rule per arm uses the hazard + budget shadow price.

## What this explicitly does NOT claim

- It does **not** beat √n for any single ECDLP instance.
- It does **not** make brute force tractable.
- It is an *allocation* optimizer: it spends a fixed budget more wisely across options, and tells
  you when to quit a losing arm. That's the honest scope.

## Implementation

`orchestrator/coordinator/allocator.py` — pure-Python policy, fed by worker telemetry over the
coordinator's RPC. Kept out of the GPU hot path entirely (runs once per epoch).
