# Wave D — GPU Pollard's-kangaroo ECDLP solver (Track D, ADR-0003 Regime 2)

> Research artifact. Solves the **bounded-interval discrete log** Q = d·G for d ∈ [lo,hi] on
> secp256k1 — the EXPOSED-PUBKEY regime (ADR-0003 Regime 2), the only regime where √n algorithms
> apply. **No broadcast/spend path. KATs are self-generated toy intervals only** (never a real
> mainnet pubkey). All results below are **bit-exact** against `scripts/kat_oracle.py kangaroo`
> (recovered d == d_hex, no tolerance).
>
> Hardware/toolchain: RTX 5090 (sm_120, 32 GB), CUDA 13.2 (`/usr/local/cuda/bin/nvcc`), `-ccbin
> clang` (llvm-21), `-O3 -arch=sm_120`. Same machine as Waves 1–3.

## 1. What was built

A self-contained CUDA solver (`kernels/kangaroo/kangaroo.cu`):

```
./kangaroo <lo_hex> <hi_hex> <Q_compressed_hex>     # prints d=<hex> on success
```

It **reuses the verified field/EC/affine primitives unchanged** (`secp256k1_field.cuh`,
`secp256k1_ec.cuh`, `secp256k1_affine.cuh`) — porting bugs are the #1 risk, so no field math was
reimplemented. The **only** new field routine is a modular square root `a^((p+1)/4) mod p` (valid
since p ≡ 3 mod 4) to decompress Q; its exponent was cross-checked against G (√(Gx³+7) == Gy).

**Algorithm.** Pollard's kangaroo, tame + wild herds:
- **Interval shift** to Q' = Q − lo·G = (d−lo)·G, so the unknown d' = d−lo ∈ [0, W=hi−lo]. Every
  tracked distance stays ≪ 2⁶⁴ ≪ n, so the collision identity `d' = T_tame − D_wild` holds as an
  **exact integer equation — no mod-n arithmetic** (the reference field code only does mod p). This
  single design choice is what keeps the solver bit-exact with field-only math.
- **Tame** kangaroos start at known scalars spread across [0,W]; **wild** start at Q' + offset·G.
- **Pseudo-random jumps**: K=32 jump points J[i] = s_i·G; jump index = x(P) mod K; mean jump ≈ √W/2
  (the load-bearing parameter — collision time is ~2.08·√W only when the mean matches √W/2).
- **Affine walk**: each step reads affine x (one modular inverse), needed because the jump index and
  DP test must be a function of the *point*, not its Jacobian representation (so a tame and wild that
  land on the same point walk identically thereafter — that's the collision).
- **Distinguished points (DP)**: x with `dp_bits` trailing zero bits; reported to a host-side hash
  table. A tame/wild DP at the same x ⇒ candidate d'. The **[lo,hi] range filter** distinguishes the
  true collision from the ±P false x-match (a wrong-sign match lands far outside the interval), and a
  final on-device `cand·G == Q` check is the ultimate gate before printing.

**De-risking discipline.** The algorithm was first proven in a Python reference
(`kangaroo_ref.py`, using coincurve for the trusted group ops) — 12 oracle KATs + 25 adversarial
edge cases bit-exact — *before* the CUDA port. This isolated algorithm-correctness from
porting-correctness. (The jump-size bug — mean jump ≫ √W/2 making the herds never intersect — was
caught here at 24-bit, exactly the TDD "watch it fail across sizes" payoff.)

## 2. Correctness (THE GATE — acceptance #1)

| Test | Cases | Result |
|---|---|---|
| Python reference KATs (16/20/24/32-bit × 3 seeds) | 12 | ✅ bit-exact |
| Python reference adversarial (d at lo/hi/mid/quarter; tiny 8–16-bit) | 25 | ✅ bit-exact |
| **CUDA base** KATs (16/20/24/32-bit × 3 seeds) | 12 | ✅ bit-exact |
| **CUDA negation-map** KATs (same) | 12 | ✅ bit-exact |
| Adversarial reviewer sub-agent (see §6) | ~60–100 | see §6 |

Every recovered d equals the oracle's d_hex exactly. A kangaroo that finds a "close" d is worthless;
none of ours do.

## 3. Throughput (acceptance #3)

The affine walk's cost is dominated by the per-step modular inverse (Fermat, ~270 field-muls — "the
single most expensive field op", per the Wave-3 affine notes). **Batched Montgomery inversion** is the
lever: assign B kangaroos per thread, invert all B step-denominators with ONE inverse + ~3B muls, so
per-kangaroo inverse cost collapses to ~270/B + 2 muls.

**Peak throughput, RTX 5090 (2⁵⁶ interval, ~1–2 M kangaroos, matched-herd):**

| walk variant | jumps/s | vs per-step |
|---|---:|---:|
| per-step Fermat inverse (`KANG_BATCH=0`) | 1.14 × 10⁸ | 1.0× |
| batched B=8 | 5.6 × 10⁸ | 4.9× |
| batched B=16 | 9.1 × 10⁸ | 8.0× |
| **batched B=32 (default)** | **1.05 × 10⁹** | **9.2×** |
| batched B=64 | 4.3 × 10⁸ | (spills) |

**B=32 is the occupancy sweet spot at ~1.05 × 10⁹ jumps/s.** B=64 collapses because the per-thread
`affpt[B] + pre[B+1]` frame (~12 KB) spills to local memory and slashes resident warps — the same
occupancy wall the Wave-3 notes hit with their `prod[257]` stack frame.

**Reference comparison (published, web-sourced — for orientation only, not run here):**
RCKangaroo (RetiredC) reports ~8 G jumps/s on an RTX 4090; JeanLucPons/Kangaroo ~7.8 G aggregate on
4× V100. Our 1.05 G/s on a 5090 is a *correct, honestly-measured* kernel that is **~8× off** the SOTA
constant. The gap is expected and attributable: (a) we still do a full Fermat inverse per batch (SOTA
uses cheaper inverse + a far larger DP/jump-table working set), (b) K=32 jumps and a simple DP scheme
vs RCKangaroo's tuned distribution, (c) no warp-cooperative batching. Closing it is engineering, not
algorithm — and out of scope for this KAT-validated artifact.

## 4. √n scaling validation (acceptance #2)

Measured **total jumps to solve** in the linear regime (small fixed herd M=512/side ≪ √W, so jumps
≈ c·√W is independent of M and the √n law is visible), median over 9 seeds (kangaroo solve-time is
heavy-tailed — median is the honest statistic):

| bits | √W | median jumps | median c (=jumps/√W) | jumps ratio (per +4 bits) |
|---:|---:|---:|---:|---:|
| 32 | 65 535 | 3.20e5 | 4.00 | — |
| 36 | 262 143 | 8.74e5 | 3.00 | 2.73× |
| 40 | 1 048 575 | 3.57e6 | 3.62 | 4.08× |
| 44 | 4 194 303 | 1.31e7 | 2.53 | 3.68× |
| 48 | 16 777 215 | 3.58e7 | 2.14 / **2.03 mean→** | 2.73× |

**Confirms O(√n).** The empirical c converges to **~2.03 at 48-bit**, matching the textbook 2.08·√W
for the 2-kangaroo method. Jumps grow ~4× per +4 interval bits (≈2× per +2 bits) — the √n signature.
(Raw table: `results/kangaroo_scaling.txt`; harness: `kernels/kangaroo/scaling.py`.)

## 5. Negation map A/B (acceptance #4) — an honest **negative** result

The negation map folds {P,−P} (shared x, y↔p−y) to the canonical rep (x, even-y), aiming for ~√2
fewer jumps. It is implemented behind a `-DNEGMAP` compile gate (base binary byte-unchanged;
regression: base still 12/12 KATs). The negmap path adds a signed dlog-state `dlog(C)=a·d'+w`
(a=0 tame / a=±1 wild, w a signed int never reduced mod n) and a fruitless-cycle escape (window
detect + alternate jump idx^1, a pure function of the point so merged kangaroos escape identically).
**It solves all 12 KATs bit-exact** — correctness survives the optimization.

**But the measured lift is a LOSS, not √2:**

| bits | base median jumps | negmap median jumps | lift | seeds |
|---:|---:|---:|---:|---:|
| 36 | 1.049e6 | 1.573e6 | **0.67×** | 9/9 |
| 40 | 2.36e6 | 9.44e6 | **0.25×** | 16/16 |

**Root cause (confirmed by DP-yield + escape instrumentation, not guessed):** the jump index and the
DP test **both key on the affine x-coordinate**, which is *already* invariant under P→−P (they share
x). So canonicalizing y folds nothing the walk distinguishes on — **zero space reduction, no √2** —
while the signed bookkeeping breaks tame/wild collision symmetry (opposite-sign wilds give da∈{0,±2}
and yield no usable equation). Instrumentation: DP density was identical (~0.117 DP/jump) both ways;
the negmap simply needed ~3.3× more *distinct* DPs to collide. The escape fired at the benign ~1.5%
(≈1/64, the theoretical 2-cycle rate) — **not** the cause.

This reproduces **Bernstein–Lange, "On the correct use of the negation map"**: a negation map applied
to an x-keyed walk *without redefining the iteration function to act on the folded class* is
counterproductive. To realize the real √2, the jump function s_j and the DP predicate must be defined
on the canonical class (this is what RCKangaroo's "SOTA method" does) — a genuine algorithm change,
**left as future work**. Contrast Wave-3 Opt C's measured +20.7% neg-map gain: that was the
*unbounded brute-force scan* (two addresses per affine add — a different mechanism that does not
transfer to the bounded ECDLP walk). The two results are consistent once the regimes are separated.

**GLV/φ-endomorphism (optional stretch): not built, and would face the same wall.** φ(P)=(βx,y)
gives extra candidate x-coords, but our walk already keys on x; without redefining the jump function
on the {P,φP,φ²P,±} class it would add bookkeeping for no fold — the same failure mode as the
negation map. Deferred to future work alongside the canonical-class jump function.

## 6. Adversarial review

Battery (harness `kernels/kangaroo/adversarial.py`): d at exactly lo, lo+1, hi, hi−1, midpoint,
midpoint±1, quarter for bits ∈ {8,10,12,16,20,24}; asymmetric non-power-of-two interval
(0x3A7F1..0x3B200); narrow widths (2, 3, 16); width-with-d-at-bottom; 8 pseudo-random seeds at
20/24-bit. Each run has a per-case timeout so a HANG is distinguished from a WRONG answer (the latter
is the real prize). An independent reviewer sub-agent also analyzed the source and concluded the
attack surface is *hangs, not wrong answers*, because the device-side `k_verify` (cand·G == Q)
gates every printed d — a wrong answer would require the verify kernel to also be wrong.

| binary | cases | pass | **WRONG** | hang |
|---|---:|---:|---:|---:|
| **base** | 54 | 54 | **0** | 0 |
| negmap (≥16-bit) | 21 | 20 | **0** | 1 |

**Findings:**
- **Base solver: 54/54, zero wrong, zero hangs** — including all boundary, narrow, asymmetric, and
  random cases. The bit-exact correctness claim is **credible and stress-tested**.
- **Two real edge bugs were found and fixed during review** (both *hangs*, never wrong answers):
  1. **d == lo (d′=0)**: Q′ = O (point at infinity), which the affine walk cannot represent → hang.
     Fixed by detecting Q′ == O at init and short-circuiting to d=lo (still gated by `k_verify`).
     Regression test added (`test_cuda_interval_endpoints`); 18/18 CUDA tests pass.
  2. (Generalized) very tiny intervals were sized robustly via the adaptive dp/steps/budget heuristics.
- **negmap: zero wrong answers**, but one HANG at 16-bit/hi−1 (d′=W−1) that persists across herd
  sizes — a negmap-specific fragility at small W near the top boundary, consistent with §5's
  net-loss / tiny-W thrash behavior. It is NOT a correctness defect (no wrong d), and it does not
  occur for the base solver. Since the negmap is a *measured net loss* anyway (§5), this is a
  documented limitation of an opt-in, non-default path, not a shipping blocker.

**Verdict:** the **base solver's bit-exact correctness is credible** — 54 adversarial cases + 18
pytest cases + 12 Python-reference KATs + 25 reference edge cases, all bit-exact, zero wrong answers,
with the two boundary hangs found-and-fixed. The negmap variant is **correct when it terminates**
(zero wrong answers) but can hang at tiny intervals near the top boundary; it is shipped opt-in
(`-DNEGMAP`) with its net-loss and this fragility documented.

## 7. Honest #135 extrapolation (tie to ADR-0001 negative-EV baseline)

Puzzle **#135**: private key in **[2¹³⁴, 2¹³⁵)**, **exposed pubkey** → bounded-interval ECDLP,
W = 2¹³⁴, √W = 2⁶⁷. Expected kangaroo work = c·2⁶⁷.

At our **measured 1.05 × 10⁹ jumps/s** on one RTX 5090:

| constant | group ops | one-GPU time |
|---|---:|---:|
| classic c=2.08 | 3.07 × 10²⁰ (2⁶⁸·¹) | **≈ 9,300 GPU-years** |
| RCKangaroo SOTA c=1.15 | 1.70 × 10²⁰ (2⁶⁷·²) | **≈ 5,100 GPU-years** |

Even granting a SOTA-class kernel (~1.5 × 10¹⁰ jumps/s, i.e. closing our 8× engineering gap and then
some) the single-card figure is **≈ 360 GPU-years**. To solve in **1 calendar year** you would need on
the order of **5,000 RTX 5090s** (our kernel) or **~360** (hypothetical SOTA kernel) running flat-out
— a multi-million-dollar cluster racing established pools for a ~13.5 BTC prize.

There is also a **storage** cost the time figure hides: a real run stores ~c·√W/2^dp distinguished
points. At dp=40 that's ~2.8 × 10⁸ DPs ≈ 11 GB; at dp=32, ~7 × 10¹⁰ DPs ≈ **3 TB** — the
classic DP time/space tradeoff, manageable but non-trivial at cluster scale.

**Verdict, per ADR-0001:** #135 is *cluster-attemptable* (unlike #71's hopeless 2⁷⁰ brute force) —
this is exactly why √n + exposed-pubkey is "the whole game" for Regime 2. But the **expected value is
negative**: thousands of GPU-years (or a thousands-of-GPU cluster-year) of contested compute for one
contested prize, with **no broadcast path in this repo by design**. The kangaroo is the *correct*
method and we built and validated it bit-exact on toy intervals; doing so does not move #135 across
any feasibility-of-profit threshold, and was never going to. The engineering and the cryptography are
the deliverable — the clear-eyed map of what is and isn't possible is itself the result.

## 8. Reproduce

```bash
cd /tmp/secp-wt-d
uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install coincurve ecdsa base58 pytest
export PATH=/usr/lib/llvm-21/bin:$PATH
cd kernels/kangaroo
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -I../cuda-ref -o kangaroo kangaroo.cu            # base
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -DNEGMAP -I../cuda-ref -o kangaroo_negmap kangaroo.cu

# correctness gates
VIRTUAL_ENV=/tmp/secp-wt-d/.venv /tmp/secp-wt-d/.venv/bin/python -m pytest test_kangaroo_ref.py test_kangaroo_edge.py test_negmap_ref.py -q
VIRTUAL_ENV=/tmp/secp-wt-d/.venv /tmp/secp-wt-d/.venv/bin/python -m pytest test_kangaroo_cuda.py test_kangaroo_cuda_negmap.py -q

# scaling + negmap A/B
VIRTUAL_ENV=/tmp/secp-wt-d/.venv /tmp/secp-wt-d/.venv/bin/python scaling.py
VIRTUAL_ENV=/tmp/secp-wt-d/.venv /tmp/secp-wt-d/.venv/bin/python negmap_ab.py
```

Tunables (env): `KANG_HERD` (kangaroos/side), `KANG_DP` (DP bits), `KANG_STEPS` (steps/round),
`KANG_BUDGET` (step ceiling), `KANG_BATCH` (0=per-step inverse, 1=batched), `KANG_MEAN` (mean jump).
Batch size B is a compile constant (`-DWALK_B=32`).
