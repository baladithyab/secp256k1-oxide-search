# Wave 3 — Opt C (negation map) + hash160 throughput investigation (notes)

Builds on `wave3-fastpath-notes.md` (Opt A: affine +G walk + batched Montgomery inversion, 98×,
BATCH=256). Same hardware/toolchain: RTX 5090 (sm_120), `/usr/local/cuda/bin/nvcc -ccbin clang-14
-O3 -arch=sm_120`, median over 12 iters, GPU idle at session start. All variants stay **bit-exact**
vs `scripts/kat_oracle.py` (no tolerance). Opt A is kept byte-identical for honest A/B: the neg-map
code is gated behind `-DNEGMAP`, and with it undefined every `if (NEG)` branch is dead-code-
eliminated (regression-guarded — Opt A's 8/8 + 1100/1100 + scan are unchanged after the refactor).

---

## Opt C — Negation-map symmetry (the one EC opt that stacks on the brute-force walk)

**Idea.** Keys `k` and `n−k` (n = group order) produce points `P` and `−P = (x, p−y)`. They share
the **same x-coordinate**; the compressed-SEC encoding differs only in the parity prefix byte
(`0x02`↔`0x03`), because `−y mod p = p−y` flips parity when `p` is odd. So from **one** affine walk
point we get **two** candidate private keys by hashing both `02‖x` and `03‖x`. In code the entire
trick is `pub[0] ^= 0x01` between two `hash160_33` calls — **zero extra field arithmetic**.

This HALVES the EC/walk work per address covered (one affine-add now covers 2 keys) but does NOT
reduce hash work (still 2 hash160/point). Per ADR-0003 this is the *only* listed opt that stacks on
the no-pubkey brute-force walk: GLV/endomorphism scatters the scalar and is therefore **out of
scope** for the sequential walk (and was not implemented).

**Bit-exact gate (new).** `kat_oracle.py negmap START COUNT` emits, per key `k`, both `k` and `n−k`
with their compressed pubkeys + hash160s + the shared x. The kernel's `negverify` mode walks
consecutive keys and checks BOTH emitted parity-hashes against the oracle:
- `negmap 1 1100` → **ALL 1100 PAIRS PASS** (both parities). Drives the doubling lane at k=2 and
  4 batch boundaries; `neg[0]` n-k-hash = `adde4c73…` is key `n−1`'s address (upper-half key).
- Adversarial reviewer (independent `coincurve`, 13 cases): k=1, k=2 (doubling), straddle N/2,
  exact N/2 floor/ceil with verified mirror-swap, top-of-range N−4..N−1, 32-bit & 64-bit limb-carry
  boundaries, 300-key multi-batch walk, doubling forced at interior slot i=127 and at the
  **advance slot i=255** (the doubled point becomes the next batch's chained base). **No divergence
  found.** The "flip parity byte == encode(−P)" claim was independently confirmed from first
  principles (`flip_parity(pub(k))` byte-identical to `pub(N−k)`).

**Result (official A/B, same thermal session, RTX 5090, BATCH=256, median/12):**

| | addresses/s | affine-points/s | thermal |
|---|---:|---:|---|
| Opt A (re-run this session) | **3,288,032,041** | 3,288,032,041 | 46 °C/44 W/5% → 50 °C/168 W |
| Opt C (neg-map) | **3,967,492,364** | 1,983,746,182 | → 63 °C/358 W |

**Neg-map lift = 1.21× (+20.7%).** Robust across a confirmation pass (Opt A 3.278 G, Opt C 3.983 G).

**Why 1.21× and not 2× (and not near-zero) — the measurement is self-interpreting.** Opt C's
affine-**points**/s = **0.603×** Opt A's, because each point now does 2 hash160. The two limiting
cases bracket this exactly:
- *Fully hash-bound*: 2 hashes/point ⇒ points/s halves to 0.50, addresses/s flat ⇒ **1.0× lift**.
- *Fully EC-bound*: the 2nd hash is free ⇒ points/s stays 1.0, addresses/s doubles ⇒ **2.0× lift**.

Measured points/s = 0.603 (lift 1.21×) lands between the bounds, leaning hash-side: the kernel is
**roughly balanced, marginally hash-bound at full occupancy**. The serial-work model `lift = 2/(2−e)`
back-solves to a *recoverable* EC fraction `e ≈ 0.34` — smaller than the ~0.65 static EC instruction
share (below) because the multiply pipe (EC) and shift/logic pipe (hash) **overlap**, so halving EC
work only helps to the extent EC was the binding pipe. The marginal cost of the 2nd hash is ~0.20 ns
/point. **This refines — does not contradict — the Opt A notes**, which guessed "near-zero if hashing
dominates": hashing does *not* fully dominate, so neg-map yields a real, modest +20.7%.

> Honest headline: **the affine walk is roughly balanced (≈⅔ EC / ⅓ hash), so the one EC opt that
> stacks (neg-map) yields a real but modest +20.7%, not the naive 2×.** Further EC-only tricks
> (which don't stack anyway) would yield even less.

---

## Task 2 — hash160 throughput investigation (SASS-driven)

Disassembled the Opt A bench kernel (`cuobjdump --dump-sass secp256k1_fast`, `kfast_bench` →
`walk_thread<MODE_BENCH>`, one ~47,672-instruction flattened `__forceinline`+`unroll` function).

**Resource usage (`--dump-resource-usage`):** `kfast_bench` — **255 reg/thread** (capped),
**8,672-byte stack frame**, 1024 B shared, LDL=358 / STL=45. The frame is `prod[BATCH+1]` =
`prod[257]` (257×32 B = 8,224 B); too large for registers, lives in the (cached) stack frame.

**Instruction mix (static, %):** MOV 29% (bookkeeping/immediates), IADD.64 15.6% (EC carry chains),
HFMA2-zero-idiom 11.5% (Blackwell immediate materialization, *not* FP math), **IMAD.WIDE.U32 9.3%
(EC 32×32→64 mul)**, **SHF 6.3% (hash rotr/rol)**, **LOP3 2.8% (hash ch/maj/f)**, IADD3 2.5%
(hash 3-input adds), LDL/STL 0.8% (`prod[]` stack traffic, clustered at frame offset ~8,584–8,656).

**Verdict — the Opt A notes' "hash-bound, ~7 field-muls/key" claim is NOT confirmed by the SASS.**
Per-key dynamic estimate ≈ **65% EC field-arith / 35% hash160**; multiply-pipe (EC) vs shift/logic-
pipe (hash) ≈ **0.92 : 1** — nearly balanced, leaning EC. Each affine add is really ~6 full `fe_mul`
+ heavy 64-bit carry chains, not "~7 field-muls" total. This is consistent with Opt C's measured
0.603 points-ratio (an independent confirmation from a totally different method).

**The three proposed hash optimizations — SASS resolves all three without a build:**
1. **Register residency of `block[64]`/`w[64]`** — *already done by nvcc*. The hash region shows
   **IMAD.WIDE = 0** (no array-index math): the buffers are fully scalarized into registers. The
   "array-residency lesson from cuda-oxide" simply does not bite here. **No lift available.**
2. **Loop-unrolling the 64-round SHA / 80-step RIPEMD** — *already done* (`#pragma unroll` flattened
   them; there is no rolled loop left). **No lift available.**
3. **Sharing the SHA first-block between the two neg-map parities** — *not viable, and now provably
   so*. The two messages differ only in byte 0 → SHA word `w[0]`. But round 0 of the compression
   function consumes `w[0]` into all of `a..h`, and `w[16]=w[0]+σ0(w[1])+w[9]+σ1(w[14])` depends on
   `w[0]`; everything downstream of round 0 diverges. The shareable prefix is **zero** rounds. SHA
   is sequential — confirmed, no sharing. (RIPEMD likewise loads byte 0 into `X[0]`, used round 0.)

**The actual highest-leverage lever the SASS points to** is *not* the hash path at all: it is the
**`prod[257]` 8,672-byte stack frame** capping occupancy at 255 reg/thread (all LDL/STL cluster
there). Shrinking `BATCH` or streaming the prefix products so the giant array never materializes
would lift occupancy. This is a structural change to the Montgomery batch, not a hash change, and is
**left as future work** (Opt A already swept BATCH and found 256 robust for *its* register pressure;
the neg-map binary carries 2 live hash160 so its optimum may differ — `secp256k1_negmap_b64`/`_b128`
were built for this sweep but not benched within this session's budget).

**Bottom line for Task 2:** the hash path has **no easy throughput win** — nvcc already scalarizes
and unrolls it, the two SHA blocks cannot be shared, and the kernel is only ~⅓ hash-bound anyway.
The dominant remaining lever is EC/occupancy (the `prod[]` stack frame), not the hashes.

---

## Cumulative picture & updated #71 estimate

Opt A delivered 98× over Wave-1 (33.5 M → 3.28 G addresses/s). Opt C stacks **1.207×** on top →
**≈3.97 G addresses/s**, a cumulative **≈118× over Wave-1 baseline**.

#71 covers the key range [2⁷⁰, 2⁷¹), i.e. **2⁷⁰ candidate keys** (range width, not 2⁷¹). At
3.97 G addresses/s, one RTX 5090 exhausts the full range in
2⁷⁰ / 3.97×10⁹ ≈ 2.97×10¹¹ s ≈ **≈9,400 GPU-years** (expected **≈4,700 GPU-years** to a 50%-probability
hit). Opt A alone was ≈11,600 GPU-years (full range); Opt C trims it ~17%. As ADR-0003 states
plainly: still hopeless single-GPU, still negative-EV at cluster scale. The constant factor is real
and honestly measured; it does not cross any feasibility threshold — and it was never going to.

**Caveat specific to #71's bounded range.** For a *small* puzzle key `k`, the mirror `n−k` lands
near 2²⁵⁶ — **outside** the [1, 2⁷¹] puzzle interval. So neg-map's "two keys per point" does not
give two *in-range* candidates for a bounded forward scan of #71; its value there is the +20.7%
throughput on the hash/EC pipeline, realized by covering 2 addresses per affine add regardless of
which interval they fall in. (For an *exposed-pubkey* kangaroo over the full group — ADR-0003
Regime 2 — the negation map's √2 jump reduction is the more impactful use, out of scope here.)
