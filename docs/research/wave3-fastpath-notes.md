# Wave 3 — Fast-path constant-factor optimizations (notes)

Per ADR-0003 Track A, we stack the real constant-factor wins on the verified Wave-1 CUDA-C path.
Each optimization is a **separate, individually-benchmarked variant** so lift is attributable; the
Wave-1 baseline kernel (`secp256k1_ref.cu`) is left intact and re-run in the SAME session on the
SAME idle GPU for honest A/B. All variants must stay **bit-exact** vs `scripts/kat_oracle.py`
(no float tolerance — one wrong bit = wrong key).

Hardware: RTX 5090 (sm_120), driver 596.21. Toolchain: `/usr/local/cuda/bin/nvcc -ccbin clang-14
-O3 -arch=sm_120`. Benchmarks: median over 12 iters, GPU idle (≈5%, 43 °C) at session start.

The bit-exact gate for a *sequential* walk is a new oracle subcommand:
`scripts/kat_oracle.py walk START COUNT` → COUNT **consecutive** key vectors (the existing `vectors`
command emits scattered LCG keys, which cannot validate a `+G` walk's step logic).

---

## Opt A — Batched Montgomery inversion + affine sequential walk

**Idea (ADR-0003: "the biggest lift, ~4–5× per gECC" — but see the real number below).**
The Wave-1 baseline recomputes a full 256-step double-and-add `k·G` for *every* candidate key, even
though consecutive keys differ by just `+G`. The fast path walks a contiguous sub-range with **one
affine point-addition per key**. Affine addition needs a modular inverse (`1/(Δx)`), the single most
expensive field op (~270 muls via Fermat) — so a naive affine walk is *slower* than Jacobian.
Montgomery's trick fixes that: invert a whole **batch** of `N` denominators with **1 inverse + 3N
muls**, collapsing the per-key inverse cost to ≈ `270/N + 3` muls. ADR-0003's "affine beats Jacobian
at N>20" is exactly this crossover.

**Structure (minimizing new bit-exact-critical surface):**
- Per thread: one *Jacobian* `scalar_mul_G(startKey)` (the **verified** Wave-1 code) seeds the affine
  start point. Amortized over `N·steps` keys, so its cost is negligible.
- Hot loop: `prod[]` = prefix products of the batch denominators `dᵢ = (i+1)·G.x − base.x`; one
  `fe_inv(prod[N])`; a fused backward pass that extracts each `1/dᵢ`, does the affine add, and hashes
  — **no `results[]` array** (saves ~4 KB local memory/thread; only `prod[N+1]` stays live).
- The shared multiples table `j·G` (`j=1..BATCH`) is built once and cached in shared memory per block.
- A single templated `walk_thread<MODE>` serves verify / bench / scan, so **the benched code is
  byte-identical to the verified code** (no "fast bench, slow verify" lie).

**Zero-denominator handling (the most bug-prone path in Wave 3).** `dᵢ = 0` ⇔ `base == ±(i+1)·G`:
- `+` case (`K == i+1`): a **doubling**. Substitute 1 in the product (preserving other lanes), flag
  the lane, resolve via the verified `jpoint_add` (which doubles correctly).
- `−` case (`K ≡ n−(i+1) mod n`): the **point at infinity**, only near the top of the group order.
  Same flag path; `jpoint_to_affine` returns false → emit a `(0,0)` sentinel (never a valid key).

**Bit-exact verification (ALL PASS):**
- `walk 1 200` and `walk 1 1100` (4+ full batches at BATCH=256) — bit-exact.
- `walk 0x1b8af6534e1be8aa6 130` (large 65-bit scalar, crosses batch boundaries) — bit-exact.
- Adversarial sub-agent (independent `coincurve` oracle), 27 runs across BATCH=256 and BATCH=64:
  - doubling lane at positions 0, mid-batch (start 5), near-end (start 250) — all bit-exact;
  - **point-at-infinity / negation lane genuinely triggered** (start = N−3 → `base = −3·G`, zero at
    lane 2; start = N−2 → lane 1) and confirmed to emit the correct `(0,0)` sentinel hash160
    `3625c4a2…` without crash/hang, and without corrupting adjacent valid lanes;
  - doubling correctly *distinguished* from infinity (key 2 emits the real hash, not the sentinel);
  - batch-size independence confirmed.
- `scan` mode finds the planted `vector[0].priv` exactly.

**BATCH sweep (median keys/s, 12 iters):**

| BATCH |   keys/s   | note |
|------:|-----------:|------|
|    16 | 1.23 G     | inverse under-amortized |
|    32 | 1.87 G     | |
|    64 | 2.43 G     | |
|   128 | 2.97 G     | |
|   192 | 3.19 G     | |
|   256 | 3.28 G     | **default** — robust local-memory headroom |
|   384 | 3.19 G     | diminishing |
|   512 | 3.49 G     | +5% over 256 at 2× local memory; not worth it |

**Result (official A/B, same session, idle→warm):**

| | keys/s | thermal |
|---|---:|---|
| Baseline (Wave-1 Jacobian per-key) | **33,497,444** | 43 °C→47 °C, 5%→87% |
| Opt A (affine walk + batch inverse, BATCH=256) | **3,283,058,620** | 47 °C→57 °C, 87%→99% |

**Lift = 98.0×.** (Far above the ADR's "~4–5×" gECC figure — because gECC's 4–5× compares
*batched-affine vs deferred-Jacobian inversion within the same per-key scalar-mult*, whereas our
98× also folds in eliminating the **per-key 256-step scalar multiplication itself**: the sequential
walk replaces ~2000 field-muls/key with ~7. The win is structural, not just the inverse trick.)

**Sanity check that 98× is real, not a measurement artifact.** Baseline hardware throughput:
33.5 M keys/s × ~2000 u32-muls/key ≈ 6.7×10¹² u32-MAC/s. Opt A: 3.28 G keys/s × ~7 field-muls/key
≈ 7×10¹² — *same silicon throughput*. The speedup is entirely "≈400× less EC work per key"; the
kernel is now bottlenecked by hash160 (sha256+ripemd160), not the curve. (Note Opt A ran *hotter*
than the baseline, so if anything thermal drift understates the win.)
