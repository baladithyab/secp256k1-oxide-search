# Wave 1 — CUDA-C secp256k1 reference scanner

Single-GPU brute-force secp256k1 scanner for the no-pubkey case, plus a host correctness
harness that is **bit-exact** against `scripts/kat_oracle.py`. This is the correctness
reference for later waves (cuda-oxide port, batched-Montgomery fast path).

## Files

| File | Purpose |
|------|---------|
| `secp256k1_field.cuh` | Field arithmetic mod p (add/sub/mul/sqr/inv). |
| `secp256k1_ec.cuh`    | Jacobian point add/double, double-and-add scalar mult, affine conversion, compressed SEC pubkey. |
| `hash.cuh`            | On-GPU SHA-256 (big-endian) + RIPEMD-160 (little-endian) → hash160. |
| `secp256k1_ref.cu`    | Device kernels + host harness (verify / scan / bench), tiny hand-rolled JSON parser. |
| `test_vectors.json`   | 8 canonical KAT vectors (priv, compressed pub, hash160). |
| `build.sh`, `Makefile`| Build wrappers around the exact nvcc command. |

## Limb layout

- **8 × uint32, little-endian limb order** (`limb[0]` = least-significant 32 bits).
- 32×32→64 multiplies with explicit 64-bit carry propagation. No float anywhere in the EC/hash path.

## Field reduction

Field prime `p = 2^256 − 2^32 − 977`.
- Multiply: schoolbook 256×256 → 512-bit product, then **fold the high 256 bits** using
  the identity `2^256 ≡ 2^32 + 977 (mod p)`. The fold is applied to the high half, the
  resulting tiny overflow is folded once more, then a single conditional subtract of `p`
  (`fe_reduce_once`) brings the result into `[0, p)`.
- Add/sub: 8-limb add/sub with carry/borrow; a top carry on add is folded with `2^32+977`,
  a borrow on sub adds `p` back. Final conditional subtract on add.
- **Modular inverse**: Fermat `a^(p−2) mod p` via square-and-multiply over the fixed exponent
  bits (simplest correct choice for the reference; the batched-Montgomery trick is Wave 3).

## Per-key pipeline (matches the oracle exactly)

1. `P = k·G` — Jacobian double-and-add, MSB→LSB over the 256-bit scalar.
2. Affine `x = X/Z², y = Y/Z³ mod p` (one Fermat inverse of Z).
3. Compressed SEC pubkey (33 B): `0x02`/`0x03` (y parity) + X **big-endian**.
4. `sha256` of the 33-byte pubkey (BE schedule + BE length).
5. `ripemd160` of the 32-byte SHA-256 digest (**LE** word load + **LE** length + **LE** output words) → hash160 (20 B).
6. `memcmp` hash160 vs target (20 bytes, exact).

The SHA-256 (big-endian) vs RIPEMD-160 (little-endian) split is the classic bug; both are
verified bit-exact against the Python oracle.

## Build

```sh
# absolute nvcc path — the apt shims are stale CUDA 12.0 and mishandle sm_120
bash kernels/cuda-ref/build.sh
#   or
make -C kernels/cuda-ref
```

Exact command:
```
/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ \
    -o kernels/cuda-ref/secp256k1_ref kernels/cuda-ref/secp256k1_ref.cu
```

## Run (from the repo root, so the default `kernels/cuda-ref/test_vectors.json` path resolves)

```sh
./kernels/cuda-ref/secp256k1_ref verify        # → "ALL 8 VECTORS PASS"
./kernels/cuda-ref/secp256k1_ref scan          # → "SCAN: FOUND key=… matches vector[0].priv EXACTLY"
./kernels/cuda-ref/secp256k1_ref bench 1048576 # → median keys/s + results/wave1-cuda-keys-per-sec.csv
./kernels/cuda-ref/secp256k1_ref all 1048576   # verify + scan + bench (default mode)
```

Set `VEC_PATH=/path/to/vectors.json` to run against a different vector file (e.g. a fresh
`.venv/bin/python scripts/kat_oracle.py vectors 32` dump — verified 32/32 pass).

- **verify**: runs all vectors through the full GPU pipeline, asserts hash160 bit-exact.
- **scan**: target = `vector[0].hash160`; scans the inclusive window `[priv−1000, priv+1000]`
  on the GPU and asserts it reports the exact `vector[0].priv`.
- **bench**: full scalar-mult → hash160 → compare per key; reports the **median keys/s over
  12 iterations** and writes `results/wave1-cuda-keys-per-sec.csv` (median + min/max + an
  `nvidia-smi temperature/power/utilization` thermal line).

## Measured throughput (RTX 5090, sm_120)

~27 M keys/s median for the unoptimised reference (plain double-and-add + per-key Fermat
inverse). This is intentionally the *correct* baseline, not the fast path; constant-factor
optimisation (batched Montgomery inversion, windowed/precomputed G, wNAF) is Wave 3.
