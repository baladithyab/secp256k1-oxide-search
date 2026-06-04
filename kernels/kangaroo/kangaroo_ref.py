"""Pollard's kangaroo (lambda) ECDLP solver — Python reference (Track D).

This is the ALGORITHM reference that the CUDA kernel ports. It uses `coincurve`
for the EC group ops (so the group math is trusted) and implements the kangaroo
method itself from scratch. Its job: prove the jump table, distinguished-point
collision, and distance bookkeeping recover d BIT-EXACT before we commit the same
logic to a GPU kernel (where porting bugs are the #1 risk).

Method (bounded interval ECDLP): given Q = d*G with d in [lo, hi], recover d.

  Interval shift: attack Q' = Q - lo*G = (d-lo)*G, so the unknown d' = d-lo lives
  in [0, W] where W = hi-lo. Distances are tracked as ORDINARY integers. Because
  every distance stays << 2^128 << n (the group order), the collision identity
        d' = base_tame + dist_tame - dist_wild
  holds as an EXACT integer equation — no mod-n arithmetic is ever needed. This is
  the design choice that keeps the whole solver bit-exact with field-only math.

  Two herds:
    - TAME kangaroos start at known scalars near the middle of [0, W] (so their
      absolute scalar = start + accumulated jumps is always known).
    - WILD kangaroos start at Q' (unknown scalar d'), and accumulate the SAME kind
      of jumps; their tracked "distance" is the offset from Q'.

  Pseudo-random jumps: jump index = x(point) mod K selects one of K precomputed
  jump points J[i] = s_i * G (s_i = jump scalars). A kangaroo at point P with
  tracked distance t does: P += J[i]; t += s_i. The walk is deterministic in the
  point, so a tame and a wild kangaroo that ever land on the same point will follow
  identical subsequent jumps and stay merged — the classic kangaroo collision.

  Distinguished points (DP): a point is "distinguished" if x(P) has `dp_bits`
  trailing zero bits. We only store DPs in the hash table (keyed by x). A tame DP
  meeting a wild DP at the same x yields the collision and thus d.

This reference is intentionally simple/serial; the GPU parallelises over thousands
of kangaroos. Same math, same jump table convention.
"""
from __future__ import annotations

import hashlib

from coincurve import PublicKey
from coincurve.keys import PrivateKey

# secp256k1 group order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


# ---- EC helpers via coincurve (trusted group ops) -------------------------

def _scalar_mul_G(k: int) -> PublicKey | None:
    """k*G as a coincurve PublicKey, or None for k % n == 0 (point at infinity)."""
    k %= N
    if k == 0:
        return None
    return PublicKey.from_valid_secret(k.to_bytes(32, "big"))


def _point_add(P: PublicKey | None, Q: PublicKey | None) -> PublicKey | None:
    """P + Q on the curve. Handles the identity (None). Returns None if P + Q = O."""
    if P is None:
        return Q
    if Q is None:
        return P
    try:
        return PublicKey.combine_keys([P, Q])
    except Exception:
        # combine raised => result is the point at infinity (P == -Q)
        return None


def _x_of(P: PublicKey | None) -> int:
    """Affine x-coordinate of P as an int (0 for the identity, which never distinguishes)."""
    if P is None:
        return 0
    # uncompressed: 0x04 || X(32) || Y(32)
    raw = P.format(compressed=False)
    return int.from_bytes(raw[1:33], "big")


# ---- jump table -----------------------------------------------------------

def _make_jumps(num_jumps: int, mean_jump: int):
    """Precompute K jump scalars s_i and their points J_i = s_i*G.

    The jump scalars are deterministic pseudo-random values in [1, 2*mean_jump] so
    the AVERAGE jump size is ~mean_jump (kangaroo-optimal mean = ~sqrt(W)/2). The
    mean is the load-bearing parameter: collision time is ~2*sqrt(W) only when the
    mean jump matches sqrt(W)/2; an oversized mean makes the herds' DP trails too
    sparse to intersect. Deterministic (LCG seeded by a fixed constant, independent
    of the secret) so a KAT run is reproducible. Returns (scalars, points).
    """
    scalars = []
    points = []
    state = 0x9E3779B97F4A7C15  # fixed seed (golden-ratio constant), secret-independent
    span = max(1, 2 * mean_jump)
    for _ in range(num_jumps):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        s = 1 + (state % span)  # in [1, 2*mean_jump]
        scalars.append(s)
        points.append(_scalar_mul_G(s))
    return scalars, points


# ---- the solver -----------------------------------------------------------

def solve_kat(Q_compressed_hex: str, lo: int, hi: int,
              dp_bits: int | None = None, num_jumps: int | None = None,
              max_steps: int | None = None, seed: int = 1) -> int:
    """Recover d in [lo, hi] s.t. d*G == Q (compressed hex). Returns d (exact int).

    Raises RuntimeError if the walk exceeds max_steps without a collision (should
    not happen for the KAT interval sizes with the chosen parameters).
    """
    W = hi - lo
    Q = PublicKey(bytes.fromhex(Q_compressed_hex))

    # shifted target Q' = Q - lo*G  => unknown d' = d - lo in [0, W]
    neg_loG = _scalar_mul_G(N - (lo % N))   # = -(lo*G)
    Qp = _point_add(Q, neg_loG)

    # --- parameter heuristics tuned for the small KAT regime ---
    import math
    Wf = max(W, 2)
    sqrtW = int(math.isqrt(Wf))
    # K=32 jump scalars (classic JLP layout); mean jump ~ sqrt(W)/2 (kangaroo-optimal)
    if num_jumps is None:
        num_jumps = 32
    if dp_bits is None:
        # DP density: want a handful of jumps between DPs. For tiny W, use 0-ish.
        dp_bits = max(0, (W.bit_length() // 2) - 3)
    if max_steps is None:
        max_steps = 200 * (sqrtW + 1) + 100000

    scalars, points = _make_jumps(num_jumps, max(1, sqrtW // 2))
    dp_mask = (1 << dp_bits) - 1

    # hash tables of distinguished points: x -> (absolute_scalar_if_tame_else_distance, is_tame)
    # We store the kangaroo's *absolute candidate scalar*: for tame that's its true
    # scalar; for wild it's (d') + accumulated jumps. A tame/wild x-match gives d'.
    tame_dp = {}
    wild_dp = {}

    # interleave one tame and one wild kangaroo (serial reference). Everything is
    # tracked in the SHIFTED frame (Q' = (d-lo)*G). Tame starts at the middle of
    # [0, W] with known shifted-scalar W/2; wild starts at Q' with offset 0.
    tame_start = W // 2
    tame_pt = _scalar_mul_G(tame_start) if tame_start > 0 else None
    tame_scalar = tame_start

    wild_pt = Qp
    wild_dist = 0  # wild candidate shifted-scalar = d' + wild_dist (unknown d')

    steps = 0
    while steps < max_steps:
        steps += 1

        # --- advance tame ---
        xj = _x_of(tame_pt) % num_jumps
        tame_pt = _point_add(tame_pt, points[xj])
        tame_scalar += scalars[xj]
        xt = _x_of(tame_pt)
        if (xt & dp_mask) == 0:
            if xt in wild_dp:
                # collision: tame_scalar == d' + wild_dist  => d' = tame_scalar - wild_dist
                wd = wild_dp[xt]
                dprime = tame_scalar - wd
                cand = lo + dprime
                if _verify(cand, Q, lo, hi):
                    return cand
            tame_dp[xt] = tame_scalar

        # --- advance wild ---
        xj = _x_of(wild_pt) % num_jumps
        wild_pt = _point_add(wild_pt, points[xj])
        wild_dist += scalars[xj]
        xw = _x_of(wild_pt)
        if (xw & dp_mask) == 0:
            if xw in tame_dp:
                ts = tame_dp[xw]
                dprime = ts - wild_dist
                cand = lo + dprime
                if _verify(cand, Q, lo, hi):
                    return cand
            wild_dp[xw] = wild_dist

    raise RuntimeError(f"no collision after {max_steps} steps (W={W}, dp_bits={dp_bits})")


def _verify(cand: int, Q: PublicKey, lo: int, hi: int) -> bool:
    """Final bit-exact check: cand in [lo,hi] and cand*G == Q."""
    if cand < lo or cand > hi:
        return False
    cg = _scalar_mul_G(cand)
    if cg is None:
        return False
    return cg.format(compressed=True) == Q.format(compressed=True)
