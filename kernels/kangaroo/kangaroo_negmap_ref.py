"""Negation-map kangaroo — Python reference (Track D, ADR-0003 Regime 2).

The negation map identifies P with -P (they share x; y <-> p-y). Walking on the
CANONICAL representative C(P) = (x, min(y, p-y)) folds the two into one, so the
effective search space shrinks by ~2 -> ~sqrt(2) fewer jumps to collision. This is
the regime (within-interval / full-group walk) where the map genuinely pays off
(unlike the bounded #71 forward scan, where the mirror falls out of range — see the
wave3-negmap ERRATUM).

THE BOOKKEEPING (where correctness lives):
  A step at canonical point C with jump j computes C+J_j, then canonicalizes:
        eps = +1 if (C+J_j).y is already even (canonical), else -1 (we negated).
        dlog(C(C+J_j)) = eps * (dlog(C) + s_j)            [as integers, never mod n]
  Because every tracked scalar stays << sqrt(W)*max_jump << n, the sign just flips a
  bounded integer; no mod-n reduction is needed.

  TAME tracks a signed integer m with dlog(C_tame) = m  (starts at a known scalar).
  WILD tracks (sign s in {+1,-1}, signed integer w) with dlog(C_wild) = s*d' + w
        (starts at C(Q'), so s=+1, w=0, dlog = d').
  Update on a step with eps:   m   -> eps*(m + s_j)
                               (s,w)-> (eps*s, eps*(w + s_j))
  At a tame/wild x-collision (same canonical x => same canonical point):
        m = s*d' + w   =>   d' = s*(m - w)      (exact integer; verified by cand*G==Q)

FRUITLESS CYCLES (the negation-map hazard):
  Canonicalization can create a 2-cycle: if C+J_j negates and the resulting point
  re-selects the same index j, then -(C+J_j)+J_j = -C, whose canonical form is C
  again -> infinite loop (probability ~1/(2K) per step). Mitigation: detect a repeat
  of the point from two steps ago (x == x_prev2) and take an ALTERNATE deterministic
  jump index (idx ^ 1) for that step. The alternate jump is still a known-scalar jump,
  so the distance bookkeeping stays exact.
"""
from __future__ import annotations

import math

from coincurve import PublicKey

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P_FIELD = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F


def _mulG(k: int):
    k %= N
    return None if k == 0 else PublicKey.from_valid_secret(k.to_bytes(32, "big"))


def _add(P, Q):
    if P is None:
        return Q
    if Q is None:
        return P
    try:
        return PublicKey.combine_keys([P, Q])
    except Exception:
        return None  # P + Q = O


def _xy(P):
    raw = P.format(compressed=False)
    return int.from_bytes(raw[1:33], "big"), int.from_bytes(raw[33:65], "big")


def _canon(P):
    """Return (canonical_point, was_negated). Canonical = the rep with EVEN y."""
    if P is None:
        return None, False
    x, y = _xy(P)
    if y & 1:  # odd -> negate to (x, p-y) which is even
        return PublicKey(bytes([0x02 if (P_FIELD - y) & 1 == 0 else 0x03]) + x.to_bytes(32, "big")), True
    return P, False


def _make_jumps(num_jumps, mean_jump):
    scalars, points = [], []
    state = 0x9E3779B97F4A7C15
    span = max(1, 2 * mean_jump)
    for _ in range(num_jumps):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        s = 1 + (state % span)
        scalars.append(s)
        points.append(_mulG(s))
    return scalars, points


def solve_kat_negmap(Q_hex, lo, hi, num_jumps=32, max_steps=None, dp_bits=None,
                     num_kangaroos=64):
    W = hi - lo
    sqrtW = math.isqrt(max(W, 2))
    if max_steps is None:                          # per-kangaroo step budget (herd of ~num_kangaroos/side)
        max_steps = 64 * (sqrtW // max(1, num_kangaroos) + 1) + 20000
    if dp_bits is None:
        dp_bits = max(0, (W.bit_length() // 2) - 4)
    dp_mask = (1 << dp_bits) - 1

    scalars, points = _make_jumps(num_jumps, max(1, sqrtW // 2))
    E_SCALAR = 1                                  # fixed additive escape jump (+G); keeps a in {+-1}
    E_POINT = _mulG(E_SCALAR)
    CYC_WINDOW = 64                               # detect a point reappearing within this many steps
    MAX_CYC = 256                                 # bound the cycle traversal (safety)

    Q = PublicKey(bytes.fromhex(Q_hex))
    Qp = _add(Q, _mulG(N - (lo % N)))             # Q' = (d-lo)*G

    # Unified dlog-state: dlog(C) = a*d' + w (signed integers, NEVER reduced mod n).
    #   tame: a=0, w=m (a known absolute scalar);  wild: a=+-1, w (starts a=1,w=0 -> dlog=d').
    # One _step advances any state by its jump; one _escape teleports out of a fruitless cycle.
    def _step(C, a, w, idx_override=None):
        x, _ = _xy(C)
        idx = (x % num_jumps) if idx_override is None else idx_override
        cC, neg = _canon(_add(C, points[idx]))
        eps = -1 if neg else 1
        # dlog(C+J) = (a*d'+w) + s_idx ; after canon eps: (eps*a)*d' + eps*(w+s_idx)
        return cC, eps * a, eps * (w + scalars[idx]), idx

    def _escape(C, a, w):
        """Teleport out of a fruitless cycle: find the cycle's UNIQUE min-x point (a pure function
        of the cycle, so merged kangaroos escape identically) then apply the fixed +E jump."""
        start_x, _ = _xy(C)
        best = (C, a, w); best_x = start_x
        cC, ca, cw = C, a, w
        for _ in range(MAX_CYC):
            cC, ca, cw, _ = _step(cC, ca, cw)
            xx, _ = _xy(cC)
            if xx == start_x:                     # returned to start -> full cycle traversed
                break
            if xx < best_x:
                best_x = xx; best = (cC, ca, cw)
        Cb, ab, wb = best
        cC2, neg = _canon(_add(Cb, E_POINT))
        eps = -1 if neg else 1
        return cC2, eps * ab, eps * (wb + E_SCALAR)

    def _solve_pair(a_t, w_t, a_w, w_w):
        # at a same-point collision: a_t*d'+w_t = a_w*d'+w_w => d' = (w_w - w_t)/(a_t - a_w)
        da = a_t - a_w
        if da == 0:
            return None
        num = w_w - w_t
        if num % da != 0:
            return None
        return lo + num // da

    # MULTI-KANGAROO herd. A single tame+single wild walk is uniquely fragile under the negation
    # map (one trapped walker stalls the whole search); a herd escapes cycles statistically and is
    # the configuration that actually runs on the GPU (thousands of kangaroos). Each entry is a
    # mutable [type, C, a, w] with dlog(C)=a*d'+w; type 'T' has a=0 (known scalar), 'W' has a=+-1.
    M = max(8, min(num_kangaroos, sqrtW + 1))
    states = []
    stride = max(1, W // M)
    for g in range(M):                            # tame: spread known start scalars across [0,W]
        st = stride // 2 + g * stride
        if st < 1:
            st = 1
        C, neg = _canon(_mulG(st))
        states.append(['T', C, 0, (-st if neg else st)])
    for g in range(M):                            # wild: Q' + offset*G, offset spread across [0,W]
        off = stride // 2 + g * stride
        base = _add(Qp, _mulG(off)) if off else Qp
        C, neg = _canon(base)
        states.append(['W', C, (-1 if neg else 1), (-off if neg else off)])

    tame_dp, wild_dp = {}, {}                      # x -> (a, w)
    wins = [dict() for _ in range(len(states))]    # per-kangaroo cycle-detection window

    for step in range(max_steps):
        for i, stt in enumerate(states):
            typ, C, a, w = stt[0], stt[1], stt[2], stt[3]
            x, _ = _xy(C)
            if x in wins[i] and step - wins[i][x] <= CYC_WINDOW:    # fruitless cycle -> escape
                C, a, w = _escape(C, a, w)
                wins[i] = {}
            else:
                C, a, w, _ = _step(C, a, w)
            wins[i][x] = step
            stt[1], stt[2], stt[3] = C, a, w
            cx, _ = _xy(C)
            if (cx & dp_mask) == 0:
                if typ == 'T':
                    if cx in wild_dp:
                        aw, ww = wild_dp[cx]
                        cand = _solve_pair(a, w, aw, ww)
                        if cand is not None and _verify(cand, Q, lo, hi):
                            return cand
                    tame_dp.setdefault(cx, (a, w))
                else:
                    if cx in tame_dp:
                        at, wt = tame_dp[cx]
                        cand = _solve_pair(at, wt, a, w)
                        if cand is not None and _verify(cand, Q, lo, hi):
                            return cand
                    wild_dp.setdefault(cx, (a, w))

    raise RuntimeError(f"negmap: no collision after {max_steps} steps (W={W})")


def _verify(cand, Q, lo, hi):
    if cand < lo or cand > hi:
        return False
    cg = _mulG(cand)
    return cg is not None and cg.format(compressed=True) == Q.format(compressed=True)
