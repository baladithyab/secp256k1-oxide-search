"""Bit-exact gate for the NEGATION-MAP reference kangaroo (Track D, Regime 2).

The negation map folds {P, -P} to one canonical representative -> ~sqrt(2) fewer
distinct points -> fewer jumps to collision. The subtle part is the signed-distance
+ sign-bit bookkeeping (negation flips the discrete-log sign) and the fruitless-cycle
mitigation. These tests prove the neg-map solver still recovers d EXACTLY, including
at adversarial d positions, before the same logic is ported to CUDA.
"""
import math

import pytest
from coincurve import PublicKey

from kangaroo_negmap_ref import solve_kat_negmap

N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def _Q_hex(d):
    return PublicKey.from_valid_secret(d.to_bytes(32, "big")).format(compressed=True).hex()


# This pure-Python+coincurve reference validates the negation-map SIGN / CYCLE-ESCAPE / COLLISION
# math (the bit-exact gate). It uses a large herd (M=256/side): a single tame+wild walk is uniquely
# fragile under the negation map (one trapped walker stalls everything), and at tiny W the canonical
# space is so small that the per-walker cycle window false-fires constantly — both are artifacts of
# the toy scale, NOT the algorithm, and both vanish at GPU scale (40+ bit, huge space, thousands of
# kangaroos). So the reference is exercised at 14-bit, where it is fast AND robust; the GPU validates
# correctness up to 32-bit (test_kangaroo_cuda_negmap.py) and measures the actual sqrt(2) lift. The
# 'hi'/'quarter' positions specifically guard the fruitless-cycle fix (they trapped the naive walker).
@pytest.mark.parametrize("where", ["lo", "lo+1", "hi", "hi-1", "mid", "quarter"])
def test_negmap_recovers_d(where):
    bits = 14
    lo = 1 << bits
    hi = (1 << (bits + 1)) - 1
    W = hi - lo
    d = {"lo": lo, "lo+1": lo + 1, "hi": hi, "hi-1": hi - 1,
         "mid": lo + W // 2, "quarter": lo + W // 4}[where]
    recovered = solve_kat_negmap(_Q_hex(d), lo, hi, num_kangaroos=256, max_steps=8000)
    assert recovered == d, f"where={where}: 0x{recovered:x} != 0x{d:x}"
