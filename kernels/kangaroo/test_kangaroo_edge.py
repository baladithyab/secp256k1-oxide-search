"""Adversarial edge-case gate for the reference kangaroo (Track D).

The oracle picks d pseudo-randomly, so these tests construct KATs with d placed at
ADVERSARIAL positions in the interval (lo, lo+1, hi, hi-1, exact midpoint) and at
tiny widths. d near a boundary stresses the shifted frame d' near 0 or near W;
midpoint coincides with the tame start. Every case must recover d bit-exact.
"""
import math

import pytest
from coincurve import PublicKey

from kangaroo_ref import N, solve_kat


def _Q_hex(d: int) -> str:
    return PublicKey.from_valid_secret(d.to_bytes(32, "big")).format(compressed=True).hex()


def _case(bits: int, where: str):
    lo = 1 << bits
    hi = (1 << (bits + 1)) - 1
    W = hi - lo
    d = {
        "lo": lo,
        "lo+1": lo + 1,
        "hi": hi,
        "hi-1": hi - 1,
        "mid": lo + W // 2,        # coincides with tame start in shifted frame
        "mid+1": lo + W // 2 + 1,
        "quarter": lo + W // 4,
    }[where]
    return _Q_hex(d), lo, hi, d


@pytest.mark.parametrize("bits", [16, 20, 24])
@pytest.mark.parametrize("where", ["lo", "lo+1", "hi", "hi-1", "mid", "mid+1", "quarter"])
def test_d_at_adversarial_position(bits, where):
    Q_hex, lo, hi, d = _case(bits, where)
    recovered = solve_kat(Q_hex, lo, hi)
    assert recovered == d, f"bits={bits} where={where}: got 0x{recovered:x} != 0x{d:x}"


@pytest.mark.parametrize("bits", [8, 10, 12, 16])
def test_tiny_intervals(bits):
    """Very small intervals must still terminate with the exact d (no hang)."""
    lo = 1 << bits
    hi = (1 << (bits + 1)) - 1
    d = lo + (hi - lo) // 3
    recovered = solve_kat(_Q_hex(d), lo, hi)
    assert recovered == d
