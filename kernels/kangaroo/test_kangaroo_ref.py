"""Bit-exact gate for the Python reference kangaroo (Track D).

TDD red phase: these tests load the oracle-generated KAT instances {Q, lo, hi}
and assert the solver recovers d EXACTLY (== d_hex). No tolerance. A kangaroo
that finds a "close" d is wrong.

Run: VIRTUAL_ENV=.venv .venv/bin/python -m pytest kernels/kangaroo/test_kangaroo_ref.py -v
"""
import json
import os

import pytest

from kangaroo_ref import solve_kat

HERE = os.path.dirname(__file__)


def _load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


# (bits, seed) pairs covering tiny -> 32-bit across several seeds.
KATS = [
    ("kat16_0x1.json"), ("kat16_0x2.json"), ("kat16_0x3.json"),
    ("kat20_0x1.json"), ("kat20_0x2.json"), ("kat20_0x3.json"),
    ("kat24_0x1.json"), ("kat24_0x2.json"), ("kat24_0x3.json"),
    ("kat32_0x1.json"), ("kat32_0x2.json"), ("kat32_0x3.json"),
]


@pytest.mark.parametrize("katfile", KATS)
def test_recovers_d_bit_exact(katfile):
    kat = _load(katfile)
    lo = int(kat["lo_hex"], 16)
    hi = int(kat["hi_hex"], 16)
    Q_hex = kat["Q_compressed_hex"]
    expected_d = int(kat["d_hex"], 16)

    recovered = solve_kat(Q_hex, lo, hi)

    assert recovered == expected_d, (
        f"{katfile}: recovered d=0x{recovered:x} != expected 0x{expected_d:x}"
    )
