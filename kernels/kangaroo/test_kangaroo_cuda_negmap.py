"""Bit-exact gate for the NEGATION-MAP CUDA solver (Track D, Regime 2).

Builds the kernel with -DNEGMAP and asserts it recovers d EXACTLY on the KATs. The
negation map adds signed-distance + canonicalization + fruitless-cycle escape on top
of the base solver, so it gets its own bit-exact gate (correctness must survive the
sqrt(2) optimization — the whole point of the A/B).

Run: VIRTUAL_ENV=.venv .venv/bin/python -m pytest kernels/kangaroo/test_kangaroo_cuda_negmap.py -q
"""
import json
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "kangaroo_negmap")
NVCC = "/usr/local/cuda/bin/nvcc"


def _load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def negmap_bin():
    src = os.path.join(HERE, "kangaroo.cu")
    cmd = [NVCC, "-O3", "-arch=sm_120", "-DNEGMAP",
           "-I", os.path.join(HERE, "..", "cuda-ref"), "-o", BIN, src]
    env = dict(os.environ, PATH="/usr/lib/llvm-21/bin:" + os.environ.get("PATH", ""))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    assert r.returncode == 0, f"nvcc -DNEGMAP build failed:\n{r.stdout}\n{r.stderr}"
    return BIN


def _solve(binp, kat, timeout=120):
    lo = "0x" + (kat["lo_hex"].lstrip("0") or "0")
    hi = "0x" + (kat["hi_hex"].lstrip("0") or "0")
    r = subprocess.run([binp, lo, hi, kat["Q_compressed_hex"]],
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f"solver exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    found = None
    for line in r.stdout.splitlines():
        if line.startswith("d="):
            found = int(line[2:], 16)
    assert found is not None, f"no d= line:\n{r.stdout}\n{r.stderr}"
    return found


KATS = [
    "kat16_0x1.json", "kat16_0x2.json", "kat16_0x3.json",
    "kat20_0x1.json", "kat20_0x2.json", "kat20_0x3.json",
    "kat24_0x1.json", "kat24_0x2.json", "kat24_0x3.json",
    "kat32_0x1.json", "kat32_0x2.json", "kat32_0x3.json",
]


@pytest.mark.parametrize("katfile", KATS)
def test_negmap_cuda_bit_exact(negmap_bin, katfile):
    kat = _load(katfile)
    expected = int(kat["d_hex"], 16)
    recovered = _solve(negmap_bin, kat)
    assert recovered == expected, f"{katfile}: 0x{recovered:x} != 0x{expected:x}"
