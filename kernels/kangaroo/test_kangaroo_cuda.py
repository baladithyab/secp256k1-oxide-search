"""Bit-exact gate for the CUDA kangaroo solver (Track D).

Builds the kernel once (session-scoped), then runs the compiled solver on each
oracle KAT and asserts it prints the EXACT d (== d_hex). The solver is given ONLY
(lo, hi, Q_compressed) on argv — never d. No tolerance.

Run: VIRTUAL_ENV=.venv .venv/bin/python -m pytest kernels/kangaroo/test_kangaroo_cuda.py -v -s
"""
import json
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "kangaroo")
NVCC = "/usr/local/cuda/bin/nvcc"


def _load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def solver_bin():
    """Compile the CUDA kangaroo solver once for the whole test session."""
    src = os.path.join(HERE, "kangaroo.cu")
    cmd = [
        NVCC, "-O3", "-arch=sm_120",
        "-I", os.path.join(HERE, "..", "cuda-ref"),
        "-o", BIN, src,
    ]
    env = dict(os.environ, PATH="/usr/lib/llvm-21/bin:" + os.environ.get("PATH", ""))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    assert r.returncode == 0, f"nvcc build failed:\n{r.stdout}\n{r.stderr}"
    assert os.path.exists(BIN), "solver binary not produced"
    return BIN


def _solve(solver_bin, kat, timeout=120):
    lo_hex = kat["lo_hex"].lstrip("0") or "0"
    hi_hex = kat["hi_hex"].lstrip("0") or "0"
    q_hex = kat["Q_compressed_hex"]
    r = subprocess.run(
        [solver_bin, "0x" + lo_hex, "0x" + hi_hex, q_hex],
        capture_output=True, text=True, timeout=timeout,
    )
    # The solver prints the recovered d as a line "d=<hex>"; parse it.
    assert r.returncode == 0, f"solver exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    found = None
    for line in r.stdout.splitlines():
        if line.startswith("d="):
            found = int(line[2:], 16)
    assert found is not None, f"solver printed no d= line:\n{r.stdout}\n{r.stderr}"
    return found


KATS = [
    "kat16_0x1.json", "kat16_0x2.json", "kat16_0x3.json",
    "kat20_0x1.json", "kat20_0x2.json", "kat20_0x3.json",
    "kat24_0x1.json", "kat24_0x2.json", "kat24_0x3.json",
    "kat32_0x1.json", "kat32_0x2.json", "kat32_0x3.json",
]


@pytest.mark.parametrize("katfile", KATS)
def test_cuda_recovers_d_bit_exact(solver_bin, katfile):
    kat = _load(katfile)
    expected = int(kat["d_hex"], 16)
    recovered = _solve(solver_bin, kat)
    assert recovered == expected, (
        f"{katfile}: recovered 0x{recovered:x} != expected 0x{expected:x}"
    )
