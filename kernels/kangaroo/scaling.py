#!/usr/bin/env python3
"""Empirical sqrt(n) scaling harness for the Track-D kangaroo solver.

For each interval size, runs the solver on several seeds and records:
  - total jumps to solve (the sqrt(n) quantity: expected ~ c*sqrt(W))
  - wall-clock ms
  - empirical c = jumps / sqrt(W)
Averages over seeds (single kangaroo runs have ~2-3x variance), then reports the
size-to-size ratio to confirm the ~2x-per-+2-bits law. Also records correctness
(recovered d == oracle d) for every run — the scaling table is only meaningful if
every point is a bit-exact solve.

Usage: VIRTUAL_ENV=.venv .venv/bin/python kernels/kangaroo/scaling.py
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "kangaroo")

SIZES = [int(x) for x in os.environ.get("SCALE_SIZES", "32,36,40,44,48").split(",")]
SEEDS = ["0x1", "0x2", "0x3", "0x4", "0x5", "0x6", "0x7", "0x8", "0x9"]
# Small FIXED herd so M << sqrt(W) for ALL listed sizes (the linear regime where total jumps ~ c*sqrt(W)
# is independent of M and the sqrt(n) law is visible). Large herds collide in round 0 and pin the floor.
HERD = os.environ.get("SCALE_HERD", "512")

JUMPS_RE = re.compile(r"jumps=([0-9.eE+]+)")
TIME_RE = re.compile(r"time=([0-9.]+)ms")
SPK_RE = re.compile(r"total_steps/kangaroo=([0-9]+)")


def run_one(kat, herd):
    lo = "0x" + (kat["lo_hex"].lstrip("0") or "0")
    hi = "0x" + (kat["hi_hex"].lstrip("0") or "0")
    q = kat["Q_compressed_hex"]
    exp = int(kat["d_hex"], 16)
    env = dict(os.environ, KANG_HERD=herd)
    # fine granularity for accurate jumps-to-solve: small steps/round
    env["KANG_STEPS"] = os.environ.get("SCALE_STEPS", "256")
    r = subprocess.run([BIN, lo, hi, q], capture_output=True, text=True, env=env, timeout=600)
    out = r.stdout + r.stderr
    d = None
    for line in r.stdout.splitlines():
        if line.startswith("d="):
            d = int(line[2:], 16)
    jumps = float(JUMPS_RE.search(out).group(1)) if JUMPS_RE.search(out) else float("nan")
    tms = float(TIME_RE.search(out).group(1)) if TIME_RE.search(out) else float("nan")
    ok = (d == exp)
    return ok, jumps, tms, d, exp


def main():
    print(f"{'bits':>4} {'sqrtW':>12} {'mean_jumps':>12} {'mean_c':>7} {'med_c':>7} "
          f"{'mean_ms':>9} {'ratio':>6} {'allOK':>6}")
    print("-" * 78)
    prev_jumps = None
    for bits in SIZES:
        cs, jumpss, tmss, allok = [], [], [], True
        sqrtW = None
        for seed in SEEDS:
            f = os.path.join(HERE, f"kat{bits}_{seed}.json")
            if not os.path.exists(f):
                continue
            kat = json.load(open(f))
            W = int(kat["hi_hex"], 16) - int(kat["lo_hex"], 16)
            sqrtW = math.isqrt(W)
            ok, jumps, tms, d, exp = run_one(kat, HERD)
            if not ok:
                allok = False
                print(f"  !! bits={bits} seed={seed} FAIL d={d:x} exp={exp:x}", file=sys.stderr)
            jumpss.append(jumps)
            cs.append(jumps / sqrtW)
            tmss.append(tms)
        mean_j = statistics.mean(jumpss)
        ratio = (mean_j / prev_jumps) if prev_jumps else float("nan")
        print(f"{bits:>4} {sqrtW:>12} {mean_j:>12.3e} {statistics.mean(cs):>7.2f} "
              f"{statistics.median(cs):>7.2f} {statistics.mean(tmss):>9.1f} "
              f"{ratio:>6.2f} {str(allok):>6}")
        prev_jumps = mean_j


if __name__ == "__main__":
    main()
