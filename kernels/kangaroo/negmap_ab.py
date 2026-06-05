#!/usr/bin/env python3
"""Negation-map A/B: measure the sqrt(2) jumps-to-solve lift (Track D, acceptance #4).

The negation map's value is ALGORITHMIC: folding {P,-P} halves the walk space, so
collision needs ~1/sqrt(2) as many jumps (~1.41x fewer). It is NOT a jumps/s win
(canonicalization adds a tiny per-step cost). So the honest metric is TOTAL JUMPS TO
SOLVE, base vs negmap, at the same interval, averaged over seeds (median for the
heavy tail). Both must recover d bit-exact (correctness must survive the optimization).

Usage: VIRTUAL_ENV=.venv .venv/bin/python kernels/kangaroo/negmap_ab.py
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "kangaroo")
NEG = os.path.join(HERE, "kangaroo_negmap")

SIZES = [int(x) for x in os.environ.get("AB_SIZES", "40").split(",")]
# Many seeds so the median is stable (kangaroo solve-time is heavy-tailed).
SEEDS = ["0x%x" % s for s in range(1, int(os.environ.get("AB_NSEEDS", "16")) + 1)]
# Moderate herd: each solve spans MANY rounds at fine step granularity, so round-floor quantization
# averages out and the algorithmic sqrt(2) shows cleanly. Per-run timeout caps the heavy tail.
HERD = os.environ.get("AB_HERD", "4096")
STEPS = os.environ.get("AB_STEPS", "64")
TIMEOUT = int(os.environ.get("AB_TIMEOUT", "60"))

JUMPS_RE = re.compile(r"jumps=([0-9.eE+]+)")


def run(binp, kat):
    lo = "0x" + (kat["lo_hex"].lstrip("0") or "0")
    hi = "0x" + (kat["hi_hex"].lstrip("0") or "0")
    exp = int(kat["d_hex"], 16)
    env = dict(os.environ, KANG_HERD=HERD, KANG_STEPS=STEPS)
    try:
        r = subprocess.run([binp, lo, hi, kat["Q_compressed_hex"]],
                           capture_output=True, text=True, env=env, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, float("nan")    # heavy-tail run: drop it (counted as not-OK, excluded from median)
    out = r.stdout + r.stderr
    d = None
    for line in r.stdout.splitlines():
        if line.startswith("d="):
            d = int(line[2:], 16)
    m = JUMPS_RE.search(out)
    jumps = float(m.group(1)) if m else float("nan")
    return (d == exp), jumps


def main():
    print(f"{'bits':>4} {'sqrtW':>11} {'base_jumps':>11} {'neg_jumps':>11} "
          f"{'lift':>5} {'base_c':>6} {'neg_c':>6} {'n':>4}")
    print("-" * 70)
    for bits in SIZES:
        bj, nj = [], []
        sqrtW = None
        for seed in SEEDS:
            f = os.path.join(HERE, f"kat{bits}_{seed}.json")
            if not os.path.exists(f):
                continue
            kat = json.load(open(f))
            W = int(kat["hi_hex"], 16) - int(kat["lo_hex"], 16)
            sqrtW = math.isqrt(W)
            ok_b, jb = run(BASE, kat)
            ok_n, jn = run(NEG, kat)
            if ok_b and not math.isnan(jb):
                bj.append(jb)
            if ok_n and not math.isnan(jn):
                nj.append(jn)
            if not (ok_b and ok_n):
                print(f"  drop bits={bits} seed={seed} base_ok={ok_b} neg_ok={ok_n}", file=sys.stderr)
        mb = statistics.median(bj) if bj else float("nan")
        mn = statistics.median(nj) if nj else float("nan")
        lift = mb / mn if mn else float("nan")
        print(f"{bits:>4} {sqrtW:>11} {mb:>11.3e} {mn:>11.3e} {lift:>5.2f} "
              f"{mb/sqrtW:>6.2f} {mn/sqrtW:>6.2f} {len(bj)}/{len(nj)}")


if __name__ == "__main__":
    main()
