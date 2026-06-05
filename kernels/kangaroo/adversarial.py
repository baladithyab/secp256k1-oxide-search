#!/usr/bin/env python3
"""Adversarial correctness sweep for the kangaroo solver (Track D review).

Constructs KATs with d at adversarial positions and random seeds, runs base + negmap
binaries with a per-run timeout, and asserts recovered d == constructed d EXACTLY.
Distinguishes WRONG-ANSWER (the real prize) from HANG (timeout). The device k_verify
makes a wrong answer require the verify kernel to also be wrong, so we hunt mainly for
hangs and sign-bookkeeping bugs in negmap.
"""
import os
import subprocess
import sys

from coincurve import PublicKey

HERE = os.path.dirname(os.path.abspath(__file__))


def Qh(d):
    return PublicKey.from_valid_secret(d.to_bytes(32, "big")).format(compressed=True).hex()


def solve(binp, lo, hi, d, timeout, herd="256"):
    env = dict(os.environ, KANG_HERD=herd)
    try:
        r = subprocess.run([binp, hex(lo), hex(hi), Qh(d)],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "HANG"
    for line in r.stdout.splitlines():
        if line.startswith("d="):
            return int(line[2:], 16)
    return "NODP"


def cases():
    # (label, lo, hi, d)
    out = []
    for bits in (8, 10, 12, 16, 20, 24):
        lo = 1 << bits
        hi = (1 << (bits + 1)) - 1
        W = hi - lo
        for nm, d in [("lo", lo), ("lo+1", lo + 1), ("hi", hi), ("hi-1", hi - 1),
                      ("mid", lo + W // 2), ("mid+1", lo + W // 2 + 1), ("quarter", lo + W // 4)]:
            out.append((f"{bits}b/{nm}", lo, hi, d))
    # asymmetric / narrow
    out.append(("asym", 0x3A7F1, 0x3B200, 0x3AC00))
    out.append(("narrow2", 0x80000, 0x80002, 0x80001))
    out.append(("narrow3-lo", 0x90000, 0x90003, 0x90000))
    out.append(("narrow16", 0x100000, 0x100010, 0x100007))
    # random-ish seeds at 20/24-bit via LCG
    s = 0xABCDEF
    for i in range(8):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        bits = 20 if i % 2 == 0 else 24
        lo = 1 << bits; hi = (1 << (bits + 1)) - 1
        d = lo + (s % (hi - lo))
        out.append((f"rand{i}/{bits}b", lo, hi, d))
    return out


def main():
    base = os.path.join(HERE, "kangaroo")
    neg = os.path.join(HERE, "kangaroo_negmap")
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    binp = base if which == "base" else neg
    timeout = int(os.environ.get("ADV_TIMEOUT", "45"))
    npass = wrong = hang = nodp = 0
    fails = []
    for label, lo, hi, d in cases():
        got = solve(binp, lo, hi, d, timeout)
        if got == d:
            npass += 1
        elif got == "HANG":
            hang += 1; fails.append((label, hex(lo), hex(hi), hex(d), "HANG"))
        elif got == "NODP":
            nodp += 1; fails.append((label, hex(lo), hex(hi), hex(d), "NODP/FAIL"))
        else:
            wrong += 1; fails.append((label, hex(lo), hex(hi), hex(d), f"WRONG={hex(got)}"))
    print(f"[{which}] pass={npass} WRONG={wrong} hang={hang} nodp/fail={nodp} (total {len(cases())})")
    for f in fails:
        print("   FAIL", f)
    return 1 if (wrong or hang or nodp) else 0


if __name__ == "__main__":
    sys.exit(main())
