"""Pure-Python CPU "stub kernel" — stands in for the GPU brute-force scanner.

The real worker invokes the Wave-1/3 CUDA kernel via subprocess to sweep a
leased block (scalar-mult -> hash160 -> compare against the target rmd160). For
tests we need something with the *same shape* but no GPU: scan a block of keys,
derive a per-key digest, and report a hit when it matches the planted target
digest.

To keep tests dependency-free, fast, and deterministic — while still exercising
"derive a value from the key and compare exactly, no tolerance" — we use a
SHA-256-based digest as the stand-in for hash160. This is explicitly a *test
oracle*, not the secp256k1 pipeline: it has the avalanche property (key ``k``
and ``k+1`` give uncorrelated digests), so a brute-force scan really must visit
the planted key to find it, exactly like the real search.

This module performs NO elliptic-curve math and produces NO spendable artifact.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional


def target_digest(key: int) -> bytes:
    """Stand-in for hash160(pubkey(key)): SHA-256 of the key's 32-byte BE form.

    Avalanche stand-in only — NOT the secp256k1 hash160. Used so the planted
    target is a function of the key and a scan must hit the exact key.
    """
    return hashlib.sha256(key.to_bytes(32, "big")).digest()


@dataclass
class SweepResult:
    """Outcome of sweeping one block."""

    found: bool
    key: Optional[int]  # the matching private key, if found
    keys_scanned: int  # keys actually examined (for keys/s telemetry)
    elapsed_s: float


def sweep_block(
    lo: int,
    hi: int,
    target: bytes,
    progress_cb=None,
    progress_every: int = 4096,
) -> SweepResult:
    """Scan ``[lo, hi)`` for a key whose :func:`target_digest` equals ``target``.

    Parameters
    ----------
    lo, hi:
        Half-open block bounds.
    target:
        The planted target digest (32 bytes from :func:`target_digest`).
    progress_cb:
        Optional ``callable(keys_scanned_so_far, current_key)`` invoked roughly
        every ``progress_every`` keys; used by the worker to emit heartbeats
        with ``block_progress``.
    progress_every:
        Heartbeat cadence in keys.

    Returns
    -------
    SweepResult
        ``found=True`` with the matching ``key`` on the first match, else
        ``found=False`` after the whole block is scanned.
    """
    t0 = time.monotonic()
    scanned = 0
    for key in range(lo, hi):
        if target_digest(key) == target:
            return SweepResult(
                found=True,
                key=key,
                keys_scanned=scanned + 1,
                elapsed_s=time.monotonic() - t0,
            )
        scanned += 1
        if progress_cb is not None and scanned % progress_every == 0:
            progress_cb(scanned, key)
    return SweepResult(
        found=False,
        key=None,
        keys_scanned=scanned,
        elapsed_s=time.monotonic() - t0,
    )
