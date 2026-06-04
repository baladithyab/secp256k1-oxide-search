"""Distinguished-point (DP) collision store — kangaroo-mode stub.

In Pollard-kangaroo mode (exposed pubkey, README "Mode 2") all workers walk the
*same* interval and emit distinguished points: curve points whose x-coordinate
has ``d`` trailing zero bits. The coordinator stores DPs keyed by x-coordinate.
A **collision** — the same x reached by a *tame* and a *wild* kangaroo — yields
the private key via ``d = (a_tame - a_wild) mod n`` (the standard kangaroo
recovery).

This is a STUB for Wave 4: the full kangaroo solver lives in the kernel layer.
Here we implement just enough of the collision store to be exercised by the
allocator's kangaroo skeleton and by tests:

* an x-keyed dict (the README's "hash map keyed by x-coord"),
* tame/wild kind tracking,
* collision detection on insert (returns recovered scalar via the supplied
  ``solve`` callback or the recorded walk distances),
* an observed DP-rate accumulator the allocator reads each epoch.

It does NOT compute curve arithmetic and does NOT produce a spendable artifact.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class DPRecord:
    x: int  # distinguished-point x-coordinate (the map key)
    kind: str  # "tame" or "wild"
    dist: int  # accumulated walk distance for this kangaroo
    worker_id: str
    t: float  # arrival timestamp (monotonic)


@dataclass
class Collision:
    x: int
    tame: DPRecord
    wild: DPRecord
    recovered_key: Optional[int] = None


class DPStore:
    """Thread-safe x-keyed DP store with collision detection (kangaroo stub).

    Parameters
    ----------
    interval_lo, interval_hi:
        The shared search interval ``[lo, hi)`` (informational; not used for
        arithmetic in the stub).
    clock:
        Injectable monotonic clock for deterministic DP-rate tests.
    """

    def __init__(
        self,
        interval_lo: int = 0,
        interval_hi: int = 0,
        clock=time.monotonic,
    ):
        self.interval_lo = interval_lo
        self.interval_hi = interval_hi
        self._clock = clock
        self._lock = threading.RLock()
        self._by_x: Dict[int, DPRecord] = {}
        self._collision: Optional[Collision] = None
        # DP-rate telemetry: count + window start, read once per epoch.
        self._dp_count = 0
        self._window_start = clock()

    # -- ingestion ---------------------------------------------------------

    def add_batch(
        self,
        worker_id: str,
        points: List[Tuple[int, str, int]],
        solve: Optional[Callable[[DPRecord, DPRecord], int]] = None,
    ) -> Optional[Collision]:
        """Insert a batch of DPs ``(x, kind, dist)``; return a Collision if any.

        ``kind`` is ``"tame"`` or ``"wild"``. A collision is a same-``x`` hit
        across the two kinds. If a ``solve`` callback is supplied it is invoked
        with ``(tame_record, wild_record)`` to recover the scalar; otherwise we
        store the recovered key as ``tame.dist - wild.dist`` (the toy
        distance-difference recovery used by tests).
        """
        with self._lock:
            for x, kind, dist in points:
                self._dp_count += 1
                if kind not in ("tame", "wild"):
                    raise ValueError(f"bad DP kind: {kind!r}")
                prior = self._by_x.get(x)
                rec = DPRecord(
                    x=x, kind=kind, dist=dist, worker_id=worker_id, t=self._clock()
                )
                if prior is None:
                    self._by_x[x] = rec
                    continue
                if prior.kind == kind:
                    # Same-kind re-hit: keep the earlier one (a fruitless cycle
                    # in real kangaroo; nothing to recover).
                    continue
                # Cross-kind collision -> recover the key.
                tame, wild = (
                    (prior, rec) if prior.kind == "tame" else (rec, prior)
                )
                if solve is not None:
                    key = solve(tame, wild)
                else:
                    key = tame.dist - wild.dist
                self._collision = Collision(
                    x=x, tame=tame, wild=wild, recovered_key=key
                )
                return self._collision
            return None

    # -- telemetry / introspection ----------------------------------------

    def collision(self) -> Optional[Collision]:
        with self._lock:
            return self._collision

    def size(self) -> int:
        with self._lock:
            return len(self._by_x)

    def observed_dp_rate(self) -> float:
        """DPs/second since the last call, then reset the window.

        The allocator reads this each epoch to update its posterior on
        time-to-collision (kangaroo arm).
        """
        with self._lock:
            now = self._clock()
            elapsed = max(now - self._window_start, 1e-9)
            rate = self._dp_count / elapsed
            self._dp_count = 0
            self._window_start = now
            return rate
