"""Range registry: leases DISJOINT fixed-size blocks with crash-tolerant TTL.

The coordinator owns a global keyspace ``[lo, hi)`` and partitions it into
fixed-size blocks of ``block_size`` keys (the last block may be short if the
range is not a multiple of ``block_size``). Each block is in exactly one state:

    FREE     -> never leased, or reclaimed after a lease expired
    LEASED   -> handed to a worker, awaiting ack/heartbeats (TTL armed)
    SWEPT    -> worker reported the block fully swept with no hit

A lease has a TTL. If the owning worker never acks, or stops sending
heartbeats, the lease expires and the block returns to the FREE pool so another
worker can pick it up — this is the entire crash-recovery story for brute-force
mode (README "Crash recovery: unacked leases expire and return to the free
pool").

Concurrency: every public method takes an internal lock, so the registry is
safe to share across the coordinator's per-connection threads.

Disjointness is structural: blocks are derived from a fixed integer grid
``lo + k*block_size``; a block is leased to at most one worker at a time, so two
*concurrent* LEASED blocks can never overlap. The tests assert this invariant
directly.
"""

from __future__ import annotations

import enum
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class LeaseState(enum.Enum):
    FREE = "free"
    LEASED = "leased"
    SWEPT = "swept"


@dataclass
class Lease:
    """A leased block of the keyspace.

    ``lo``/``hi`` are the inclusive-low/exclusive-high key bounds of the block.
    ``expires_at`` is a monotonic-clock deadline; past it the lease is dead.
    """

    lease_id: int
    block_index: int
    lo: int
    hi: int
    worker_id: str
    state: LeaseState
    leased_at: float
    expires_at: float
    acked: bool = False

    def overlaps(self, other: "Lease") -> bool:
        """True iff the two half-open intervals share any key."""
        return self.lo < other.hi and other.lo < self.hi

    def as_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "block_index": self.block_index,
            "lo": self.lo,
            "hi": self.hi,
            "worker_id": self.worker_id,
            "state": self.state.value,
        }


@dataclass
class _Block:
    """Internal per-block bookkeeping on the fixed grid."""

    index: int
    lo: int
    hi: int
    state: LeaseState = LeaseState.FREE
    lease: Optional[Lease] = field(default=None)


class RangeRegistry:
    """Owns ``[lo, hi)``; leases disjoint fixed-size blocks with TTL reclaim.

    Parameters
    ----------
    lo, hi:
        Global keyspace bounds, half-open ``[lo, hi)``.
    block_size:
        Keys per block. The final block is truncated to ``hi`` if the range is
        not an exact multiple.
    lease_ttl:
        Seconds a lease survives without a renewing heartbeat/ack before it is
        considered dead and reclaimable.
    clock:
        Injectable monotonic clock (seconds, float). Defaults to
        :func:`time.monotonic`; tests inject a fake clock for determinism.
    """

    def __init__(
        self,
        lo: int,
        hi: int,
        block_size: int,
        lease_ttl: float = 30.0,
        clock=time.monotonic,
    ):
        if hi <= lo:
            raise ValueError("hi must be > lo")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")
        self.lo = lo
        self.hi = hi
        self.block_size = block_size
        self.lease_ttl = lease_ttl
        self._clock = clock
        self._lock = threading.RLock()
        self._lease_ids = itertools.count(1)

        # Build the fixed integer grid of blocks. Disjointness is guaranteed by
        # construction: block k is [lo+k*bs, min(lo+(k+1)*bs, hi)).
        self._blocks: List[_Block] = []
        idx = 0
        cur = lo
        while cur < hi:
            top = min(cur + block_size, hi)
            self._blocks.append(_Block(index=idx, lo=cur, hi=top))
            cur = top
            idx += 1

        # lease_id -> Lease, for the currently-active (LEASED, unexpired) leases
        # plus historical ones we keep for FOUND/observability. Active set is the
        # source of truth for disjointness checks.
        self._leases_by_id: Dict[int, Lease] = {}

    # -- introspection -----------------------------------------------------

    @property
    def n_blocks(self) -> int:
        return len(self._blocks)

    def snapshot_states(self) -> List[LeaseState]:
        with self._lock:
            self._reclaim_expired_locked()
            return [b.state for b in self._blocks]

    def active_leases(self) -> List[Lease]:
        """Currently-LEASED, unexpired leases (reclaims expired ones first)."""
        with self._lock:
            self._reclaim_expired_locked()
            return [
                b.lease
                for b in self._blocks
                if b.state is LeaseState.LEASED and b.lease is not None
            ]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            self._reclaim_expired_locked()
            out = {s.value: 0 for s in LeaseState}
            for b in self._blocks:
                out[b.state.value] += 1
            return out

    def is_exhausted(self) -> bool:
        """True when no block is FREE and none is LEASED (all SWEPT)."""
        with self._lock:
            self._reclaim_expired_locked()
            return all(b.state is LeaseState.SWEPT for b in self._blocks)

    # -- TTL reclaim -------------------------------------------------------

    def _reclaim_expired_locked(self) -> List[int]:
        """Return any LEASED block whose lease deadline has passed to FREE.

        Must be called with the lock held. Returns the list of reclaimed
        block indices (for logging/metrics).
        """
        now = self._clock()
        reclaimed: List[int] = []
        for b in self._blocks:
            if b.state is LeaseState.LEASED and b.lease is not None:
                if now >= b.lease.expires_at:
                    dead_id = b.lease.lease_id
                    self._leases_by_id.pop(dead_id, None)
                    b.state = LeaseState.FREE
                    b.lease = None
                    reclaimed.append(b.index)
        return reclaimed

    def reclaim_expired(self) -> List[int]:
        """Public sweep for expired leases; safe to call periodically."""
        with self._lock:
            return self._reclaim_expired_locked()

    # -- lease lifecycle ---------------------------------------------------

    def lease_block(self, worker_id: str) -> Optional[Lease]:
        """Lease the lowest-index FREE block to ``worker_id``.

        Returns the new :class:`Lease`, or ``None`` if no block is free. The
        lease starts unacked with a TTL deadline armed; the worker must ack
        (and then heartbeat) to keep it alive.
        """
        with self._lock:
            self._reclaim_expired_locked()
            for b in self._blocks:
                if b.state is LeaseState.FREE:
                    now = self._clock()
                    lease = Lease(
                        lease_id=next(self._lease_ids),
                        block_index=b.index,
                        lo=b.lo,
                        hi=b.hi,
                        worker_id=worker_id,
                        state=LeaseState.LEASED,
                        leased_at=now,
                        expires_at=now + self.lease_ttl,
                        acked=False,
                    )
                    b.state = LeaseState.LEASED
                    b.lease = lease
                    self._leases_by_id[lease.lease_id] = lease
                    return lease
            return None

    def ack_lease(self, lease_id: int, worker_id: str) -> bool:
        """Mark a lease acked and renew its TTL. Returns False if unknown/stale."""
        with self._lock:
            self._reclaim_expired_locked()
            lease = self._leases_by_id.get(lease_id)
            if lease is None or lease.worker_id != worker_id:
                return False
            lease.acked = True
            lease.expires_at = self._clock() + self.lease_ttl
            return True

    def renew_lease(self, lease_id: int, worker_id: str) -> bool:
        """Renew the TTL on heartbeat. Returns False if the lease is gone/stale."""
        with self._lock:
            self._reclaim_expired_locked()
            lease = self._leases_by_id.get(lease_id)
            if lease is None or lease.worker_id != worker_id:
                return False
            lease.expires_at = self._clock() + self.lease_ttl
            return True

    def mark_swept(self, lease_id: int, worker_id: str) -> bool:
        """Transition a leased block to SWEPT. Returns False if unknown/stale."""
        with self._lock:
            self._reclaim_expired_locked()
            lease = self._leases_by_id.get(lease_id)
            if lease is None or lease.worker_id != worker_id:
                return False
            b = self._blocks[lease.block_index]
            if b.lease is None or b.lease.lease_id != lease_id:
                return False
            b.state = LeaseState.SWEPT
            lease.state = LeaseState.SWEPT
            self._leases_by_id.pop(lease_id, None)
            return True

    def release_worker(self, worker_id: str) -> List[int]:
        """Free all LEASED blocks held by ``worker_id`` (clean disconnect)."""
        with self._lock:
            freed: List[int] = []
            for b in self._blocks:
                if (
                    b.state is LeaseState.LEASED
                    and b.lease is not None
                    and b.lease.worker_id == worker_id
                ):
                    self._leases_by_id.pop(b.lease.lease_id, None)
                    b.state = LeaseState.FREE
                    b.lease = None
                    freed.append(b.index)
            return freed
