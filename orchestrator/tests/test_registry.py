"""Registry tests: lease disjointness (a) and lease-TTL reclaim (b)."""

from __future__ import annotations

import threading

import pytest

from orchestrator.coordinator.registry import LeaseState, RangeRegistry


class FakeClock:
    """Manually-advanced monotonic clock for deterministic TTL tests."""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# -- (a) lease disjointness -------------------------------------------------


def test_blocks_partition_keyspace_exactly():
    reg = RangeRegistry(lo=0, hi=100, block_size=30)
    # 0..30, 30..60, 60..90, 90..100 -> 4 blocks, last truncated.
    assert reg.n_blocks == 4
    leases = []
    while True:
        ls = reg.lease_block("w")
        if ls is None:
            break
        leases.append(ls)
    bounds = sorted((l.lo, l.hi) for l in leases)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 100
    # Contiguous, non-overlapping cover of [0, 100).
    for (lo_a, hi_a), (lo_b, hi_b) in zip(bounds, bounds[1:]):
        assert hi_a == lo_b


def test_concurrent_leases_are_disjoint():
    reg = RangeRegistry(lo=1000, hi=1000 + 16 * 64, block_size=64)
    out = []
    out_lock = threading.Lock()

    def grab(worker_id):
        ls = reg.lease_block(worker_id)
        if ls is not None:
            with out_lock:
                out.append(ls)

    threads = [
        threading.Thread(target=grab, args=(f"w{i}",)) for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(out) == 16
    # Every pair of concurrently-held leases must be disjoint.
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert not out[i].overlaps(out[j]), (
                f"leases overlap: {out[i].as_dict()} vs {out[j].as_dict()}"
            )
    # Lease ids unique.
    assert len({l.lease_id for l in out}) == len(out)


def test_active_leases_never_overlap_over_churn():
    # Lease some, sweep some, re-lease; the *active* set is always disjoint.
    reg = RangeRegistry(lo=0, hi=8 * 10, block_size=10)
    a = reg.lease_block("w0")
    b = reg.lease_block("w1")
    c = reg.lease_block("w2")
    active = reg.active_leases()
    assert len(active) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert not active[i].overlaps(active[j])
    # Sweep b, lease another; still disjoint.
    assert reg.mark_swept(b.lease_id, "w1")
    d = reg.lease_block("w3")
    active = reg.active_leases()
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            assert not active[i].overlaps(active[j])


# -- (b) lease-TTL reclaim --------------------------------------------------


def test_expired_lease_is_reclaimed_and_releasable():
    clk = FakeClock(0.0)
    reg = RangeRegistry(lo=0, hi=100, block_size=100, lease_ttl=10.0, clock=clk)
    # One block total. Worker w0 leases it but "dies" (never acks/heartbeats).
    ls = reg.lease_block("w0")
    assert ls is not None
    assert reg.lease_block("w1") is None  # nothing free while w0 holds it

    # Before TTL: still owned, not reclaimable.
    clk.advance(9.0)
    assert reg.reclaim_expired() == []
    assert reg.lease_block("w1") is None

    # After TTL: reclaimed to FREE, and a different worker can take it.
    clk.advance(2.0)  # t=11 > 10
    reclaimed = reg.reclaim_expired()
    assert reclaimed == [ls.block_index]
    states = reg.snapshot_states()
    assert states == [LeaseState.FREE]
    ls2 = reg.lease_block("w1")
    assert ls2 is not None
    assert ls2.worker_id == "w1"
    assert ls2.lo == ls.lo and ls2.hi == ls.hi  # same block, re-leased


def test_heartbeat_renews_ttl_so_live_worker_keeps_block():
    clk = FakeClock(0.0)
    reg = RangeRegistry(lo=0, hi=100, block_size=100, lease_ttl=10.0, clock=clk)
    ls = reg.lease_block("w0")
    assert reg.ack_lease(ls.lease_id, "w0")

    # Heartbeat just before each expiry keeps it alive indefinitely.
    for _ in range(5):
        clk.advance(8.0)
        assert reg.renew_lease(ls.lease_id, "w0") is True
        assert reg.reclaim_expired() == []
    # Still held by w0, nothing free.
    assert reg.lease_block("w1") is None

    # Stop heartbeating -> expires -> reclaimed.
    clk.advance(11.0)
    assert reg.reclaim_expired() == [ls.block_index]


def test_stale_worker_cannot_renew_after_reclaim():
    clk = FakeClock(0.0)
    reg = RangeRegistry(lo=0, hi=50, block_size=50, lease_ttl=5.0, clock=clk)
    ls = reg.lease_block("w0")
    clk.advance(6.0)
    # A heartbeat from the dead worker after expiry must fail (block already
    # reclaimable); renew triggers reclaim and returns False.
    assert reg.renew_lease(ls.lease_id, "w0") is False
    # And it's now free for someone else.
    ls2 = reg.lease_block("w1")
    assert ls2 is not None and ls2.worker_id == "w1"


def test_release_worker_frees_its_blocks():
    reg = RangeRegistry(lo=0, hi=30, block_size=10)
    a = reg.lease_block("w0")
    b = reg.lease_block("w0")
    c = reg.lease_block("w1")
    freed = reg.release_worker("w0")
    assert sorted(freed) == sorted([a.block_index, b.block_index])
    # w1's block stays leased.
    active = reg.active_leases()
    assert len(active) == 1 and active[0].worker_id == "w1"
