"""Coordinator tests over real sockets: lease handout disjointness + FOUND (c).

These drive the threaded :class:`Coordinator` on 127.0.0.1 with an ephemeral
port, using an in-process :class:`Worker` (no subprocess) for speed. The
full-subprocess swarm test lives in ``test_integration.py``.
"""

from __future__ import annotations

import socket
import time

import pytest

from orchestrator import proto
from orchestrator.coordinator.server import Coordinator
from orchestrator.worker.stub_kernel import target_digest
from orchestrator.worker.worker import Worker


def _wait_until(pred, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_lease_handout_disjoint_over_socket():
    # Two raw clients each grab a block; the coordinator must give disjoint ones.
    target = target_digest(10**9)  # not in range -> nobody finds it
    coord = Coordinator(lo=0, hi=400, block_size=100, target=target, lease_ttl=30.0)
    coord.start()
    try:
        port = coord.port
        leases = []
        socks = []
        for i in range(3):
            s = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            s.settimeout(5.0)
            socks.append(s)
            proto.send_msg(s, proto.HELLO, worker_id=f"w{i}", gpu=i)
            assert proto.read_msg(s)["t"] == proto.WELCOME
            proto.send_msg(s, proto.WANT_LEASE, worker_id=f"w{i}")
            reply = proto.read_msg(s)
            assert reply["t"] == proto.LEASE
            proto.send_msg(s, proto.LEASE_ACK, lease_id=reply["lease_id"], worker_id=f"w{i}")
            leases.append((reply["lo"], reply["hi"]))
        # Disjoint and distinct.
        leases.sort()
        for (lo_a, hi_a), (lo_b, hi_b) in zip(leases, leases[1:]):
            assert hi_a <= lo_b
        for s in socks:
            s.close()
    finally:
        coord.stop()


def test_found_is_recorded_and_halts_allocation():
    """(c) A worker reporting FOUND is recorded and the coordinator halts."""
    planted = 137
    target = target_digest(planted)
    coord = Coordinator(lo=100, hi=200, block_size=25, target=target, lease_ttl=30.0)
    coord.start()
    try:
        # Seed allocator arms so we can prove allocation halts after FOUND.
        from orchestrator.coordinator.allocator import BruteForceArm

        coord.allocator.add_brute_force(BruteForceArm(name="here", range_size=100))
        assert coord.allocator.allocate() == "here"  # allocating pre-FOUND

        w = Worker("127.0.0.1", coord.port, "finder", heartbeat_every=8)
        w.connect()
        found_key = w.run()
        assert found_key == planted
        w.close()

        assert _wait_until(coord.is_found)
        assert coord.found is not None
        assert coord.found.key == planted
        assert coord.found.worker_id == "finder"
        # Block bounds recorded and contain the key.
        assert coord.found.block_lo <= planted < coord.found.block_hi
        # Allocation is halted: argmax returns None now.
        assert coord.allocator.halted is True
        assert coord.allocator.allocate() is None
    finally:
        coord.stop()


def test_found_stops_new_lease_handout():
    """After FOUND, WantLease returns NoWork(found) — no more blocks given out."""
    planted = 305
    target = target_digest(planted)
    coord = Coordinator(lo=300, hi=320, block_size=5, target=target, lease_ttl=30.0)
    coord.start()
    try:
        w = Worker("127.0.0.1", coord.port, "finder", heartbeat_every=4)
        w.connect()
        assert w.run() == planted
        w.close()
        assert _wait_until(coord.is_found)

        # A late worker now gets NoWork(found).
        s = socket.create_connection(("127.0.0.1", coord.port), timeout=5.0)
        s.settimeout(5.0)
        proto.send_msg(s, proto.HELLO, worker_id="late", gpu=0)
        assert proto.read_msg(s)["t"] == proto.WELCOME
        proto.send_msg(s, proto.WANT_LEASE, worker_id="late")
        reply = proto.read_msg(s)
        assert reply["t"] == proto.NO_WORK
        assert reply["reason"] == "found"
        s.close()
    finally:
        coord.stop()


def test_dead_worker_block_reclaimed_over_socket():
    """(b) over the wire: a worker that leases then vanishes -> block reclaimed."""
    target = target_digest(10**9)  # absent
    coord = Coordinator(
        lo=0, hi=50, block_size=50, target=target, lease_ttl=0.4, reaper_interval=0.05
    )
    coord.start()
    try:
        # Lease the single block, then drop the socket WITHOUT sweeping/acking.
        s = socket.create_connection(("127.0.0.1", coord.port), timeout=5.0)
        s.settimeout(5.0)
        proto.send_msg(s, proto.HELLO, worker_id="zombie", gpu=0)
        assert proto.read_msg(s)["t"] == proto.WELCOME
        proto.send_msg(s, proto.WANT_LEASE, worker_id="zombie")
        assert proto.read_msg(s)["t"] == proto.LEASE
        # Hard-drop the connection (simulate crash); also let TTL expire.
        s.close()

        # The reaper (or release_worker on disconnect) frees the block; a new
        # worker can then lease the same block.
        def fresh_can_lease():
            ls = coord.registry.lease_block("survivor")
            if ls is None:
                return False
            return True

        assert _wait_until(fresh_can_lease, timeout=5.0)
    finally:
        coord.stop()
