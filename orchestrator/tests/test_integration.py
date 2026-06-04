"""(e) Localhost swarm: 1 coordinator + 3 worker SUBPROCESSES find a planted key.

Real sockets on 127.0.0.1 + an ephemeral port. The coordinator runs in-process
(it's just a threaded TCP server); the three workers are genuine subprocesses
(``python -m orchestrator.worker``) so this exercises process isolation and the
on-the-wire protocol end to end. Deterministic and fast (<~20s): the toy
keyspace is small and the stub kernel is a tight Python loop.

Asserts:
  * the swarm collectively reports FOUND with the CORRECT planted key,
  * coverage is DISJOINT (no block swept by two workers — registry invariant),
  * teardown is clean (all subprocesses exit; coordinator threads stop).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from orchestrator.coordinator.server import Coordinator
from orchestrator.worker.stub_kernel import target_digest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _wait_until(pred, timeout=18.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _spawn_worker(worker_id, port):
    env = dict(os.environ)
    # Ensure the subprocess can import the orchestrator package.
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "orchestrator.worker",
            "--coordinator",
            f"127.0.0.1:{port}",
            "--worker-id",
            worker_id,
            "--gpu",
            "0",
            "--heartbeat-every",
            "512",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_three_worker_swarm_finds_planted_key():
    # Toy keyspace [0, 1200) split into 50-key blocks (24 blocks). Plant a key
    # near the end so multiple blocks get swept before the hit.
    lo, hi, block_size = 0, 1200, 50
    planted = 1021
    assert lo <= planted < hi
    target = target_digest(planted)

    coord = Coordinator(
        lo=lo,
        hi=hi,
        block_size=block_size,
        target=target,
        lease_ttl=10.0,
        reaper_interval=0.1,
    )
    coord.start()
    procs = []
    try:
        port = coord.port
        procs = [_spawn_worker(f"w{i}", port) for i in range(3)]

        # The swarm should collectively report FOUND.
        assert _wait_until(coord.is_found, timeout=18.0), (
            "coordinator never recorded FOUND"
        )
        assert coord.found is not None
        assert coord.found.key == planted, coord.found
        assert coord.found.block_lo <= planted < coord.found.block_hi

        # Wait for all workers to exit on their own (they stop on NoWork(found)).
        for p in procs:
            try:
                p.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5.0)
            assert p.returncode == 0, (
                f"worker exited {p.returncode}; stderr={p.stderr.read()}"
            )

        # Exactly one worker should have printed the found key on stdout.
        finders = []
        for i, p in enumerate(procs):
            out = p.stdout.read()
            if "FOUND" in out:
                finders.append((f"w{i}", out.strip()))
        assert len(finders) == 1, finders
        assert finders[0][1] == f"FOUND {planted}"

        # Coverage disjointness: no block swept/found by two workers, and no
        # conflict was ever flagged by the coordinator.
        assert coord.coverage_conflict is None, coord.coverage_conflict
        # Every covered block index is unique (dict keys are unique by
        # construction; the conflict flag would have fired otherwise).
        assert len(coord.coverage) >= 1
        # The found block must appear in coverage attributed to the finder.
        found_block_owner = coord.coverage.get(_found_block_index(coord))
        assert found_block_owner == coord.found.worker_id
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                try:
                    p.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        coord.stop()


def _found_block_index(coord):
    # Recompute the block index for the found block from its bounds.
    return (coord.found.block_lo - coord.registry.lo) // coord.registry.block_size


def test_swarm_exhausts_keyspace_when_key_absent():
    """If the planted key is NOT in the keyspace, the swarm sweeps it all and
    every worker exits cleanly (no FOUND, no hang)."""
    lo, hi, block_size = 0, 600, 50
    target = target_digest(10**9)  # absent from [0, 600)
    coord = Coordinator(
        lo=lo, hi=hi, block_size=block_size, target=target,
        lease_ttl=10.0, reaper_interval=0.1,
    )
    coord.start()
    procs = []
    try:
        port = coord.port
        procs = [_spawn_worker(f"x{i}", port) for i in range(3)]
        for p in procs:
            try:
                p.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5.0)
                pytest.fail("worker hung instead of exhausting keyspace")
            assert p.returncode == 0
        assert coord.found is None
        # Whole keyspace swept (12 blocks), disjointly.
        assert coord.registry.is_exhausted()
        assert coord.coverage_conflict is None
        assert len(coord.coverage) == coord.registry.n_blocks
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        coord.stop()
