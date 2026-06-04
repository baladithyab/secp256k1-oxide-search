"""Wave-4 scaling benchmark: aggregate stub-kernel keys/s vs worker count.

Runs the in-process coordinator + N worker subprocesses over a fixed toy
keyspace with an ABSENT target (so the whole space is swept — same total work
regardless of N), times the wall-clock to exhaustion, and reports aggregate
keys/s = total_keys / wall_time. With share-nothing disjoint leasing the
aggregate throughput should scale roughly linearly in N until the (here, tiny)
keyspace is exhausted.

Writes ``results/wave4-scaling.csv`` with a header row.

This is a STUB-KERNEL CPU benchmark (no GPU); it demonstrates the orchestration
scaling property, not real secp256k1 keys/s. Honest scope, per ADR-0004.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time

from orchestrator.coordinator.server import Coordinator
from orchestrator.worker.stub_kernel import target_digest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _spawn_worker(worker_id: str, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [
            sys.executable, "-m", "orchestrator.worker",
            "--coordinator", f"127.0.0.1:{port}",
            "--worker-id", worker_id,
            "--heartbeat-every", "4096",
        ],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )


def run_one(n_workers: int, lo: int, hi: int, block_size: int) -> dict:
    """Sweep [lo, hi) (absent target) with n_workers; return timing + keys/s."""
    target = target_digest(hi + 10**6)  # guaranteed absent
    coord = Coordinator(
        lo=lo, hi=hi, block_size=block_size, target=target,
        lease_ttl=30.0, reaper_interval=0.2,
    )
    coord.start()
    procs = []
    try:
        t0 = time.monotonic()
        procs = [_spawn_worker(f"w{i}", coord.port) for i in range(n_workers)]
        for p in procs:
            p.wait(timeout=120.0)
        wall = time.monotonic() - t0
        assert coord.registry.is_exhausted(), "keyspace not fully swept"
        assert coord.coverage_conflict is None
        total_keys = hi - lo
        return {
            "workers": n_workers,
            "total_keys": total_keys,
            "wall_s": wall,
            "aggregate_keys_s": total_keys / wall,
        }
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        coord.stop()


def main() -> None:
    # Toy keyspace sized so even 1 worker finishes in a couple seconds.
    lo, hi, block_size = 0, 30000, 500
    rows = []
    baseline = None
    for n in (1, 2, 4):
        r = run_one(n, lo, hi, block_size)
        if baseline is None:
            baseline = r["aggregate_keys_s"]
        r["speedup_vs_1"] = r["aggregate_keys_s"] / baseline
        rows.append(r)
        print(
            f"workers={n} wall={r['wall_s']:.3f}s "
            f"agg_keys_s={r['aggregate_keys_s']:.0f} "
            f"speedup={r['speedup_vs_1']:.2f}x"
        )

    out_path = os.path.join(REPO_ROOT, "results", "wave4-scaling.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        # Context comment (CPU stub kernel, no GPU): this measures orchestration
        # scaling, not real secp256k1 keys/s. See ADR-0004 / orchestrator/README.
        w.writerow(
            [
                "# Wave-4 orchestrator scaling: pure-Python CPU stub kernel "
                "(NOT a GPU keys/s number); toy keyspace [%d,%d) block=%d; "
                "aggregate_keys_s = total_keys / wall-to-exhaustion"
                % (lo, hi, block_size)
            ]
        )
        w.writerow(
            ["workers", "total_keys", "wall_s", "aggregate_keys_s", "speedup_vs_1"]
        )
        for r in rows:
            w.writerow(
                [
                    r["workers"],
                    r["total_keys"],
                    f"{r['wall_s']:.4f}",
                    f"{r['aggregate_keys_s']:.1f}",
                    f"{r['speedup_vs_1']:.3f}",
                ]
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
