"""Run a worker as a subprocess::

    python -m orchestrator.worker --coordinator 127.0.0.1:5555 --worker-id w0 --gpu 0

The integration test launches three of these against one coordinator. Exit code
0 = clean stop (swept out / peer found / exhausted); the found key (if this
worker is the finder) is printed as ``FOUND <key>`` on stdout.
"""

from __future__ import annotations

import argparse
import sys

from .worker import run_worker_process


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="orchestrator.worker")
    p.add_argument(
        "--coordinator", required=True, help="coordinator host:port (e.g. 127.0.0.1:5555)"
    )
    p.add_argument("--worker-id", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--heartbeat-every", type=int, default=2048)
    p.add_argument("--max-blocks", type=int, default=None)
    args = p.parse_args(argv)

    host, _, port_s = args.coordinator.rpartition(":")
    port = int(port_s)

    key = run_worker_process(
        host,
        port,
        args.worker_id,
        gpu=args.gpu,
        heartbeat_every=args.heartbeat_every,
        max_blocks=args.max_blocks,
    )
    if key is not None:
        print(f"FOUND {key}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
