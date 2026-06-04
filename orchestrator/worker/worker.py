"""Worker client: connect, lease, sweep via the (stub) kernel, heartbeat, FOUND.

The production worker invokes the GPU kernel binary via subprocess. For tests we
drive the pure-Python :func:`orchestrator.worker.stub_kernel.sweep_block`
directly (selectable via ``kernel="stub"``), so no GPU is needed and the run is
deterministic and fast.

Protocol flow (one worker):

    -> Hello(worker_id, gpu)        <- Welcome(target, block_size)
    loop:
      -> WantLease                  <- Lease(lease_id, lo, hi) | NoWork(reason)
      -> LeaseAck(lease_id)
      sweep [lo, hi):
        periodically -> Heartbeat(lease_id, keys_s, block_progress)
      on no hit:   -> Swept(lease_id)
      on hit:      -> Found(lease_id, key, lo, hi); stop
    on NoWork(found|exhausted): stop

Heartbeats are emitted from the sweep's progress callback (in-line), which keeps
the single socket single-writer and avoids a heartbeat-thread race on the
connection. For the stub kernel that is plenty frequent; a real subprocess
worker would run a heartbeat thread reading the kernel's stdout progress.
"""

from __future__ import annotations

import socket
import time
from typing import Optional

from .. import proto
from .stub_kernel import sweep_block


class Worker:
    """A single worker process/thread driving the stub kernel."""

    def __init__(
        self,
        host: str,
        port: int,
        worker_id: str,
        gpu: int = 0,
        heartbeat_every: int = 2048,
        connect_timeout: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.worker_id = worker_id
        self.gpu = gpu
        self.heartbeat_every = heartbeat_every
        self.connect_timeout = connect_timeout

        self.sock: Optional[socket.socket] = None
        self.target: Optional[bytes] = None
        self.found_key: Optional[int] = None
        self.blocks_swept: int = 0
        self.keys_scanned: int = 0

    # -- connection --------------------------------------------------------

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout
        )
        self.sock.settimeout(self.connect_timeout)
        proto.send_msg(self.sock, proto.HELLO, worker_id=self.worker_id, gpu=self.gpu)
        welcome = proto.read_msg(self.sock)
        if welcome is None or welcome["t"] != proto.WELCOME:
            raise ConnectionError("did not receive Welcome")
        self.target = welcome["target"]

    def close(self) -> None:
        if self.sock is not None:
            try:
                proto.send_msg(self.sock, proto.GOODBYE, worker_id=self.worker_id)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # -- main loop ---------------------------------------------------------

    def run(self, max_blocks: Optional[int] = None) -> Optional[int]:
        """Lease/sweep loop. Returns the found key, or None if it stops first.

        Stops on: FOUND (by self or, via NoWork(found), by a peer), keyspace
        exhausted, ``max_blocks`` blocks processed, or socket error.
        """
        assert self.sock is not None, "call connect() first"
        processed = 0
        while True:
            if max_blocks is not None and processed >= max_blocks:
                break
            proto.send_msg(self.sock, proto.WANT_LEASE, worker_id=self.worker_id)
            reply = proto.read_msg(self.sock)
            if reply is None:
                break
            if reply["t"] == proto.NO_WORK:
                # Either a peer found it, or the keyspace is exhausted -> stop.
                break
            if reply["t"] != proto.LEASE:
                raise ConnectionError(f"unexpected reply: {reply['t']}")

            lease_id = int(reply["lease_id"])
            lo = int(reply["lo"])
            hi = int(reply["hi"])
            block_index = int(reply.get("block_index", -1))
            proto.send_msg(
                self.sock, proto.LEASE_ACK, lease_id=lease_id, worker_id=self.worker_id
            )

            result = self._sweep_with_heartbeats(lease_id, lo, hi)
            self.keys_scanned += result.keys_scanned

            if result.found:
                self.found_key = result.key
                proto.send_msg(
                    self.sock,
                    proto.FOUND,
                    worker_id=self.worker_id,
                    lease_id=lease_id,
                    key=result.key,
                    lo=lo,
                    hi=hi,
                    block_index=block_index,
                    keys_scanned=result.keys_scanned,
                )
                return result.key

            proto.send_msg(
                self.sock,
                proto.SWEPT,
                lease_id=lease_id,
                worker_id=self.worker_id,
                block_index=block_index,
                keys_scanned=result.keys_scanned,
            )
            self.blocks_swept += 1
            processed += 1
        return None

    def _sweep_with_heartbeats(self, lease_id: int, lo: int, hi: int):
        span = max(hi - lo, 1)
        t0 = time.monotonic()

        def progress_cb(scanned: int, _current_key: int) -> None:
            elapsed = max(time.monotonic() - t0, 1e-9)
            keys_s = scanned / elapsed
            block_progress = scanned / span
            try:
                proto.send_msg(
                    self.sock,
                    proto.HEARTBEAT,
                    worker_id=self.worker_id,
                    lease_id=lease_id,
                    keys_s=keys_s,
                    block_progress=block_progress,
                )
            except OSError:
                pass  # coordinator gone; sweep will end and loop will exit

        return sweep_block(
            lo, hi, self.target, progress_cb=progress_cb, progress_every=self.heartbeat_every
        )


def run_worker_process(
    host: str,
    port: int,
    worker_id: str,
    gpu: int = 0,
    heartbeat_every: int = 2048,
    max_blocks: Optional[int] = None,
) -> Optional[int]:
    """Entry point for a worker run in a subprocess (see ``__main__``)."""
    w = Worker(host, port, worker_id, gpu=gpu, heartbeat_every=heartbeat_every)
    try:
        w.connect()
        return w.run(max_blocks=max_blocks)
    finally:
        w.close()
