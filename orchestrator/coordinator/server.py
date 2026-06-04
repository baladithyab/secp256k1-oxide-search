"""Threaded TCP coordinator: leases blocks, intakes heartbeats, halts on FOUND.

Architecture (README "COORDINATOR"):

* owns a global keyspace via :class:`RangeRegistry` (leases DISJOINT blocks,
  crash-tolerant lease TTL),
* a DP collision store stub (:class:`DPStore`) for kangaroo mode,
* a once-per-epoch :class:`Allocator` (bandit/optimal-stopping),
* an on-FOUND latch that records the winning key and STOPS handing out leases.

Transport is msgpack over length-prefixed TCP (:mod:`orchestrator.proto`). One
acceptor thread spawns one handler thread per worker connection; a background
reaper periodically reclaims expired leases so a dead worker's block returns to
the free pool even if no other worker is currently asking for work.

Threaded (not asyncio) by choice: the protocol is request/response per worker
with tiny messages, and the registry is already lock-guarded — threads keep the
handler code linear and easy to read.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import proto
from .allocator import Allocator
from .dp_store import DPStore
from .registry import RangeRegistry


@dataclass
class FoundResult:
    key: int
    worker_id: str
    lease_id: int
    block_lo: int
    block_hi: int


@dataclass
class WorkerInfo:
    worker_id: str
    gpu: int
    last_heartbeat: float = 0.0
    keys_s: float = 0.0
    block_progress: float = 0.0
    keys_done: int = 0  # cumulative keys this worker has reported sweeping


class Coordinator:
    """Threaded msgpack-over-TCP coordinator for brute-force key search.

    Parameters
    ----------
    lo, hi, block_size, lease_ttl:
        Forwarded to :class:`RangeRegistry`.
    target:
        The planted target digest (32 bytes) workers compare against; sent to
        each worker in the ``Welcome`` so the stub kernel knows what to match.
    host, port:
        Bind address. ``port=0`` picks an ephemeral port (read back from
        :attr:`port` after :meth:`start`).
    reaper_interval:
        Seconds between background expired-lease sweeps.
    """

    def __init__(
        self,
        lo: int,
        hi: int,
        block_size: int,
        target: bytes,
        lease_ttl: float = 30.0,
        host: str = "127.0.0.1",
        port: int = 0,
        reaper_interval: float = 0.2,
    ):
        self.registry = RangeRegistry(lo, hi, block_size, lease_ttl=lease_ttl)
        self.dp_store = DPStore(interval_lo=lo, interval_hi=hi)
        self.allocator = Allocator()
        self.target = target
        self.host = host
        self._requested_port = port
        self.reaper_interval = reaper_interval

        self._sock: Optional[socket.socket] = None
        self.port: Optional[int] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._reaper_thread: Optional[threading.Thread] = None
        self._handlers: List[threading.Thread] = []
        self._stop = threading.Event()

        self._state_lock = threading.RLock()
        self.workers: Dict[str, WorkerInfo] = {}
        self.found: Optional[FoundResult] = None
        self._conns: List[socket.socket] = []
        # Coverage log: block_index -> worker_id that swept/found it. Each block
        # is leased to at most one worker at a time (registry invariant), so a
        # double entry here would signal an overlap bug.
        self.coverage: Dict[int, str] = {}
        self._lease_owner: Dict[int, str] = {}  # lease_id -> worker_id
        self.coverage_conflict = None  # set if a block is ever swept twice

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> int:
        """Bind, start accepting, and start the reaper. Returns the bound port."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self._requested_port))
        self._sock.listen(64)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="coord-accept", daemon=True
        )
        self._accept_thread.start()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, name="coord-reaper", daemon=True
        )
        self._reaper_thread.start()
        return self.port

    def stop(self) -> None:
        """Stop all threads and close all sockets (clean teardown)."""
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        with self._state_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                c.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2.0)
        for h in list(self._handlers):
            h.join(timeout=2.0)

    def __enter__(self) -> "Coordinator":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- background loops --------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(1.0)
            with self._state_lock:
                self._conns.append(conn)
            h = threading.Thread(
                target=self._handle_conn, args=(conn,), name="coord-handler", daemon=True
            )
            h.start()
            self._handlers.append(h)

    def _reaper_loop(self) -> None:
        while not self._stop.wait(self.reaper_interval):
            # Reclaiming expired leases here means a crashed worker's block goes
            # back to FREE even when nobody is actively requesting work.
            self.registry.reclaim_expired()

    # -- per-connection handler -------------------------------------------

    def _handle_conn(self, conn: socket.socket) -> None:
        worker_id: Optional[str] = None
        try:
            while not self._stop.is_set():
                try:
                    msg = proto.read_msg(conn)
                except socket.timeout:
                    continue
                except (ConnectionError, OSError, proto.ProtocolError):
                    break
                if msg is None:
                    break  # clean EOF
                t = msg["t"]
                if t == proto.HELLO:
                    worker_id = msg["worker_id"]
                    self._on_hello(conn, worker_id, int(msg.get("gpu", 0)))
                elif t == proto.WANT_LEASE:
                    self._on_want_lease(conn, msg["worker_id"])
                elif t == proto.LEASE_ACK:
                    self.registry.ack_lease(int(msg["lease_id"]), msg["worker_id"])
                elif t == proto.HEARTBEAT:
                    self._on_heartbeat(msg)
                elif t == proto.SWEPT:
                    self._on_swept(msg)
                elif t == proto.FOUND:
                    self._on_found(msg)
                elif t == proto.DP_BATCH:
                    self._on_dp_batch(msg)
                elif t == proto.GOODBYE:
                    break
                # REALLOC_HINT / others are coordinator->worker only; ignore.
        finally:
            if worker_id is not None:
                # Release this worker's leased blocks so they return to FREE.
                self.registry.release_worker(worker_id)
            with self._state_lock:
                if conn in self._conns:
                    self._conns.remove(conn)
            try:
                conn.close()
            except OSError:
                pass

    # -- message handlers --------------------------------------------------

    def _on_hello(self, conn: socket.socket, worker_id: str, gpu: int) -> None:
        with self._state_lock:
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id, gpu=gpu, last_heartbeat=time.monotonic()
            )
        proto.send_msg(
            conn,
            proto.WELCOME,
            worker_id=worker_id,
            target=self.target,
            block_size=self.registry.block_size,
        )

    def _on_want_lease(self, conn: socket.socket, worker_id: str) -> None:
        # Halt: once FOUND, stop handing out leases (README FOUND handling).
        with self._state_lock:
            halted = self.found is not None
        if halted:
            proto.send_msg(conn, proto.NO_WORK, reason="found")
            return
        lease = self.registry.lease_block(worker_id)
        if lease is None:
            reason = "exhausted" if self.registry.is_exhausted() else "no_free_block"
            proto.send_msg(conn, proto.NO_WORK, reason=reason)
            return
        with self._state_lock:
            self._lease_owner[lease.lease_id] = worker_id
        proto.send_msg(
            conn,
            proto.LEASE,
            lease_id=lease.lease_id,
            lo=lease.lo,
            hi=lease.hi,
            block_index=lease.block_index,
        )

    def _on_heartbeat(self, msg: dict) -> None:
        worker_id = msg["worker_id"]
        lease_id = msg.get("lease_id")
        if lease_id is not None:
            self.registry.renew_lease(int(lease_id), worker_id)
        with self._state_lock:
            info = self.workers.get(worker_id)
            if info is not None:
                info.last_heartbeat = time.monotonic()
                info.keys_s = float(msg.get("keys_s", 0.0))
                info.block_progress = float(msg.get("block_progress", 0.0))

    def _on_swept(self, msg: dict) -> None:
        worker_id = msg["worker_id"]
        lease_id = int(msg["lease_id"])
        block_index = int(msg.get("block_index", -1))
        keys_scanned = int(msg.get("keys_scanned", 0))
        self.registry.mark_swept(lease_id, worker_id)
        self._record_coverage(block_index, worker_id, keys_scanned)

    def _record_coverage(self, block_index: int, worker_id: str, keys: int) -> None:
        with self._state_lock:
            if block_index >= 0:
                if block_index in self.coverage:
                    # Two workers swept the same block -> overlap bug. Flag it
                    # by raising; surfaces in the handler thread and fails tests
                    # that inspect coverage integrity.
                    self.coverage_conflict = (block_index, self.coverage[block_index], worker_id)
                self.coverage[block_index] = worker_id
            info = self.workers.get(worker_id)
            if info is not None:
                info.keys_done += keys

    def _on_found(self, msg: dict) -> None:
        worker_id = msg["worker_id"]
        key = int(msg["key"])
        lease_id = int(msg.get("lease_id", 0))
        lo = int(msg.get("lo", 0))
        hi = int(msg.get("hi", 0))
        block_index = int(msg.get("block_index", -1))
        keys_scanned = int(msg.get("keys_scanned", 0))
        with self._state_lock:
            if self.found is None:  # first FOUND wins; ignore duplicates
                self.found = FoundResult(
                    key=key,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    block_lo=lo,
                    block_hi=hi,
                )
        self._record_coverage(block_index, worker_id, keys_scanned)
        # Halt the allocator: stop pointing budget at any arm.
        self.allocator.halt()

    def _on_dp_batch(self, msg: dict) -> None:
        worker_id = msg["worker_id"]
        # points come over the wire as [[x, kind, dist], ...]
        points = [(int(x), str(kind), int(dist)) for x, kind, dist in msg["points"]]
        self.dp_store.add_batch(worker_id, points)

    # -- introspection -----------------------------------------------------

    def total_keys_done(self) -> int:
        """Sum of keys swept across all completed blocks + the found block."""
        with self._state_lock:
            return sum(w.keys_done for w in self.workers.values())

    def is_found(self) -> bool:
        with self._state_lock:
            return self.found is not None
