"""Wire protocol: msgpack over length-prefixed TCP.

Frame format
------------
Each message on the wire is::

    [4-byte big-endian unsigned length N][N bytes of msgpack payload]

The payload is a msgpack map with a mandatory ``"t"`` key carrying the message
type tag (see :data:`MSG_TYPES`). The remaining keys are the message fields.

Message types (per ``orchestrator/README.md`` -> "Transport")::

    Lease, LeaseAck, Swept, Found, DPBatch, Heartbeat, ReallocHint

We keep the codec dependency-light: msgpack only, no gRPC. The functions here
are transport primitives shared by both the coordinator and the worker. They
are intentionally synchronous and blocking; the coordinator runs one thread per
connection and the worker is single-threaded with a background heartbeat
thread, so blocking reads are fine.
"""

from __future__ import annotations

import socket
import struct
from typing import Any, Dict, Optional

import msgpack

# ---------------------------------------------------------------------------
# Message type tags
# ---------------------------------------------------------------------------

# Worker -> Coordinator and Coordinator -> Worker tags. Kept as short strings so
# the msgpack frames stay small; the set is closed and validated on decode.
LEASE = "Lease"  # coordinator -> worker: here is a block [lo, hi)
LEASE_ACK = "LeaseAck"  # worker -> coordinator: I accept lease <id>
SWEPT = "Swept"  # worker -> coordinator: block <id> fully swept, no hit
FOUND = "Found"  # worker -> coordinator: hit! private key + block id
DP_BATCH = "DPBatch"  # worker -> coordinator: distinguished points (kangaroo)
HEARTBEAT = "Heartbeat"  # worker -> coordinator: liveness + telemetry
REALLOC_HINT = "ReallocHint"  # coordinator -> worker: steer toward argmax arm

# Control / handshake tags (not in the README's solving set but needed for a
# real socket protocol: a worker has to announce itself and ask for work).
HELLO = "Hello"  # worker -> coordinator: register, announce worker_id/gpu
WELCOME = "Welcome"  # coordinator -> worker: registration accepted
WANT_LEASE = "WantLease"  # worker -> coordinator: request a block
NO_WORK = "NoWork"  # coordinator -> worker: nothing to hand out (idle/halted)
GOODBYE = "Goodbye"  # worker -> coordinator: clean shutdown

MSG_TYPES = frozenset(
    {
        LEASE,
        LEASE_ACK,
        SWEPT,
        FOUND,
        DP_BATCH,
        HEARTBEAT,
        REALLOC_HINT,
        HELLO,
        WELCOME,
        WANT_LEASE,
        NO_WORK,
        GOODBYE,
    }
)

_LEN = struct.Struct(">I")  # 4-byte big-endian unsigned length prefix
MAX_FRAME = 64 * 1024 * 1024  # 64 MiB hard cap, guards against bad length prefix


class ProtocolError(Exception):
    """Raised on a malformed frame, unknown message type, or oversized frame."""


def encode(msg_type: str, **fields: Any) -> bytes:
    """Encode a message into a length-prefixed msgpack frame.

    Parameters
    ----------
    msg_type:
        One of :data:`MSG_TYPES`.
    **fields:
        Arbitrary msgpack-serializable fields. ``"t"`` is reserved.
    """
    if msg_type not in MSG_TYPES:
        raise ProtocolError(f"unknown message type: {msg_type!r}")
    if "t" in fields:
        raise ProtocolError("field name 't' is reserved for the type tag")
    body: Dict[str, Any] = {"t": msg_type}
    body.update(fields)
    payload = msgpack.packb(body, use_bin_type=True)
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"frame too large: {len(payload)} > {MAX_FRAME}")
    return _LEN.pack(len(payload)) + payload


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes from ``sock`` or return ``None`` on clean EOF.

    Raises :class:`ConnectionError` only on a *partial* read (peer vanished
    mid-frame), which the caller treats the same as a dropped worker.
    """
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            if remaining == n:
                # Nothing read at all -> clean EOF at a frame boundary.
                return None
            raise ConnectionError("peer closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """Read one framed message from ``sock``.

    Returns the decoded dict (always containing ``"t"``), or ``None`` if the
    peer closed the connection cleanly at a frame boundary.
    """
    header = _recv_exact(sock, _LEN.size)
    if header is None:
        return None
    (length,) = _LEN.unpack(header)
    if length > MAX_FRAME:
        raise ProtocolError(f"declared frame too large: {length} > {MAX_FRAME}")
    payload = _recv_exact(sock, length)
    if payload is None:
        raise ConnectionError("peer closed before payload")
    obj = msgpack.unpackb(payload, raw=False)
    if not isinstance(obj, dict) or "t" not in obj:
        raise ProtocolError("frame is not a tagged map")
    if obj["t"] not in MSG_TYPES:
        raise ProtocolError(f"unknown message type on wire: {obj['t']!r}")
    return obj


def send_msg(sock: socket.socket, msg_type: str, **fields: Any) -> None:
    """Encode and send one message on ``sock`` (uses ``sendall``)."""
    sock.sendall(encode(msg_type, **fields))
