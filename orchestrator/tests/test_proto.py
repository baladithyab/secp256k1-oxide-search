"""Transport tests: length-prefixed msgpack frames round-trip; framing is exact."""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from orchestrator import proto


def _socketpair():
    a, b = socket.socketpair()
    return a, b


def test_encode_decode_roundtrip():
    a, b = _socketpair()
    try:
        proto.send_msg(a, proto.LEASE, lease_id=7, lo=10, hi=20, block_index=3)
        msg = proto.read_msg(b)
        assert msg == {
            "t": "Lease",
            "lease_id": 7,
            "lo": 10,
            "hi": 20,
            "block_index": 3,
        }
    finally:
        a.close()
        b.close()


def test_length_prefix_is_4_byte_big_endian():
    frame = proto.encode(proto.HEARTBEAT, worker_id="w0", keys_s=1.0)
    (declared,) = struct.unpack(">I", frame[:4])
    assert declared == len(frame) - 4


def test_multiple_frames_stream_without_bleed():
    a, b = _socketpair()
    try:
        proto.send_msg(a, proto.HELLO, worker_id="w0", gpu=0)
        proto.send_msg(a, proto.WANT_LEASE, worker_id="w0")
        proto.send_msg(a, proto.GOODBYE, worker_id="w0")
        m1 = proto.read_msg(b)
        m2 = proto.read_msg(b)
        m3 = proto.read_msg(b)
        assert m1["t"] == proto.HELLO
        assert m2["t"] == proto.WANT_LEASE
        assert m3["t"] == proto.GOODBYE
    finally:
        a.close()
        b.close()


def test_clean_eof_returns_none():
    a, b = _socketpair()
    try:
        a.close()
        assert proto.read_msg(b) is None
    finally:
        b.close()


def test_unknown_type_rejected_on_encode():
    with pytest.raises(proto.ProtocolError):
        proto.encode("NotAType", x=1)


def test_reserved_field_rejected():
    with pytest.raises(proto.ProtocolError):
        proto.encode(proto.HEARTBEAT, t="oops")


def test_oversized_declared_frame_rejected():
    a, b = _socketpair()
    try:
        # Hand-craft a header that declares a frame larger than MAX_FRAME.
        a.sendall(struct.pack(">I", proto.MAX_FRAME + 1))
        with pytest.raises(proto.ProtocolError):
            proto.read_msg(b)
    finally:
        a.close()
        b.close()
