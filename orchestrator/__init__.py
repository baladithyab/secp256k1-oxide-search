"""N-GPU orchestrator for secp256k1 key search (pure-Python, Wave 4).

Coordinator/worker design per ``orchestrator/README.md`` and
``docs/adr/0004-orchestration-stochastic-allocation.md``.

This package implements ONLY the *solving* orchestration: leasing disjoint
keyspace blocks, crash-tolerant lease TTLs, FOUND propagation, heartbeat
telemetry, a distinguished-point (DP) store stub for kangaroo mode, and a
once-per-epoch bandit allocator. It deliberately contains no transaction,
signing-for-spend, or broadcast path.
"""

__all__ = ["proto", "coordinator", "worker"]
