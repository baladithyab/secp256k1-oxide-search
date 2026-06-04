# N-GPU Orchestration Architecture

The auto-scale-to-N-GPUs design. The search is **embarrassingly parallel**, so this is mostly
careful bookkeeping, not hard distributed systems.

```
                       ┌─────────────────────────────┐
                       │        COORDINATOR          │
                       │  - owns global keyspace      │
                       │  - range registry (claimed/  │
                       │    swept/free)               │
                       │  - distinguished-point (DP)  │
                       │    collision store           │
                       │  - bandit allocator (epoch)  │
                       │  - on-chain hazard monitor   │
                       └──────────┬──────────────────┘
                                  │  gRPC / msgpack-over-TCP
              ┌───────────────────┼───────────────────┐
              │                   │                   │
        ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
        │  WORKER 0 │       │  WORKER 1 │  ...  │ WORKER N-1│
        │  GPU 0    │       │  GPU 1    │       │  GPU N-1  │
        │ oxide-ec  │       │ oxide-ec  │       │ oxide-ec  │
        │  kernel   │       │  kernel   │       │  kernel   │
        └───────────┘       └───────────┘       └───────────┘
```

## Two search modes, two partition strategies

### Mode 1 — Brute force (no pubkey, e.g. #71)
- Coordinator partitions `[2^(N-1), 2^N)` into **fixed-size blocks** (e.g. 2⁴⁰ keys each).
- Worker `lease`s a block, sweeps it on-GPU (scalar mult → hash160 → compare against target
  rmd160), reports `swept` or `FOUND`.
- **Zero hot-path communication.** Only lease/ack/heartbeat. Scales linearly to N with no
  contention until the (astronomically large) range is exhausted.
- Crash recovery: unacked leases expire and return to the free pool (lease TTL).

### Mode 2 — Pollard kangaroo (exposed pubkey, e.g. #135)
- All workers run kangaroo herds in the **same** interval (they must — it's one DLP instance).
- Each worker emits **distinguished points** (DPs: points whose x-coord has `d` trailing zero
  bits) to the coordinator.
- Coordinator stores DPs in a hash map keyed by x-coord; a **collision** (same DP reached by a
  tame and a wild kangaroo) yields the private key.
- Communication = DP upload rate, tunable via `d` (higher `d` → fewer uploads, more on-GPU work
  per DP). This is the only mode with real coordinator bandwidth considerations; sizing `d` keeps
  it trivial (kB/s even at cluster scale).

## Heterogeneous + elastic
- Workers self-report `keys/s` (or `jumps/s`); coordinator weights block size / herd size by
  measured throughput → mixed GPU generations are fine.
- Workers can join/leave mid-run; leases + DP store are the only shared state, both crash-tolerant.
- A worker is just `worker --coordinator <host:port> --gpu <id>`. `N` GPUs on one box = `N` worker
  procs; multi-box = same, over the network.

## Transport
- `msgpack` over a length-prefixed TCP stream (no heavy gRPC dep for v1; can swap later).
- Messages: `Lease`, `LeaseAck`, `Swept`, `Found`, `DPBatch`, `Heartbeat`, `ReallocHint`.

## Telemetry → bandit
- Each `Heartbeat` carries `keys_s` / `jumps_s`, `dp_count`, `block_progress`.
- Coordinator feeds these to the allocator (`docs/research/bandit-allocation.md`) once per epoch
  and may issue `ReallocHint` to steer idle capacity toward the current argmax arm.

## Explicitly out of scope for v1
- No transaction construction or broadcast (see Reality Check in root README).
- No NAT traversal / auth hardening (assume trusted LAN or VPN for a research cluster).
