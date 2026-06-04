# Wave Plan

Honest, incremental. Track A is load-bearing — if cuda-oxide can't express secp256k1 point math
cleanly, the rest is moot, so it goes first and serial.

## Wave 1 — Track A bring-up (serial)
- [ ] `kernels/cuda-ref/`: minimal correct secp256k1 in CUDA C — field arithmetic (256-bit mod p),
      point add/double (Jacobian), scalar mult (double-and-add), hash160 (sha256+ripemd160),
      compare vs target rmd160. Correctness vs Python `coincurve`/`ecdsa` reference, bit-exact.
- [ ] `kernels/oxide-ec/`: same algorithm in cuda-oxide Rust. Document every API friction
      (expect: 256-bit limb arithmetic without inline asm is the crux).
- [ ] Bench keys/s both, same idle GPU same session. SASS instruction-mix (IMAD/IADD3 count).
- [ ] `docs/research/wave1-trackA.md` + `results/trackA-keys-per-sec.csv`.

**Gate:** if oxide can express correct point math (even slowly), proceed. The keys/s gap IS the
headline result either way.

## Wave 2 — Track B orchestration (after A correctness)
- [ ] `orchestrator/coordinator/`: range registry, lease TTL, DP store, msgpack TCP server.
- [ ] `orchestrator/worker/`: wraps the Track A kernel, lease loop, heartbeat, FOUND report.
- [ ] Multi-process single-box test (simulate N GPUs via N workers on time-sliced GPU or CPU stub).
- [ ] `allocator.py`: brute-force ranking + kangaroo Gittins skeleton (per bandit doc).
- [ ] Linear-scaling validation: keys/s vs worker count.

## Wave 3 — Track C INT8 spike (after A baseline modmul exists)
- [ ] Reuse Track A scalar modmul as baseline.
- [ ] CUDA C INT8 tensor-core modmul (`mma.sync .s8.s8.s32`), correctness + perf vs baseline.
- [ ] Attempt oxide IMMA path; likely negative result → document.
- [ ] `docs/research/wave3-trackC.md`.

## Wave 4 — Kangaroo mode (optional, if a #135-class target is pursued)
- [ ] Kangaroo-jump kernel (tame/wild herds, DP detection on-GPU).
- [ ] Coordinator DP-collision path end-to-end on a *small* solvable interval (e.g. 2^40 toy) to
      prove correctness before any real target.
- [ ] Validate on a known-answer small puzzle, never broadcast.

## Discipline (from rust-gpu-compute skill)
- Re-bench baselines in the SAME session on the SAME idle GPU; cite `nvidia-smi` thermal context.
- `/usr/local/cuda/bin/{nvcc,cuobjdump}` absolute paths (apt shims are stale CUDA 12.0 on sm_120).
- `export CUDA_HOME=/usr/local/cuda; export LIBNVVM_PATH=...; export PATH=/usr/lib/llvm-21/bin:$PATH`
  before any `cargo oxide` command (libNVVM shadow bug).
- Bit-exact correctness for all crypto — NO tolerance, one wrong bit = wrong key.
