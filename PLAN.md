# PLAN.md — Executable Backlog for ultracode

Repo: `baladithyab/secp256k1-oxide-search` | base branch: `master` | base SHA: see `git rev-parse HEAD`

This is the implementer contract. Each wave has **testable acceptance**. Build in order. The
correctness oracle already exists and passes — your kernels must match it bit-exact.

## READ FIRST, IN ORDER
1. `README.md` — scope, the three tracks, the nvfp4 refutation
2. `docs/adr/0001-scope-economics-nongoals.md` — negative-EV baseline, non-goals (NO broadcast)
3. `docs/adr/0003-alternative-gpu-methods.md` — which methods are real; what NOT to build
4. `docs/adr/0004-orchestration-stochastic-allocation.md` — orchestration + bandit scope
5. `docs/SETUP.md` — toolchain, the cuda-oxide env exports, absolute nvcc/cuobjdump paths
6. `scripts/kat_oracle.py` + `kernels/cuda-ref/test_vectors.json` — THE bit-exact oracle. Run
   `.venv/bin/python scripts/kat_oracle.py verify` → must print ALL PASS before you trust anything.

## HARD INVARIANTS (violating any of these breaks the project — rationale included)
- **Bit-exact crypto, NO tolerance.** One wrong bit in a 256-bit result = wrong key. Every kernel
  output is compared *exactly* to `kat_oracle.py`. Never use float tolerance anywhere in the EC/hash
  path. (Rationale: this domain has zero partial credit.)
- **No transaction broadcast / signing-for-spend, ever.** (Rationale: ADR-0001 non-goal; broadcast =
  theft vector. The repo demonstrates *solving*, never *spending*.)
- **cuda-oxide env before any `cargo oxide`:** `export CUDA_HOME=/usr/local/cuda;
  export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so;
  export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin`. (Rationale: libNVVM shadow bug → fake
  perf + can't target sm_120.)
- **Absolute toolchain paths:** `/usr/local/cuda/bin/nvcc` and `/usr/local/cuda/bin/cuobjdump`
  (the apt shims are stale CUDA 12.0 and silently mishandle sm_120).
- **Target `sm_120`** (RTX 5090, consumer Blackwell). No `tcgen05`/`sm_100a`-only instructions.
- **Python via uv only.** `.venv` exists; install with `VIRTUAL_ENV=.venv uv pip install ...`.

## VERIFICATION DISCIPLINE
- Prove each acceptance with a real run, not a claim. Run the relevant test/bench green before
  moving to the next wave. Commit small (one logical unit per commit), push when green.
- Re-bench any baseline in the SAME session on the SAME idle GPU; cite
  `nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv`.
- For any cuda-oxide perf gap, disassemble the cubin (`cuobjdump --dump-sass`) — PTX alone misses it.

---

## Wave 1 — CUDA-C secp256k1 reference (`kernels/cuda-ref/`)  [LOAD-BEARING, do first]
Build a correct, single-GPU CUDA-C brute-force scanner for the no-pubkey case.
- secp256k1 field arithmetic mod p (256-bit: add/sub/mul/sqr/inv via Fermat or batched Montgomery)
- point add/double (Jacobian), scalar mult (double-and-add), compressed pubkey encode
- sha256 + ripemd160 (hash160) on-GPU
- compare hash160 against a target; report match
- host harness: load `test_vectors.json`, run each privkey through the GPU pipeline, assert the
  GPU-produced hash160 == the vector's `hash160_hex` **bit-exact**

**Acceptance (testable):**
- `kernels/cuda-ref/` builds with `/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120`.
- A test binary reads `test_vectors.json` and prints `ALL <N> VECTORS PASS` (hash160 bit-exact).
  Add a tiny scan test: seed the target = vector[0].hash160, scan a small range containing its
  privkey, assert FOUND with the right key.
- `results/wave1-cuda-keys-per-sec.csv` with median keys/s over 10 iters + nvidia-smi thermal line.

## Wave 2 — cuda-oxide port (`kernels/oxide-ec/`)
Same algorithm in cuda-oxide Rust→PTX. Expect 256-bit limb arithmetic without inline asm to be the
crux (cuda-oxide v0.1.0 has no `asm!` in kernels — document every friction in
`docs/research/wave2-oxide-notes.md`).

**Acceptance:**
- `cargo oxide run` (with env exports) produces correct hash160 for all `test_vectors.json` entries,
  bit-exact vs the oracle.
- `results/wave2-oxide-keys-per-sec.csv` on the SAME idle GPU as Wave 1 (re-bench cuda-ref in the
  same session). SASS instruction-mix (IMAD/IADD3 count) for both in `docs/research/wave2-oxide-notes.md`.
- The keys/s gap (oxide vs cuda-ref) is the headline result — report it honestly either direction.

## Wave 3 — Fast-path constant-factor optimizations (`kernels/cuda-ref/`, then port)
Per ADR-0003, stack the real constant-factor wins on the CUDA-C path:
- batched Montgomery inversion (Montgomery's trick: 1 modinv + 3n modmul), affine coords
- GLV/endomorphism scalar split (secp256k1 λ)
- negation-map symmetry

**Acceptance:** each optimization lands as its own commit with a before/after keys/s delta in
`results/wave3-fastpath.csv`; correctness still ALL PASS vs the oracle after each. Document which
optimization gave what lift (expect batched-inversion >> endomorphism > negation).

## Wave 4 — N-GPU orchestrator (`orchestrator/`)
Per `orchestrator/README.md` + ADR-0004. Pure-Python coordinator/worker (msgpack over TCP).
- coordinator: range registry w/ lease TTL, FOUND handling, heartbeat, DP store stub
- worker: wraps the Wave-1/3 kernel binary via subprocess, lease loop, heartbeat
- `allocator.py`: brute-force ranking + kangaroo Gittins skeleton per `docs/research/bandit-allocation.md`

**Acceptance:**
- `pytest orchestrator/tests/` green: lease disjointness (no two workers get overlapping ranges),
  lease-TTL reclaim on worker death, FOUND propagation, allocator ranks a narrower range higher.
- A localhost integration test: 1 coordinator + 3 worker procs (kernel may be a CPU stub for the
  test) sweep a small toy keyspace and collectively report FOUND for a planted key. Linear-ish
  aggregate keys/s vs worker count logged to `results/wave4-scaling.csv`.

## Wave 5 — Quantum toy simulator (`research/quantum-toy/`)  [pedagogical, NOT to scale]
Per `docs/research/quantum-angle.md`. A Qiskit (local Aer simulator — NO AWS/Braket creds needed,
do NOT attempt real Braket calls) demonstration of Shor's ECDLP on a **tiny** curve (≤ 8-bit field).
Clearly labeled: this is mechanism illustration; it does NOT scale to secp256k1 (see ADR-0002).

**Acceptance:** `research/quantum-toy/shor_toy_ecdlp.py` runs on the Aer simulator, recovers the
discrete log on a documented small curve, and `pytest` asserts the recovered `d` is correct. README
in that dir restates the not-to-scale caveat and the logical/physical-qubit gap.

---

## DO-NOT
- Do NOT build any transaction/broadcast/spend path.
- Do NOT add index calculus, ML-keyspace-learning, rainbow tables, or any nvfp4/FP4 "coarse search"
  (all ruled out in ADR-0003 / README — they're category errors here).
- Do NOT attempt real Amazon Braket API calls (no creds, and it can't run the real problem anyway).
- Do NOT touch anything outside this repo. Do NOT restart any system service. Do NOT recursively
  spawn another background coding agent.
- Do NOT claim a perf result without a real run + nvidia-smi thermal context in the CSV.
