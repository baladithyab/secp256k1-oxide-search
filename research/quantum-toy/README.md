# Quantum TOY: Shor's algorithm vs ECDLP (pedagogical, NOT to scale)

> **This is a mechanism illustration only.** It recovers a discrete log on a
> deliberately tiny elliptic curve using the *local* Qiskit-Aer statevector
> simulator. It provides **zero** leverage against secp256k1 or any real key.
> It exists to make the quantum attack concrete and honest, per **ADR-0002**
> (`docs/research/quantum-angle.md`).

No AWS / Amazon Braket / network access is used or attempted anywhere. Local
Aer only.

## What it does

Solves the **elliptic-curve discrete log problem (ECDLP)**: given a base point
`P` of small order `r` and a public point `Q = d·P`, recover the secret scalar
`d`. This is *exactly* the problem shape Shor's algorithm breaks in polynomial
time — the same shape as attacking an exposed-pubkey Bitcoin key — shrunk so the
whole circuit fits in **9 qubits**.

### The toy instance (documented)

| Param | Value |
|---|---|
| Field prime `p` | `5` |
| Curve | `y² = x³ + 4x + 1` over `GF(5)` |
| Base point `P` | `(0, 1)` |
| Group order `r = ord(P)` | `8` (cyclic, **a power of two**) |
| Secret scalar `d` (planted) | `3` |
| Public point `Q = d·P` | `(1, 4)` |
| Qubit count | **9** (two 3-qubit index registers `a`, `b` + a 3-qubit output register) |

`r = 8` is chosen as a power of two so the QFT period peaks are **exact** and
the toy is **fully deterministic** (no continued-fraction step needed). The
simulator is seeded, so every run reproduces.

### The mechanism (ECDLP as period finding / hidden subgroup)

Define `f(a, b) = a·P + b·Q`. Since `Q = d·P`, we have `f(a, b) = (a + b·d)·P`,
so `f` depends only on `(a + b·d) mod r`. It is constant on cosets of the hidden
subgroup `L = { (a, b) : a + b·d ≡ 0 (mod r) }`. Quantum recipe:

1. Hadamard the two index registers → uniform superposition over all `(a, b)`.
2. Apply the oracle `U_f` writing the group element `f(a, b)` into the output register.
3. QFT each index register, then measure → `(yₐ, y_b)`.
4. With `r` a power of two the peaks are exact and obey `y_b ≡ d·yₐ (mod r)`,
   so for any shot with `gcd(yₐ, r) = 1`: **`d = y_b · yₐ⁻¹ (mod r)`**.
5. Majority-vote over shots (unanimous here), then verify `d·P == Q` in plain Python.

## Run it

```bash
.venv/bin/python research/quantum-toy/shor_toy_ecdlp.py
.venv/bin/python -m pytest research/quantum-toy/tests/ -v
```

The test suite runs the real Aer simulation end-to-end and asserts the recovered
`d` equals the planted `d` **and** that `d·P == Q` on the curve, for every
non-trivial `d ∈ {1..7}`.

## The part that does NOT scale (read this)

A *faithful* Shor-on-ECDLP oracle synthesizes the elliptic-curve group law as a
fully reversible arithmetic circuit. That is enormous. We sidestep it entirely:
we **precompute the group action `a·P + b·Q` over the tiny group into a lookup
table** and realize the oracle as a single permutation unitary over 9 qubits.

> Building that table requires enumerating the **whole group** (8 elements here;
> ~2²⁵⁶ for secp256k1 — impossible). The table-driven oracle is the *only* reason
> the toy is tractable, and it is *precisely* the step that does not generalize.
> A real attack must synthesize the EC group law reversibly, which is what blows
> the qubit/gate budget into the fault-tolerant regime.

## The logical-vs-physical qubit gap (per ADR-0002)

| | This toy | secp256k1 (real ECDLP via Shor) |
|---|---|---|
| Logical qubits | **9** | **~1,098–2,330** (1,098: Chevignard–Fouque–Schrottenloher EUROCRYPT 2026; 1,200–1,450: Google Quantum AI Mar-2026) |
| Toffoli gates | a few hundred | **~70–90 million** |
| Physical qubits | 9 (simulated) | **~500,000** superconducting, with error correction |
| Hardware that can run it | a laptop | **none in 2026** |

This toy uses **9** qubits; a real attack needs **~1,100+ logical** qubits —
roughly **122×** more *logical* qubits (and ~500,000 *physical*, error-corrected
qubits to realize them). The current experimental ECDLP ceiling on real quantum
hardware is a **6-bit** prime field — versus the 256 bits we would need.

### Why Amazon Braket does not help

Braket brokers only **NISQ** hardware — IQM/Rigetti (low-hundreds of noisy
*physical* qubits), IonQ/AQT (tens of high-fidelity physical qubits), QuEra
(hundreds of neutral atoms). **None has error correction**, so none yields even
*one* logical qubit at the fidelity Shor requires, and all are 2–4 orders of
magnitude short on physical count. Per ADR-0002 there is **no executable Braket
integration** in this repo and none is planned — the quantum angle is a
documentation + toy-simulation track, not an execution track.

The one forward-looking note: *if* anyone ever fields ~10⁵–10⁶ physical qubits
with fault tolerance, every **exposed-pubkey** key (puzzles and real BTC alike)
falls in minutes — which is exactly why the network is debating post-quantum
migration. That is a real story; it is not a 2026 story.
