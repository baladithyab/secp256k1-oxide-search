# ADR-0002: Amazon Braket / Quantum — What It Would and Would Not Buy Us

- **Status:** Accepted
- **Date:** 2026-06
- **Context:** Codeseys asked "see what we could do if we had access to Amazon Braket." This ADR is
  the honest, current (2026) answer, separated by puzzle type because the answer differs sharply.

## TL;DR

| Target | Quantum relevance | Verdict |
|---|---|---|
| **#71 etc. (no exposed pubkey)** | Shor needs a **public key**. There isn't one. Only Grover applies → quadratic, not exponential, speedup over the 2⁷⁰ brute force, and **Braket has nothing remotely close to the qubit count**. | ❌ No help, today or for a long time |
| **#135 etc. (exposed pubkey)** | This IS the ECDLP Shor's algorithm solves in polynomial time. **But** it needs ~1,100–2,330 *logical* qubits and ~70–90M Toffoli gates. | 🟡 The *right problem shape*, but ❌ not runnable on any Braket device now or in the foreseeable Braket roadmap |

## The crucial distinction: Shor needs the public key

Shor's algorithm solves the **discrete log** problem: given public key `Q = dG`, recover `d`. That
requires `Q` to be *known*.

- **No-pubkey puzzles (#71, #72, ...)**: the address is a *hash* of the pubkey; the pubkey is not
  revealed until the coins are spent. There is no `Q` to feed Shor. The only quantum lever is
  **Grover's search** over the keyspace → speedup from `N` to `√N`. For #71 that's 2⁷⁰ → ~2³⁵
  *quantum* iterations — but each iteration is a full reversible EC-mult + hash160 circuit on a
  fault-tolerant machine. The qubit/gate budget for even one such oracle evaluation dwarfs anything
  Braket offers, and Grover doesn't parallelize well. **Net: irrelevant.**
- **Exposed-pubkey puzzles (#130/#135-class)**: the 2019 creator dust-sends exposed many pubkeys.
  *Those* are genuine ECDLP instances Shor could attack — same problem class as "break Bitcoin."

## Current (2026) resource estimates for 256-bit ECDLP via Shor

- Chevignard–Fouque–Schrottenloher (EUROCRYPT 2026): **1,098 logical qubits**, space 3.12n, via
  Legendre-symbol output compression (avoids explicit modular inversion). Below 3072-bit RSA.
- Google Quantum AI (Mar 2026 whitepaper): **1,200–1,450 logical qubits**, 70–90M Toffoli gates,
  ~9-minute wall window on a fast-clock machine — **but ~500,000 physical qubits** on a
  superconducting plane to reach that logical threshold.
- Older estimates: 2,124–2,330 logical qubits.

## What Amazon Braket actually offers (2026)

Braket brokers access to **NISQ** hardware, not fault-tolerant machines:
- Superconducting: IQM, Rigetti (≲ low-hundreds of *physical*, noisy qubits)
- Ion-trap: IonQ, AQT (tens of high-fidelity physical qubits)
- Neutral-atom: QuEra (hundreds of atoms, analog/Rydberg)

**The gap:** Shor on secp256k1 needs ~1,100+ *logical* (error-corrected) qubits = ~hundreds of
thousands of *physical* qubits. Braket's largest devices are 2–4 orders of magnitude short on
physical count and have **no error correction** to produce even one logical qubit at the required
fidelity. The current experimental ECDLP ceiling on real quantum hardware is a **6-bit** prime field
(via Shor) — vs the 256-bit we'd need.

### Quantum annealing (D-Wave-style) — also on offer elsewhere, also no

There's a line of work (Wroński et al.) mapping ECDLP → QUBO for quantum annealing. It's real and
generalizes to any curve model now, but: (a) the QUBO size for 256-bit is enormous, (b) annealing
gives **no proven complexity advantage** for this problem, (c) it's heuristic with no runtime
guarantee. Not a path.

## Decision

- **Do not** plan any executable Braket integration. We have no AWS creds wired (consistent with the
  "native auth > external creds" preference — and there's nothing to authenticate *to* that helps).
- **Do** include a `docs/research/quantum-angle.md` that documents this verdict with citations, and
  a *toy* educational artifact: a Braket/Qiskit simulation of Shor on a **tiny** curve (e.g. a
  5–8 bit toy ECDLP) to make the mechanism concrete without pretending it scales. This is the only
  honest "what we could do with Braket" deliverable — a pedagogical simulator, clearly labeled as
  not-to-scale.

## Consequences

The quantum angle is a **documentation + toy-simulation track**, not an execution track. It does not
change the negative-EV baseline (ADR-0001) for any puzzle reachable today. The one genuinely
interesting forward-looking note: if Braket (or anyone) ever fields ~10⁵–10⁶ physical qubits with
FTQC, the *exposed-pubkey* puzzles (and all exposed-pubkey BTC) fall in minutes — which is exactly
why the live network is debating post-quantum migration. That's a real story; it's just not a 2026
GPU-cluster story.
