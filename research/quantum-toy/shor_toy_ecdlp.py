#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shor_toy_ecdlp.py -- Shor's algorithm recovering an elliptic-curve discrete log
on a DOCUMENTED, TINY toy curve, using only the LOCAL Qiskit-Aer statevector
simulator.

================================  WHAT THIS IS  ===============================
This is a PEDAGOGICAL mechanism illustration of the quantum attack on ECDLP.
It is NOT to scale and provides NO leverage against secp256k1. See ADR-0002
(docs/research/quantum-angle.md) and the README in this directory.

We solve the elliptic-curve discrete log problem (ECDLP) on a tiny curve:
given a base point P of small order r and a public point Q = d*P, recover the
scalar d. This is *exactly* the problem shape Shor's algorithm attacks in
polynomial time -- the same shape as "break an exposed-pubkey Bitcoin key" --
just shrunk to a group of order r = 8 so the whole thing fits in 9 qubits.

================================  THE MECHANISM  ==============================
ECDLP-as-hidden-subgroup / period-finding (Proos-Zalka, Eicher-Opoku, Kaye):

  Define f(a, b) = a*P + b*Q  on the cyclic group <P> of order r.
  Because Q = d*P, we have  f(a, b) = (a + b*d)*P, so f only depends on
  (a + b*d) mod r. The function is therefore constant on the cosets of the
  hidden subgroup  L = { (a, b) : a + b*d == 0  (mod r) }  of Z_r x Z_r.

  Quantum recipe:
    1. Two index registers a, b each of n qubits with 2^n = r (here n = 3).
    2. Hadamard both -> uniform superposition over all (a, b).
    3. Oracle U_f writes the group element f(a, b) into an output register.
    4. QFT over each index register, then measure (a, b) -> (y_a, y_b).
    5. The measured pairs concentrate on the dual lattice of L. With r a power
       of two the peaks are EXACT and obey  y_b == d * y_a (mod r).
       Hence for any measurement with gcd(y_a, r) == 1:
           d == y_b * y_a^{-1}   (mod r).

  We take the classical majority vote over shots to read off d, then verify
  d*P == Q with ordinary Python EC arithmetic.

============================  THE PART THAT DOES NOT SCALE  ===================
A faithful, fully-reversible elliptic-curve point-addition oracle (the real
Shor-on-ECDLP circuit) is enormous: ~1,100+ logical qubits and ~70-90M Toffoli
gates for secp256k1 (ADR-0002). We sidestep that *entirely* for the toy by
PRECOMPUTING the group action a*P + b*Q over the small group into a lookup
table, and realizing the oracle as a single (controlled-)permutation unitary
over a handful of qubits.

  >>> This table-driven oracle is the ONLY reason the toy is tractable, and it
  >>> is precisely the step that does NOT scale: building the table requires
  >>> enumerating the whole group (here 8 elements; for secp256k1 it would be
  >>> ~2^256 elements -- impossible). A real attack must instead synthesize the
  >>> EC group law as a reversible arithmetic circuit, which is what blows the
  >>> qubit/gate budget up to the fault-tolerant regime.

The honest takeaway: the *algorithm* is correct and recovers d here exactly as
it would on a 256-bit curve given a fault-tolerant machine -- but the resource
cost on a real curve is astronomically beyond any 2026 hardware (Amazon Braket
included; it brokers only NISQ devices, no error correction). See README.

No AWS / Braket / network access is used or attempted anywhere in this file.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

# Qiskit 2.x API: execute() was removed in 1.0+. We use AerSimulator().run(...).
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit_aer import AerSimulator

# The QFT *class* is deprecated in favour of QFTGate in Qiskit 2.1+, but it is
# still fully functional and the clearest pedagogical form. Silence the notice
# (match on the deprecation message text so it is suppressed at call time too).
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*qiskit\.circuit\.library\.basis_change\.qft\.QFT.*",
)

# Deterministic simulator seed so the toy is reproducible / test-stable.
DEFAULT_SEED = 12345
DEFAULT_SHOTS = 2048


# =============================================================================
# Plain-Python elliptic-curve arithmetic over GF(p)  (the classical oracle of
# truth -- the quantum result is verified against this).
# =============================================================================
@dataclass(frozen=True)
class Curve:
    """Short Weierstrass curve y^2 = x^3 + a*x + b over GF(p)."""

    p: int
    a: int
    b: int

    def __post_init__(self) -> None:
        # Non-singularity: discriminant 4a^3 + 27b^2 != 0 (mod p).
        if (4 * self.a**3 + 27 * self.b**2) % self.p == 0:
            raise ValueError("singular curve (discriminant == 0)")


# Point at infinity is represented by None; affine points by (x, y) tuples.
Point = "tuple[int, int] | None"


def _inv_mod(x: int, p: int) -> int:
    """Modular inverse via extended Euclid (p need not be prime for our small p,
    but here p is prime so every nonzero element is invertible)."""
    x %= p
    if x == 0:
        raise ZeroDivisionError("no inverse of 0")
    g, s = p, 0
    r_, t = x, 1
    while r_ != 0:
        q = g // r_
        g, r_ = r_, g - q * r_
        s, t = t, s - q * t
    if g != 1:
        raise ZeroDivisionError(f"{x} not invertible mod {p}")
    return s % p


def ec_add(P, Q, cv: Curve):
    """Elliptic-curve point addition on `cv`. Handles infinity and doubling."""
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % cv.p == 0:
        return None  # P + (-P) = O
    if P == Q:
        m = ((3 * x1 * x1 + cv.a) * _inv_mod(2 * y1, cv.p)) % cv.p
    else:
        m = ((y2 - y1) * _inv_mod((x2 - x1) % cv.p, cv.p)) % cv.p
    x3 = (m * m - x1 - x2) % cv.p
    y3 = (m * (x1 - x3) - y1) % cv.p
    return (x3, y3)


def ec_mul(k: int, P, cv: Curve):
    """Scalar multiplication k*P via double-and-add (k may be reduced mod order
    by the caller; here groups are tiny so a simple ladder is fine)."""
    R = None
    addend = P
    k = int(k)
    if k < 0:
        # -k * P = k * (-P); negate by flipping y.
        k = -k
        addend = None if P is None else (P[0], (-P[1]) % cv.p)
    while k > 0:
        if k & 1:
            R = ec_add(R, addend, cv)
        addend = ec_add(addend, addend, cv)
        k >>= 1
    return R


def point_order(P, cv: Curve) -> int:
    """Order of P in the group (smallest n>0 with n*P == O)."""
    if P is None:
        return 1
    R = P
    n = 1
    while R is not None:
        R = ec_add(R, P, cv)
        n += 1
    return n


# =============================================================================
# The documented toy instance.
# =============================================================================
@dataclass(frozen=True)
class ToyInstance:
    curve: Curve
    P: tuple
    Q: tuple  # may be None if d is a multiple of r, but we keep d in 1..r-1
    d_true: int
    r: int  # order of P
    n: int  # qubits per index register, 2^n == r

    @property
    def num_qubits(self) -> int:
        # two index registers (a, b) + one output register, each n qubits.
        return 3 * self.n


def make_default_instance(d_true: int = 3) -> ToyInstance:
    """The chosen toy: curve y^2 = x^3 + 4x + 1 over GF(5).

    The point P = (0, 1) has order r = 8 -- a *power of two*, which makes the
    QFT period peaks exact and the toy fully deterministic. Q = d*P.
    """
    cv = Curve(p=5, a=4, b=1)
    P = (0, 1)
    r = point_order(P, cv)
    if r != 8:
        raise AssertionError(f"expected order-8 base point, got {r}")
    n = int(round(math.log2(r)))
    if (1 << n) != r:
        raise AssertionError("this toy requires r to be a power of two for exact QFT")
    if not (1 <= d_true < r):
        raise ValueError("d_true must be in 1..r-1")
    Q = ec_mul(d_true, P, cv)
    return ToyInstance(curve=cv, P=P, Q=Q, d_true=d_true, r=r, n=n)


# =============================================================================
# Oracle: the precomputed (table-driven) group-action permutation.  <<< DOES NOT SCALE
# =============================================================================
def build_group_action_table(inst: ToyInstance) -> dict:
    """Precompute f(a, b) = index of (a*P + b*Q) in the cyclic group <P>.

    THIS ENUMERATION over all (a, b) is the non-scaling step: it touches the
    whole group. For secp256k1 the group has ~2^256 elements, so no such table
    can ever be built -- a real oracle must be a reversible EC arithmetic
    circuit instead (the source of the ~1100-logical-qubit cost; see ADR-0002).
    """
    cv = inst.curve
    # discrete-log index map:  point -> k  such that point == k*P
    index_of = {ec_mul(k, inst.P, cv): k for k in range(inst.r)}
    table = {}
    for av in range(inst.r):
        for bv in range(inst.r):
            pt = ec_add(ec_mul(av, inst.P, cv), ec_mul(bv, inst.Q, cv), cv)
            table[(av, bv)] = index_of[pt]
    return table


def build_oracle_unitary(inst: ToyInstance, table: dict) -> np.ndarray:
    """Reversible oracle U_f : |a>|b>|out>  ->  |a>|b>|out XOR f(a,b)>.

    Realized as an explicit permutation matrix over the 3n qubits. (For the toy
    this is a 2^9 x 2^9 = 512x512 permutation -- trivial. For a real curve the
    matrix would have 2^(huge) rows, which is why this representation is purely
    a toy device.)

    Qubit/bit ordering of the basis integer used here:
        bit i in [0, n)        -> register a (little-endian within register)
        bit i in [n, 2n)       -> register b
        bit i in [2n, 3n)      -> output register
    This matches the qubit list we pass to UnitaryGate below.
    """
    n = inst.n
    N = 1 << (3 * n)

    def encode(av: int, bv: int, ov: int) -> int:
        return (av & (inst.r - 1)) | ((bv & (inst.r - 1)) << n) | ((ov & (inst.r - 1)) << (2 * n))

    U = np.zeros((N, N), dtype=float)
    for av in range(inst.r):
        for bv in range(inst.r):
            f = table[(av, bv)]
            for ov in range(inst.r):
                src = encode(av, bv, ov)
                dst = encode(av, bv, ov ^ f)  # XOR-write keeps it reversible
                U[dst, src] = 1.0
    # Sanity: must be a genuine permutation (orthogonal, U U^T == I).
    assert np.allclose(U @ U.T, np.eye(N)), "oracle is not a valid permutation"
    return U


# =============================================================================
# The quantum circuit and Aer execution.
# =============================================================================
def build_circuit(inst: ToyInstance) -> QuantumCircuit:
    n = inst.n
    table = build_group_action_table(inst)
    U = build_oracle_unitary(inst, table)

    reg_a = QuantumRegister(n, "a")
    reg_b = QuantumRegister(n, "b")
    reg_o = QuantumRegister(n, "o")
    cl_a = ClassicalRegister(n, "ma")
    cl_b = ClassicalRegister(n, "mb")
    qc = QuantumCircuit(reg_a, reg_b, reg_o, cl_a, cl_b)

    # 1. Uniform superposition over both index registers.
    for q in reg_a:
        qc.h(q)
    for q in reg_b:
        qc.h(q)

    # 2. Apply the group-action oracle. Qubit order MUST match build_oracle_unitary:
    #    [a..., b..., o...].
    qc.append(UnitaryGate(U, label="U_f (group action)"), list(reg_a) + list(reg_b) + list(reg_o))

    # 3. QFT over each index register to expose the period / dual lattice.
    qc.append(QFT(n, do_swaps=True), reg_a)
    qc.append(QFT(n, do_swaps=True), reg_b)

    # 4. Measure the two index registers.
    qc.measure(reg_a, cl_a)
    qc.measure(reg_b, cl_b)
    return qc


def run_on_aer(qc: QuantumCircuit, shots: int = DEFAULT_SHOTS, seed: int = DEFAULT_SEED) -> dict:
    """Run on the LOCAL Aer statevector simulator (no network, no Braket)."""
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


# =============================================================================
# Classical post-processing: read d off the measured period lattice.
# =============================================================================
def recover_d(counts: dict, inst: ToyInstance) -> tuple:
    """From measured (y_a, y_b) pairs, recover d.

    Because r is a power of two the QFT peaks are exact and satisfy
        y_b == d * y_a   (mod r).
    For any shot with gcd(y_a, r) == 1 this inverts to d = y_b * y_a^{-1} mod r.
    We majority-vote over all such shots (here it is unanimous).

    Returns (d_recovered, candidate_votes_dict).
    """
    r = inst.r
    votes: dict = {}
    for bitstring, count in counts.items():
        # Qiskit prints classical registers space-separated in reverse order of
        # declaration: "<mb> <ma>".
        mb_str, ma_str = bitstring.split()
        y_a = int(ma_str, 2)
        y_b = int(mb_str, 2)
        if math.gcd(y_a, r) != 1:
            continue  # y_a not invertible mod r -> ambiguous, skip
        d_cand = (y_b * _inv_mod(y_a, r)) % r
        votes[d_cand] = votes.get(d_cand, 0) + count
    if not votes:
        raise RuntimeError("no invertible measurement found; cannot recover d")
    d_rec = max(votes, key=votes.get)
    return d_rec, votes


# =============================================================================
# End-to-end driver.
# =============================================================================
def solve_ecdlp_toy(d_true: int = 3, shots: int = DEFAULT_SHOTS, seed: int = DEFAULT_SEED,
                    verbose: bool = True) -> dict:
    """Full pipeline: build instance -> quantum period-find on Aer -> recover d.

    Returns a result dict with everything the test asserts on.
    """
    inst = make_default_instance(d_true=d_true)
    qc = build_circuit(inst)
    counts = run_on_aer(qc, shots=shots, seed=seed)
    d_rec, votes = recover_d(counts, inst)

    # Independent classical verification of the curve arithmetic.
    Q_check = ec_mul(d_rec, inst.P, inst.curve)
    curve_ok = Q_check == inst.Q

    out = {
        "p": inst.curve.p,
        "a": inst.curve.a,
        "b": inst.curve.b,
        "P": inst.P,
        "Q": inst.Q,
        "d_true": inst.d_true,
        "r": inst.r,
        "n_qubits": inst.num_qubits,
        "d_recovered": d_rec,
        "votes": votes,
        "curve_check_dP_eq_Q": curve_ok,
        "Q_from_recovered_d": Q_check,
    }

    if verbose:
        print("=" * 72)
        print("Shor's algorithm vs ECDLP -- TOY (pedagogical, NOT to scale)")
        print("=" * 72)
        print(f"  Curve:        y^2 = x^3 + {inst.curve.a}*x + {inst.curve.b}  over GF({inst.curve.p})")
        print(f"  Base point P: {inst.P}")
        print(f"  Public Q:     {inst.Q}   (= d*P)")
        print(f"  True d:       {inst.d_true}")
        print(f"  Group order r (= ord(P)): {inst.r}   [power of two -> exact QFT]")
        print(f"  Qubit count (2 index + 1 output register, {inst.n} each): {inst.num_qubits}")
        print(f"  Aer shots/seed: {shots} / {seed}")
        print("-" * 72)
        print(f"  Recovered d:  {d_rec}   (candidate votes: {votes})")
        print(f"  Verify d*P == Q in plain Python: {curve_ok}  "
              f"({d_rec}*P = {Q_check})")
        print(f"  SUCCESS: {d_rec == inst.d_true and curve_ok}")
        print("-" * 72)
        print(f"  This toy used {inst.num_qubits} qubits. A real attack on secp256k1 needs")
        print(f"  ~1,100+ LOGICAL qubits (~500,000 physical) -- see ADR-0002 / README.")
        print(f"  Gap factor (logical, vs ~1100): ~{1100 / inst.num_qubits:.0f}x more qubits,")
        print(f"  and the table-driven oracle here does NOT exist for a real curve.")
        print("=" * 72)

    return out


if __name__ == "__main__":
    solve_ecdlp_toy()
