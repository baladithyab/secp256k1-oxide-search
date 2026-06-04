#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end tests for the toy Shor ECDLP simulator.

Each test runs the *actual* Qiskit-Aer statevector simulation (local only --
no AWS/Braket, no network) and asserts the quantum-recovered discrete log d
matches the planted value AND that d*P == Q on the curve. The plain-Python EC
arithmetic is independently checked too.

The simulator is seeded, so these are deterministic and fast (each Aer run is a
9-qubit circuit -> a 512-amplitude statevector).
"""

import math
import os
import sys

import pytest

# Make the simulator module importable regardless of pytest's rootdir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shor_toy_ecdlp as toy  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-Python curve arithmetic sanity (the oracle of truth).
# ---------------------------------------------------------------------------
def test_curve_is_nonsingular_and_order_is_8():
    inst = toy.make_default_instance(d_true=3)
    cv = inst.curve
    # Non-singular discriminant.
    assert (4 * cv.a**3 + 27 * cv.b**2) % cv.p != 0
    # Base point has order exactly 8, a power of two.
    assert toy.point_order(inst.P, cv) == 8
    assert inst.r == 8
    assert (1 << inst.n) == inst.r


def test_qubit_count_is_small_and_tractable():
    inst = toy.make_default_instance(d_true=3)
    # 3 registers x 3 qubits = 9 qubits => 2^9 = 512 amplitudes. Well under the
    # ~24-26 qubit statevector ceiling.
    assert inst.num_qubits == 9
    assert inst.num_qubits < 24


def test_group_action_table_matches_linear_formula():
    """f(a,b) must equal (a + b*d) mod r for the planted d -- the property the
    whole period-finding argument rests on."""
    d = 3
    inst = toy.make_default_instance(d_true=d)
    table = toy.build_group_action_table(inst)
    for a in range(inst.r):
        for b in range(inst.r):
            assert table[(a, b)] == (a + b * d) % inst.r


def test_dP_equals_Q_in_plain_python():
    for d in range(1, 8):
        inst = toy.make_default_instance(d_true=d)
        assert toy.ec_mul(d, inst.P, inst.curve) == inst.Q


# ---------------------------------------------------------------------------
# The real quantum end-to-end test.
# ---------------------------------------------------------------------------
def test_shor_recovers_planted_d_on_aer():
    res = toy.solve_ecdlp_toy(d_true=3, shots=2048, seed=toy.DEFAULT_SEED, verbose=True)
    # Quantum-recovered discrete log equals the planted scalar.
    assert res["d_recovered"] == res["d_true"] == 3
    # And the recovered scalar genuinely satisfies d*P == Q on the curve.
    assert res["curve_check_dP_eq_Q"] is True
    assert res["Q_from_recovered_d"] == res["Q"]
    assert res["n_qubits"] == 9


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6, 7])
def test_shor_recovers_every_d(d):
    """Sweep all non-trivial discrete logs in the order-8 group; the algorithm
    must recover each one exactly via the Aer simulation."""
    res = toy.solve_ecdlp_toy(d_true=d, shots=2048, seed=1000 + d, verbose=False)
    assert res["d_recovered"] == d
    assert res["curve_check_dP_eq_Q"] is True
    assert toy.ec_mul(res["d_recovered"], res["P"], toy.Curve(res["p"], res["a"], res["b"])) == res["Q"]


def test_recovery_relation_is_yb_eq_d_times_ya():
    """White-box check: every coprime measurement obeys y_b == d*y_a (mod r),
    and the majority vote is unanimous (deterministic toy)."""
    d = 3
    inst = toy.make_default_instance(d_true=d)
    qc = toy.build_circuit(inst)
    counts = toy.run_on_aer(qc, shots=2048, seed=toy.DEFAULT_SEED)
    seen = 0
    for bitstring, _count in counts.items():
        mb_str, ma_str = bitstring.split()
        y_a, y_b = int(ma_str, 2), int(mb_str, 2)
        if math.gcd(y_a, inst.r) == 1:
            assert (y_b - d * y_a) % inst.r == 0
            seen += 1
    assert seen > 0  # at least one usable (coprime) measurement
    d_rec, votes = toy.recover_d(counts, inst)
    assert d_rec == d
    assert len(votes) == 1  # unanimous -> fully deterministic


def test_simulation_is_deterministic_under_fixed_seed():
    qc = toy.build_circuit(toy.make_default_instance(d_true=3))
    c1 = toy.run_on_aer(qc, shots=1024, seed=777)
    c2 = toy.run_on_aer(qc, shots=1024, seed=777)
    assert c1 == c2


# ---------------------------------------------------------------------------
# Safety guard: this toy must never reach for Braket / AWS / the network.
# ---------------------------------------------------------------------------
def test_no_braket_or_aws_imports():
    import shor_toy_ecdlp  # noqa: F401

    forbidden = ("braket", "boto3", "botocore")
    for mod in forbidden:
        assert mod not in sys.modules, f"forbidden module imported: {mod}"
    # Source text must not reference Braket/AWS calls.
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shor_toy_ecdlp.py"
    )
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert "import braket" not in src
    assert "boto3" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
