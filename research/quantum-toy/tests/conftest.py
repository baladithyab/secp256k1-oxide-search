"""Pytest config local to the quantum-toy tests.

Silences the Qiskit 2.1 deprecation notice for the still-functional QFT/
BlueprintCircuit classes (we use QFT deliberately for pedagogical clarity).
Pytest installs its own warnings filter, so we re-register here at collection
time rather than relying on the module-level filter in shor_toy_ecdlp.py.
"""

import warnings


def pytest_configure(config):
    for pat in (
        r".*qiskit\.circuit\.library\.basis_change\.qft\.QFT.*",
        r".*BlueprintCircuit.*",
    ):
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=pat)
