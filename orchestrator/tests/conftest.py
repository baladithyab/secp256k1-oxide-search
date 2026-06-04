"""Make the repo root importable so ``import orchestrator`` works.

Running ``pytest orchestrator/tests/`` from the repo root normally puts the
root on ``sys.path``, but we add it explicitly so the suite (and the worker
subprocesses it spawns) resolve the package regardless of the invocation cwd.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
