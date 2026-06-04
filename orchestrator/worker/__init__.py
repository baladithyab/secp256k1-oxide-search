"""Worker subpackage: stub CPU kernel + worker client."""

from .stub_kernel import sweep_block, target_digest, SweepResult

__all__ = ["sweep_block", "target_digest", "SweepResult"]
