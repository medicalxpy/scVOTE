"""
Incremental evaluation package.

This package contains utilities for evaluating incremental TopicStore
alignment/merging runs. The public entrypoint is ``run_incremental_eval``,
which is imported by the top-level ``incremental_eval.py`` CLI wrapper.
"""

from .core import run_incremental_eval  # noqa: F401

