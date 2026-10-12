"""Compatibility import for deployments using the old single-file entry point.

The installed package under src/ is the sole implementation.
"""
import sys
from pathlib import Path

_source = str(Path(__file__).resolve().parent / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)
from harbor_singularity_hpc.environment import SingularityWritableEnvironment

__all__ = ["SingularityWritableEnvironment"]
