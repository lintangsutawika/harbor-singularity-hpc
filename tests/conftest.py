"""Make ``src/`` importable so tests run against a checkout without installing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
