"""Make the flat eval/ scripts importable from tests (they are scripts, not a package)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
