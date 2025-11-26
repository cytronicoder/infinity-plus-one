import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC not in map(Path, sys.path):
    sys.path.insert(0, str(SRC))
