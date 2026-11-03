import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "limb"
sys.path.insert(0, str(EXAMPLES))
