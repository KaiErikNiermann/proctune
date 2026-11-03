"""Make this checkout importable inside Blender: ``src`` first, ``.blender-site`` (packages Blender's Python lacks,
see scripts/blender-site.sh) last, so Blender's own numpy wins and two numpys never mix."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup() -> Path:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    if str(ROOT / ".blender-site") not in sys.path:
        sys.path.append(str(ROOT / ".blender-site"))
    return ROOT
