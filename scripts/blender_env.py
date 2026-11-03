"""Make this checkout importable inside Blender: ``src`` first, the dev site folder (packages Blender's Python lacks,
see scripts/blender_site.py; ``dev_site`` in the proctune config) last, so Blender's own numpy wins."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup() -> Path:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from proctune import config  # noqa: PLC0415

    site = str(config.load().dev_site)
    if site not in sys.path:
        sys.path.append(site)
    return ROOT
