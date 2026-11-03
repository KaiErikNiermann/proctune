"""Run inside Blender (scripts/gui.sh): put this checkout and .blender-site on the path, register proctune and open
the project given after ``--``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# last: it only fills in what Blender's Python lacks (its own numpy must win, or two numpys mix)
if str(ROOT / ".blender-site") not in sys.path:
    sys.path.append(str(ROOT / ".blender-site"))

from proctune.blender import addon  # noqa: E402

addon.register()
args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if args:
    addon.open_project(Path(args[0]).resolve())
