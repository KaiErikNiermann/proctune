"""Run inside Blender (scripts/gui.py): register proctune from this checkout and open the project given after
``--``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_env

blender_env.setup()
from proctune.blender import addon  # noqa: E402

addon.register()
args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if args:
    addon.open_reporting(Path(args[0]).resolve())
