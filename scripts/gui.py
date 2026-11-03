"""Open Blender with proctune loaded from this checkout: python scripts/gui.py [PROJECT.json] [blender args...]

Blender is ``blender`` from the proctune config (``PROCTUNE_BLENDER`` or the config file).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proctune import config  # noqa: E402


def main(argv: list[str]) -> int:
    if subprocess.run([sys.executable, str(ROOT / "scripts" / "blender_site.py")], check=False).returncode != 0:
        return 1
    project = Path(argv[0]) if argv else ROOT / "examples" / "limb" / "limb.json"
    cmd = [config.load().blender, *argv[1:], "--python", str(ROOT / "scripts" / "blender_dev.py"), "--", str(project)]
    os.execvp(cmd[0], cmd)  # replaces this process with Blender
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
