"""Open Blender with proctune loaded from this checkout:

    python scripts/gui.py [PROJECT.json] [--install-missing] [-- blender args...]

Before starting Blender it checks the project with Blender's own Python and reports what is missing, which Python
looked and where, and where an install would go. It installs nothing unless asked (--install-missing). Blender is
``blender`` from the proctune config (``PROCTUNE_BLENDER`` or the config file); folders in its ``paths`` (your own
environment's site-packages, say) are added to Blender's import path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proctune import config  # noqa: E402


def preflight(blender: str, project: Path, install: bool) -> dict[str, object]:
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    cmd = [blender, "-b", "--factory-startup", "--python", str(ROOT / "scripts" / "preflight.py"), "--", str(project)]
    r = subprocess.run(
        [*cmd, *(["--install-missing"] if install else [])], env=env, capture_output=True, text=True, check=False
    )
    report = [
        line for line in r.stdout.splitlines() if not line.startswith(("PROCTUNE_PREFLIGHT", "Blender ", "Read "))
    ]
    sys.stdout.write("\n".join(line for line in report if line.strip()) + "\n")
    last = next((line for line in reversed(r.stdout.splitlines()) if line.startswith("PROCTUNE_PREFLIGHT ")), None)
    if last is None:
        return {"error": f"the check did not finish (exit {r.returncode}): {r.stderr.strip()[-800:]}"}
    return json.loads(last.removeprefix("PROCTUNE_PREFLIGHT "))


def main(argv: list[str]) -> int:
    blender_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    ap = argparse.ArgumentParser(
        prog="python scripts/gui.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("project", nargs="?", type=Path, default=ROOT / "examples" / "limb" / "limb.json")
    ap.add_argument(
        "--install-missing", action="store_true", help="install what the project lacks (after reporting where)"
    )
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    blender = config.load().blender
    project = args.project.resolve()
    result = preflight(blender, project, args.install_missing)
    if result.get("ok"):
        print(f"{project.name}: everything it needs imports in Blender's Python")
    elif missing := result.get("missing"):
        print(
            f"\n{project.name} cannot open yet: missing {', '.join(map(str, missing))}"
            " (see above for where it looked).\n"
            "Blender opens anyway; its proctune panel offers to install them, after showing where. Or rerun with\n"
            "--install-missing, or point `paths` in the proctune config at your own environment's site-packages."
        )
    elif result.get("error"):
        print(f"\n{project.name}: {result['error']}")
    cmd = [blender, *blender_args, "--python", str(ROOT / "scripts" / "blender_dev.py"), "--", str(project)]
    sys.stdout.flush()  # execvp replaces the process: anything still buffered would be lost
    os.execvp(cmd[0], cmd)  # replaces this process with Blender
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
