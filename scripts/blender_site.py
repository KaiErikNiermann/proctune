"""Install the packages the example models and tests need into the dev site folder, for the Python that Blender
embeds: python scripts/blender_site.py. Prints the folder.

Runs Blender (``blender`` from the proctune config) in the background and lets `proctune.deps.install` call pip
with Blender's own interpreter, so the packages match it, and Blender's own numpy is not installed again.
Repeated runs are free: a marker in the folder records what was installed for which Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proctune import config  # noqa: E402

PACKAGES = ("scipy", "trimesh", "libigl", "optuna")


def main() -> int:
    cfg = config.load()
    site = cfg.dev_site
    expr = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); from pathlib import Path; from proctune import deps; "
        f"site = Path({str(site)!r}); marker = site / ('.installed-' + sys.version.split()[0]); "
        f"want = {' '.join(PACKAGES)!r}; "
        "done = marker.exists() and marker.read_text() == want; "
        "r = None if done else deps.install(want.split(), site); "
        "(None if done else marker.write_text(want)); "
        "print('PROCTUNE_SITE', 'up to date' if done else r)"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    r = subprocess.run(
        [cfg.blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", expr],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if r.returncode != 0 or "PROCTUNE_SITE" not in r.stdout:
        sys.stderr.write(f"installing into {site} with {cfg.blender} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}\n")
        return 1
    print(site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
