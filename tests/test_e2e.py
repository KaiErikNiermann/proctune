"""End to end: the add-on in the real Blender GUI on a virtual display, driven by simulated input
(tests/e2e/driver.py). Needs ``blender``, ``xvfb-run`` (Arch: xorg-server-xvfb) and ``magick`` (imagemagick, for
screenshots); skipped without them. Set
PROCTUNE_E2E_KEEP=DIR to keep the screenshots."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 240

pytestmark = pytest.mark.skipif(
    any(shutil.which(t) is None for t in ("blender", "xvfb-run", "magick")),
    reason="needs blender, xvfb-run (xorg-server-xvfb) and magick (imagemagick)",
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = Path(os.environ.get("PROCTUNE_E2E_KEEP") or tmp_path_factory.mktemp("e2e"))
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([ROOT / "scripts" / "blender-site.sh"], check=True, capture_output=True)
    project = out / "limb.json"
    shutil.copy(ROOT / "examples" / "limb" / "limb.json", project)
    doc = json.loads(project.read_text())
    doc["import_paths"] = [str(ROOT / "examples" / "limb")]
    project.write_text(json.dumps(doc))
    # Blender prefers Wayland when it is there; without WAYLAND_DISPLAY it can only use the virtual X display
    env = {k: v for k, v in os.environ.items() if k != "WAYLAND_DISPLAY"} | {"DBUS_SESSION_BUS_ADDRESS": "/dev/null"}
    cmd = [
        "xvfb-run", "-a", "-s", "-screen 0 1600x1000x24",
        "blender", "--factory-startup", "--enable-event-simulate",
        "--python", str(ROOT / "tests" / "e2e" / "driver.py"), "--", str(out), str(project),
    ]  # fmt: skip
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    results = out / "results.json"
    if not results.exists():
        pytest.fail(f"no results (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    data: dict[str, Any] = json.loads(results.read_text())
    data["dir"] = str(out)
    return data


def test_driver_ran_to_the_end(run: dict[str, Any]) -> None:
    assert not run["error"], run["error"]


CHECKS = [
    "project opens",
    "slider rebuilds through the timer",
    "G starts a grab",
    "grab is captured as a tweak",
    "object transform is reset",
    "muscle stays where it was dropped",
    "skin follows the moved muscle",
    "still in sculpt mode",
    "stroke is captured as a sculpt",
    "sculpt is not empty",
    "skin follows the stroke",
    "sculpt-mode mesh matches the rebuilt muscle",
    "sculpt survives the slider",
    "viewport redraws in sculpt mode",
    "dragging the wrist bends the joint",
    "slider shows the dragged value",
    "project saves with its sculpt",
]


@pytest.mark.parametrize("name", CHECKS)
def test_check(run: dict[str, Any], name: str) -> None:
    found = [c for c in run["checks"] if c["name"] == name]
    assert found, f"{name!r} did not run (screenshots in {run['dir']})"
    assert found[0]["ok"], f"{name}: {found[0]['detail']} (screenshots in {run['dir']})"
