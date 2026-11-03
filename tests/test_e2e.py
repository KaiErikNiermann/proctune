"""End to end: the add-on in the real Blender GUI on a virtual display, driven by simulated input
(tests/e2e/driver.py). Linux only. Needs ``blender``, ``xvfb-run`` (Arch: xorg-server-xvfb) and ``magick``
(imagemagick, for screenshots), each settable in the proctune config; skipped without them. Set
PROCTUNE_E2E_KEEP=DIR to keep the screenshots."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from proctune import config

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 240

CFG = config.load()
MISSING = [t for t in (CFG.blender, CFG.xvfb_run, CFG.magick) if shutil.which(t) is None]
UNAVAILABLE = (
    "Linux only"
    if sys.platform != "linux"
    else f"missing {', '.join(MISSING)} (see proctune.config)"
    if MISSING
    else ""
)
if UNAVAILABLE and os.environ.get("PROCTUNE_E2E_REQUIRED"):
    # in CI a skip would pass silently: fail instead
    raise RuntimeError(f"end-to-end tests required but cannot run: {UNAVAILABLE}")
pytestmark = pytest.mark.skipif(bool(UNAVAILABLE), reason=UNAVAILABLE or "available")


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = Path(os.environ.get("PROCTUNE_E2E_KEEP") or tmp_path_factory.mktemp("e2e"))
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, ROOT / "scripts" / "blender_site.py"], check=True, capture_output=True)
    project = out / "limb.json"
    shutil.copy(ROOT / "examples" / "limb" / "limb.json", project)
    doc = json.loads(project.read_text())
    doc["import_paths"] = [str(ROOT / "examples" / "limb")]
    project.write_text(json.dumps(doc))
    # Blender prefers Wayland when it is there; without WAYLAND_DISPLAY it can only use the virtual X display
    env = {k: v for k, v in os.environ.items() if k != "WAYLAND_DISPLAY"} | {"DBUS_SESSION_BUS_ADDRESS": "/dev/null"}
    cmd = [
        CFG.xvfb_run, "-a", "-s", "-screen 0 1600x1000x24",
        CFG.blender, "--factory-startup", "--enable-event-simulate",
        "--python", str(ROOT / "tests" / "e2e" / "driver.py"), "--", str(out), str(project),
    ]  # fmt: skip
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    results = out / "results.json"
    if not results.exists():
        pytest.fail(f"no results (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    data: dict[str, Any] = json.loads(results.read_text())
    data["dir"] = str(out)
    data["stderr"] = proc.stderr
    return data


def test_driver_ran_to_the_end(run: dict[str, Any]) -> None:
    assert not run["error"], run["error"]


def test_blender_reported_no_error_from_proctune(run: dict[str, Any]) -> None:
    """Exceptions in handlers, timers and panel drawing do not stop Blender; they only print a traceback."""
    blocks = [b for b in str(run["stderr"]).split("Traceback (most recent call last):")[1:] if "proctune" in b]
    assert not blocks, blocks[0][:3000]


CHECKS = [
    "project opens",
    "slider rebuilds through the timer",
    "G starts a grab",
    "grab is captured as a tweak",
    "the muscle's Location shows the move",
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
    "turning the lower bone bends the joint",
    "a bone move is a parameter change, not a tweak",
    "stretching the upper bone lengthens it",
    "the slider follows the stretched bone",
    "scaling the skin raises no error",
    "scaling the skin is kept as a sculpt",
    "the next rebuild is clean",
    "a curve edit rebuilds the model",
    "arrow keys step the timeline",
    "the model grows along the timeline",
    "project saves with its sculpt",
    "project saves the keyframes as a track",
]


@pytest.mark.parametrize("name", CHECKS)
def test_check(run: dict[str, Any], name: str) -> None:
    found = [c for c in run["checks"] if c["name"] == name]
    assert found, f"{name!r} did not run (screenshots in {run['dir']})"
    assert found[0]["ok"], f"{name}: {found[0]['detail']} (screenshots in {run['dir']})"
