"""End-to-end driver, run inside the real Blender GUI (tests/test_e2e.py starts it under Xvfb):

    blender --factory-startup --enable-event-simulate --python tests/e2e/driver.py -- OUT_DIR PROJECT.json

It opens the project and works through the add-on the way a person would: simulated mouse and keyboard events for the
grab, the sculpt stroke and the handle drag; slider values set through the panel's properties (their update callbacks
run as for a slider drag: there is no API for a button's screen position). Rebuilds happen through the add-on's own
timer and depsgraph handler, never by calling it directly. Every check and a screenshot per step go to OUT_DIR
(``results.json``, ``NN-step.png``); Blender quits at the end, also on an error.

The stepping follows Blender's own UI tests (tests/python/ui_simulate/modules/easy_keys.py): one event per step and a
few timer ticks between steps so the event loop handles each one; the splash screen is turned off first, as it takes
all keyboard input.
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportMissingTypeStubs=false

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import blender_env  # pyright: ignore[reportMissingImports]

blender_env.setup()
from proctune import config  # noqa: E402
from proctune.blender import addon, scene  # noqa: E402

MAGICK = config.load().magick

TICKS = 4
#: Longest wait for the add-on's own rebuild after an action (s)
SETTLE_S = 5.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Results:
    checks: list[Check] = field(default_factory=list[Check])
    error: str = ""
    screenshots: list[str] = field(default_factory=list[str])

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, bool(ok), detail))


ARGS = sys.argv[sys.argv.index("--") + 1 :]
OUT, PROJECT = Path(ARGS[0]), Path(ARGS[1])
R = Results()


# ------------------------------------------------------------------------------------------------- helpers


def window() -> bpy.types.Window:
    return bpy.context.window_manager.windows[0]


def view3d() -> tuple[bpy.types.Area, bpy.types.Region, bpy.types.RegionView3D]:
    area = next(a for a in window().screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    return area, region, cast(bpy.types.RegionView3D, cast(bpy.types.SpaceView3D, area.spaces[0]).region_3d)


def to_window(co: Any) -> tuple[int, int]:
    """A world point to window pixel coordinates."""
    _, region, rv3d = view3d()
    p = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(co))
    if p is None:
        raise RuntimeError(f"{tuple(co)} is behind the view")
    return region.x + round(p.x), region.y + round(p.y)


def event(kind: str, value: str, xy: tuple[int, int]) -> None:
    window().event_simulate(
        type=cast(Any, kind), value=cast(Any, value), x=xy[0], y=xy[1]
    )  # literal enums in the stubs


def key(kind: str, xy: tuple[int, int]) -> Iterator[None]:
    event(kind, "PRESS", xy)
    yield
    event(kind, "RELEASE", xy)
    yield


def drag(path: list[tuple[int, int]], button: str = "LEFTMOUSE") -> Iterator[None]:
    """Press at the first point, move through the rest, release at the last."""
    event("MOUSEMOVE", "NOTHING", path[0])
    yield
    event(button, "PRESS", path[0])
    yield
    for xy in path[1:]:
        event("MOUSEMOVE", "NOTHING", xy)
        yield
    event(button, "RELEASE", path[-1])
    yield


def move_to(xy: tuple[int, int]) -> Iterator[None]:
    event("MOUSEMOVE", "NOTHING", xy)
    yield


def settle() -> Iterator[None]:
    """Wait for the add-on's own rebuild: its timer pending, then done, and no modal operator running."""
    t = time.time()
    yield
    while (addon.STATE.pending or len(window().modal_operators) > 0) and time.time() - t < SETTLE_S:
        yield


def screenshot(name: str) -> None:
    """The X display as it is on screen. Blender's own screenshot operator reads back black under Xvfb's software
    OpenGL (.upstream-issues/blender-screenshot-black-under-xvfb-*), so the frame is taken from the X server."""
    path = OUT / f"{len(R.screenshots):02d}-{name}.png"
    # ImageMagick 7 is `magick import`; ImageMagick 6 (Ubuntu) has `import` on its own (PROCTUNE_MAGICK=import)
    cmd = [MAGICK, "import"] if Path(MAGICK).name.startswith("magick") else [MAGICK]
    subprocess.run([*cmd, "-window", "root", str(path)], check=True, capture_output=True)
    R.screenshots.append(path.name)


def show_only(*layers: str) -> None:
    for c in scene.root().children:
        layer = c.name.removeprefix(f"{scene.ROOT}: ")
        c.hide_viewport = bool(layers) and layer not in layers


def verts(name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    return scene.world_verts(bpy.data.objects[name])


def override() -> Any:
    area, region, _ = view3d()
    return bpy.context.temp_override(window=window(), area=area, region=region)


def select_only(name: str) -> None:
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob = bpy.data.objects[name]
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


# ------------------------------------------------------------------------------------------------- scenario


def scenario() -> Iterator[None]:  # noqa: PLR0915 -- one linear story, step by step
    for _ in range(10):
        yield
    addon.open_project(PROJECT)
    s = addon.STATE.session
    R.check(
        "project opens", s is not None and "muscle" in bpy.data.objects, str(sorted(o.name for o in bpy.data.objects))
    )
    assert s is not None
    for name in ("Cube", "Camera", "Light"):
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name])
    with override():
        bpy.ops.view3d.view_axis(type="TOP")
        bpy.ops.view3d.view_all()
    for _ in range(5):
        yield
    screenshot("open")

    # a slider: the property's update callback schedules the add-on's timer, which rebuilds
    bone_top = verts("upper bone")[:, 1].max()
    bpy.context.scene.proctune_params.upper__length = 0.36
    yield from settle()
    grew = float(verts("upper bone")[:, 1].max() - bone_top)
    R.check("slider rebuilds through the timer", abs(grew - 0.06) < 0.003, f"bone tip moved {grew:.4f} m")
    with override():
        bpy.ops.view3d.view_all()
    for _ in range(3):
        yield

    # grab the muscle with G and the mouse
    select_only("muscle")
    centre = verts("muscle").mean(axis=0)
    start = to_window(centre)
    skin_before = verts("skin")
    muscle_before = verts("muscle")
    yield from move_to(start)
    yield from key("G", start)
    R.check("G starts a grab", any(op.bl_idname == "TRANSFORM_OT_translate" for op in window().modal_operators))
    for k in range(1, 9):
        yield from move_to((start[0] + 4 * k, start[1]))
    yield from key("LEFTMOUSE", (start[0] + 32, start[1]))
    yield from settle()
    edit = s.edits.get("muscle")
    R.check("grab is captured as a tweak", edit is not None and not edit.tweak.identity)
    loc = np.array(bpy.data.objects["muscle"].location)
    R.check("the muscle's Location shows the move", float(np.linalg.norm(loc)) > 0.002, f"location {np.round(loc, 4)}")
    moved = float(np.linalg.norm(verts("muscle").mean(axis=0) - muscle_before.mean(axis=0)))
    R.check("muscle stays where it was dropped", moved > 0.002, f"moved {moved * 1000:.1f} mm")
    R.check("skin follows the moved muscle", float(np.abs(verts("skin") - skin_before).max()) > 1e-4)
    screenshot("grabbed")

    # sculpt a real brush stroke along the muscle
    with override():
        bpy.ops.object.mode_set(mode="SCULPT")
    for _ in range(3):
        yield
    m = verts("muscle")
    axis_pts = [
        m[np.argsort(np.abs(m[:, 1] - y))[:40]].mean(axis=0)
        for y in np.linspace(np.percentile(m[:, 1], 30), np.percentile(m[:, 1], 70), 12)
    ]
    stroke = [to_window(p) for p in axis_pts]
    skin_before = verts("skin")
    rebuilds_before = addon.STATE.rebuilds
    yield from drag(stroke)
    yield from settle()
    edit = s.edits.get("muscle")
    R.check("still in sculpt mode", bpy.context.object is not None and bpy.context.object.mode == "SCULPT")
    R.check(
        "stroke is captured as a sculpt",
        edit is not None and edit.displacement is not None,
        f"pending {addon.STATE.pending}, rebuilds {addon.STATE.rebuilds - rebuilds_before}, "
        f"edits seen now {bool(scene.hand_edits(addon.STATE.written))}",
    )
    if edit is not None and edit.displacement is not None:
        R.check(
            "sculpt is not empty",
            float(np.abs(edit.displacement).max()) > 1e-4,
            f"max {np.abs(edit.displacement).max():.4f}",
        )
    R.check("skin follows the stroke", float(np.abs(verts("skin") - skin_before).max()) > 1e-4)
    screenshot("sculpted")

    # a slider while in sculpt mode: the sculpted muscle is rewritten in place and keeps its sculpt; with only the
    # muscle shown, the viewport must change where it is drawn
    show_only("muscle")
    for _ in range(5):
        yield
    screenshot("sculpt-mode-muscle-only")
    shot_before = len(R.screenshots)
    bpy.context.scene.proctune_params.upper__length = 0.42
    yield from settle()
    for _ in range(3):
        yield
    resolved = s.evaluate().elements["muscle"].verts
    err = float(np.abs(verts("muscle") - resolved).max())
    R.check("sculpt-mode mesh matches the rebuilt muscle", err < 1e-5, f"max error {err:.2e} m")
    R.check("sculpt survives the slider", s.edits["muscle"].displacement is not None)
    screenshot("sculpt-mode-slider")
    changed = _changed_pixels(R.screenshots[shot_before - 1], R.screenshots[-1])
    R.check("viewport redraws in sculpt mode", changed > 500, f"{changed} pixels changed in the viewport")
    show_only()
    with override():
        bpy.ops.object.mode_set(mode="OBJECT")
    for _ in range(3):
        yield

    # drag the wrist handle
    wrist = bpy.data.objects["handle: wrist"]
    select_only(wrist.name)
    angle_before = s.base.joint_deg
    at = to_window(wrist.location)
    yield from move_to(at)
    yield from key("G", at)
    for k in range(1, 9):
        yield from move_to((at[0] + 6 * k, at[1] - 3 * k))
    yield from key("LEFTMOUSE", (at[0] + 48, at[1] - 24))
    yield from settle()
    R.check(
        "dragging the wrist bends the joint",
        abs(s.base.joint_deg - angle_before) > 2,
        f"{angle_before:.1f} -> {s.base.joint_deg:.1f} deg",
    )
    R.check(
        "slider shows the dragged value", abs(bpy.context.scene.proctune_params.joint_deg - s.base.joint_deg) < 1e-4
    )
    screenshot("handle")

    yield from turn_elbow_handle(s)
    yield from bone_moves(s)
    yield from skin_scale(s)
    yield from curve_and_timeline(s)

    out = OUT / "saved.json"
    addon.save_project(out)
    R.check("project saves with its sculpt", out.exists() and out.with_suffix(".npz").exists())
    saved = json.loads(out.read_text())
    R.check("project saves the keyframes as a track", "growth.stage" in saved.get("tracks", {}))


def turn_elbow_handle(s: Any) -> Iterator[None]:
    """R on the elbow's handle, turned about 20 degrees by the mouse: the joint bends by about as much, once."""
    h = bpy.data.objects["handle: elbow"]
    select_only(h.name)
    centre = to_window(h.location)
    start = (centre[0] + 60, centre[1])
    before = s.base.joint_deg
    yield from move_to(start)
    yield from key("R", start)
    for k in range(1, 9):
        a = math.radians(-2.5 * k)
        yield from move_to((centre[0] + round(60 * math.cos(a)), centre[1] + round(60 * math.sin(a))))
    yield from key("LEFTMOUSE", start)
    yield from settle()
    bent = s.base.joint_deg - before
    R.check(
        "turning the elbow handle bends the joint once", 12 < bent < 28, f"{before:.1f} -> {s.base.joint_deg:.1f} deg"
    )
    R.check(
        "the elbow handle stands upright again",
        bool(np.allclose([h.rotation_euler.x, h.rotation_euler.y, h.rotation_euler.z], 0.0, atol=1e-5)),
    )


def bone_moves(s: Any) -> Iterator[None]:
    """R turns the lower bone about the view axis (the bend axis in this top view); S Y stretches the upper bone."""
    lower = bpy.data.objects["lower bone"]
    select_only(lower.name)
    joint = to_window(limb_chain(s).joint)
    tip = to_window(limb_chain(s).tip)
    angle_before = s.base.joint_deg
    yield from move_to(tip)
    yield from key("R", tip)
    # rotate the mouse about the joint (the pivot is the object's origin, the joint) by about 20 degrees
    for k in range(1, 9):
        a = math.radians(-2.5 * k)
        dx, dy = tip[0] - joint[0], tip[1] - joint[1]
        yield from move_to(
            (
                joint[0] + round(dx * math.cos(a) - dy * math.sin(a)),
                joint[1] + round(dx * math.sin(a) + dy * math.cos(a)),
            )
        )
    yield from key("LEFTMOUSE", tip)
    yield from settle()
    R.check(
        "turning the lower bone bends the joint",
        abs(s.base.joint_deg - angle_before) > 5,
        f"{angle_before:.1f} -> {s.base.joint_deg:.1f} deg",
    )
    R.check("a bone move is a parameter change, not a tweak", "lower bone" not in s.edits)
    upper = bpy.data.objects["upper bone"]
    select_only(upper.name)
    length_before = s.base.upper.length
    root, elbow = to_window(limb_chain(s).root), to_window(limb_chain(s).joint)
    yield from move_to(elbow)
    yield from key("S", elbow)
    yield from key("Y", elbow)
    for k in range(1, 9):
        yield from move_to((elbow[0] + (elbow[0] - root[0]) * k // 40, elbow[1] + (elbow[1] - root[1]) * k // 40))
    yield from key("LEFTMOUSE", elbow)
    yield from settle()
    R.check(
        "stretching the upper bone lengthens it",
        s.base.upper.length > length_before * 1.05,
        f"{length_before:.3f} -> {s.base.upper.length:.3f} m",
    )
    R.check(
        "the slider follows the stretched bone",
        abs(bpy.context.scene.proctune_params.upper__length - s.base.upper.length) < 1e-4,
    )
    screenshot("bones-moved")


def skin_scale(s: Any) -> Iterator[None]:
    """S on the skin (which has no place of its own): kept as a sculpt, no error, the next rebuild clean."""
    skin = bpy.data.objects["skin"]
    select_only(skin.name)
    centre = to_window(verts("skin").mean(axis=0))
    edge = (centre[0] + 80, centre[1])
    yield from move_to(edge)
    yield from key("S", edge)
    for k in range(1, 6):
        yield from move_to((edge[0] + 4 * k, edge[1]))
    yield from key("LEFTMOUSE", (edge[0] + 20, edge[1]))
    yield from settle()
    R.check("scaling the skin raises no error", addon.STATE.error is None, str(addon.STATE.error))
    R.check(
        "scaling the skin is kept as a sculpt",
        s.edits.get("skin") is not None and s.edits["skin"].displacement is not None,
    )
    bpy.context.scene.proctune_params.joint_deg = s.base.joint_deg + 1.0
    yield from settle()
    R.check("the next rebuild is clean", addon.STATE.error is None, str(addon.STATE.error))


def limb_chain(s: Any) -> Any:
    import limb  # noqa: PLC0415  # the project's model module, importable once the project is open

    return limb.chain(limb.develop(s.effective))


def curve_and_timeline(s: Any) -> Iterator[None]:
    from proctune.blender import curves  # noqa: PLC0415

    # a curve widget edited (through its points: no API gives the widget's screen position); the add-on's poll timer
    # must notice and rebuild by itself
    before = verts("lower fat")
    curves.node("fat__lower__curve").mapping.curves[0].points[1].location = (0.3, 0.8)
    t = time.time()
    while abs(s.base.fat.lower.curve.points[1][0] - 0.3) > 1e-6 and time.time() - t < SETTLE_S:
        yield
    yield from settle()
    R.check("a curve edit rebuilds the model", float(np.abs(verts("lower fat") - before).max()) > 1e-3)
    # the panel with its curve widgets
    area, _, _ = view3d()
    space = cast(bpy.types.SpaceView3D, area.spaces[0])
    space.show_region_ui = True
    for _ in range(4):
        yield
    ui = next(r for r in area.regions if r.type == "UI")
    over = (ui.x + ui.width // 2, ui.y + ui.height // 2)
    yield from move_to(over)
    # show the panel in the tab that is open: simulated Ctrl+wheel switches the sidebar's tab once and then no more
    # (LOG.md), and Python cannot set the tab; the panel's draw code is what this step exercises
    panel = addon.PROCTUNE_PT_panel
    category = panel.bl_category
    bpy.utils.unregister_class(panel)
    panel.bl_category = ui.active_panel_category
    bpy.utils.register_class(panel)
    for _ in range(6):
        yield
    screenshot("panel")
    bpy.utils.unregister_class(panel)
    panel.bl_category = category
    bpy.utils.register_class(panel)

    # keyframes on growth, then the right arrow steps frames like a person would; the frame handler rebuilds
    sc = bpy.context.scene
    sc.frame_set(1)
    sc.proctune_params.growth__stage = 0.0
    sc.keyframe_insert("proctune_params.growth__stage", frame=1)
    sc.proctune_params.growth__stage = 1.0
    sc.keyframe_insert("proctune_params.growth__stage", frame=11)
    sc.frame_set(1)
    yield from settle()
    small = float(verts("upper bone")[:, 1].max())
    centre = to_window(verts("upper bone").mean(axis=0))
    for _ in range(10):
        yield from key("RIGHT_ARROW", centre)
    yield from settle()
    big = float(verts("upper bone")[:, 1].max())
    R.check("arrow keys step the timeline", sc.frame_current == 11, f"frame {sc.frame_current}")
    R.check("the model grows along the timeline", big > small * 2.5, f"bone top {small:.3f} -> {big:.3f} m")
    screenshot("grown")


def _changed_pixels(a: str, b: str) -> int:
    """Pixels of the 3D viewport's main region that differ between two screenshots."""
    ia, ib = bpy.data.images.load(str(OUT / a)), bpy.data.images.load(str(OUT / b))
    w, h = (int(v) for v in cast(Any, ia).size)
    pa = np.array(cast(Any, ia).pixels[:]).reshape(h, w, 4)  # bottom row first, as window coordinates
    pb = np.array(cast(Any, ib).pixels[:]).reshape(h, w, 4)
    bpy.data.images.remove(ia)
    bpy.data.images.remove(ib)
    _, region, _ = view3d()
    box = (slice(region.y, region.y + region.height), slice(region.x, region.x + region.width))
    return int((np.abs(pa[box] - pb[box]).max(axis=2) > 0.05).sum())


# ------------------------------------------------------------------------------------------------- runner


def finish() -> None:
    (OUT / "results.json").write_text(json.dumps(asdict(R), indent=2))
    bpy.ops.wm.quit_blender()


def run(steps: Iterator[None]) -> None:
    count = [0]

    def tick() -> float | None:
        count[0] += 1
        if count[0] % TICKS or bpy.context.window_manager.is_event_handling_break:
            return 0.0
        try:
            next(steps)
        except StopIteration:
            finish()
            return None
        except Exception:  # record any failure and quit, so a run never hangs
            R.error = traceback.format_exc()
            finish()
            return None
        return 0.0

    bpy.app.timers.register(tick, first_interval=0.0, persistent=True)


prefs = bpy.context.preferences
prefs.view.show_splash = False  # the splash takes all keyboard input
prefs.view.smooth_view = 0
prefs.view.use_save_prompt = False
prefs.filepaths.use_auto_save_temporary_files = False
OUT.mkdir(parents=True, exist_ok=True)
addon.register()
run(scenario())
