"""The add-on in headless Blender (the bpy module): sliders, hand edits and handles through the scene."""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportOptionalMemberAccess=false

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from proctune.blender import addon, scene  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "limb"


@pytest.fixture
def project() -> Iterator[None]:
    bpy.ops.wm.read_homefile(use_empty=True)
    addon.register()
    addon.new_project("limb:MODEL", EXAMPLE)
    yield
    addon.unregister()
    addon.STATE.__init__()


def _verts(name: str) -> np.ndarray:  # pyright: ignore[reportMissingTypeArgument]
    """World space (a movable element's mesh is in its frame)."""
    return scene.world_verts(bpy.data.objects[name])


def _sculpt(name: str, offset: tuple[float, float, float], rows: slice) -> None:
    """Move some vertices in the object's own space, as a sculpt stroke does."""
    obj = bpy.data.objects[name]
    v = scene.read_verts(obj)
    v[rows] += np.array(offset)
    obj.data.vertices.foreach_set("co", v.astype(np.float32).ravel())
    obj.data.update()


def test_objects_per_element_and_handles(project: None) -> None:
    names = {o.name for o in bpy.data.objects}
    assert {"upper bone", "lower bone", "muscle", "skin", "handle: elbow", "handle: wrist"} <= names
    assert bpy.data.collections["proctune: muscle"].objects["muscle"]


def test_slider_changes_the_model(project: None) -> None:
    before = _verts("upper bone")[:, 1].max()
    bpy.context.scene.proctune_params.upper__length = 0.45
    assert addon.STATE.session is not None and addon.STATE.session.base.upper.length == pytest.approx(0.45)
    addon.rebuild()
    assert _verts("upper bone")[:, 1].max() - before == pytest.approx(0.15, abs=0.005)


def test_composite_slider(project: None) -> None:
    bpy.context.scene.proctune_params.composite__flex = 1.0
    addon.rebuild()
    assert addon.STATE.session is not None and addon.STATE.session.effective.joint_deg == pytest.approx(135.0)


def test_sculpt_is_captured_and_survives_a_slider(project: None) -> None:
    _sculpt("muscle", (0.004, 0.0, 0.0), slice(100, 140))
    skin_before = _verts("skin")
    addon.rebuild()
    s = addon.STATE.session
    assert s is not None and s.edits["muscle"].displacement is not None
    assert np.abs(_verts("skin") - skin_before).max() > 1e-4  # the skin followed
    bpy.context.scene.proctune_params.upper__length = 0.4
    addon.rebuild()
    assert s.edits["muscle"].displacement is not None  # not lost, and not re-captured as a new sculpt
    np.testing.assert_allclose(_verts("muscle"), s.evaluate().elements["muscle"].verts, atol=1e-6)


def test_object_move_becomes_a_tweak(project: None) -> None:
    obj = bpy.data.objects["muscle"]
    before = _verts("muscle")
    obj.location = (0.0, 0.02, 0.0)  # 2 cm along the muscle, in its frame
    addon.rebuild()
    assert tuple(obj.location) == pytest.approx((0.0, 0.02, 0.0))  # the move stays visible, relative to the default
    axis = np.array(obj.parent.matrix_world)[:3, 1]
    np.testing.assert_allclose(_verts("muscle"), before + 0.02 * axis, atol=1e-6)
    s = addon.STATE.session
    assert s is not None and not s.edits["muscle"].tweak.identity


def test_dragging_a_handle_sets_parameters(project: None) -> None:
    h = bpy.data.objects["handle: elbow"]
    h.location = (0.0, 0.38, 0.0)
    bpy.context.view_layer.update()
    addon.rebuild()
    s = addon.STATE.session
    assert s is not None and s.base.upper.length == pytest.approx(0.38)
    assert bpy.context.scene.proctune_params.upper__length == pytest.approx(0.38)


def test_clear_removes_the_edit(project: None) -> None:
    obj = bpy.data.objects["muscle"]
    obj.location = (0.0, 0.02, 0.0)
    addon.rebuild()
    bpy.ops.proctune.clear(element="muscle")
    s = addon.STATE.session
    assert s is not None and "muscle" not in s.edits
    addon.rebuild()
    assert "muscle" not in s.edits


def test_save_and_open(project: None, tmp_path: Path) -> None:
    _sculpt("muscle", (0.004, 0.0, 0.0), slice(100, 140))
    bpy.context.scene.proctune_params.joint_deg = 40.0
    addon.save_project(tmp_path / "limb.json")
    saved = _verts("skin")
    addon.open_project(tmp_path / "limb.json")
    np.testing.assert_allclose(_verts("skin"), saved, atol=1e-6)
    assert bpy.context.scene.proctune_params.joint_deg == pytest.approx(40.0)


def test_a_hand_move_schedules_a_rebuild(project: None) -> None:
    """The depsgraph handler notices an element moved by hand without any button."""
    addon.STATE.pending = False
    bpy.data.objects["muscle"].location = (0.0, 0.01, 0.0)
    bpy.context.view_layer.update()
    assert addon.STATE.pending
    addon.STATE.pending = False
    bpy.app.timers.unregister(addon._tick)  # pyright: ignore[reportPrivateUsage]


def test_loading_a_blend_does_not_run_its_project(project: None, tmp_path: Path) -> None:
    """A .blend naming a project only offers it: opening the project imports and runs model code."""
    addon.save_project(tmp_path / "limb.json")
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / "scene.blend"))
    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / "scene.blend"))
    assert addon.STATE.session is None
    assert addon.STATE.stored == tmp_path / "limb.json"
    bpy.ops.proctune.open_stored("EXEC_DEFAULT")  # what the confirmation dialog's button does
    assert addon.STATE.session is not None


def test_a_curve_edited_in_its_widget_reaches_the_model(project: None) -> None:
    from proctune.blender import curves  # noqa: PLC0415

    s = addon.STATE.session
    assert s is not None
    before = scene.read_verts(bpy.data.objects["lower fat"])
    pts = curves.node("fat__lower__curve").mapping.curves[0].points
    pts[1].location = (0.3, 0.8)  # the lower deposit now fills in early
    addon._poll_curves()  # pyright: ignore[reportPrivateUsage]  # what the timer does every 0.2 s
    assert s.base.fat.lower.curve.points[1] == pytest.approx((0.3, 0.8))
    addon.rebuild()
    assert np.abs(scene.read_verts(bpy.data.objects["lower fat"]) - before).max() > 1e-3


def test_the_widget_draws_close_to_the_evaluated_curve(project: None) -> None:
    """The widget's auto-clamped handles and PCHIP are different splines through the same points."""
    from proctune.blender import curves  # noqa: PLC0415

    s = addon.STATE.session
    assert s is not None
    c = s.base.growth.curve
    m = curves.node("growth__curve").mapping
    m.initialize()
    xs = np.linspace(0.0, 1.0, 41)
    drawn = np.array([m.evaluate(m.curves[0], float(x)) for x in xs])
    assert np.abs(drawn - c.evaluate(xs)).max() < 0.03


def test_keyframed_sliders_follow_the_timeline(project: None) -> None:
    sc = bpy.context.scene
    sc.proctune_params.growth__stage = 0.0
    sc.keyframe_insert("proctune_params.growth__stage", frame=1)
    sc.proctune_params.growth__stage = 1.0
    sc.keyframe_insert("proctune_params.growth__stage", frame=21)
    sc.frame_set(1)
    small = _verts("upper bone")[:, 1].max()
    sc.frame_set(21)
    big = _verts("upper bone")[:, 1].max()
    assert small == pytest.approx(0.30 * 0.3, abs=2e-3)
    assert big == pytest.approx(0.30, abs=2e-3)


def test_saving_bakes_keyframes_into_tracks(project: None, tmp_path: Path) -> None:
    from proctune.core import Session  # noqa: PLC0415

    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, 21
    sc.proctune_params.growth__stage = 0.0
    sc.keyframe_insert("proctune_params.growth__stage", frame=1)
    sc.proctune_params.growth__stage = 1.0
    sc.keyframe_insert("proctune_params.growth__stage", frame=21)
    sc.frame_set(11)
    in_blender = _verts("upper bone")
    addon.save_project(tmp_path / "anim.json")
    headless = Session.load(tmp_path / "anim.json").at_frame(11).evaluate().elements["upper bone"].verts
    np.testing.assert_allclose(headless, in_blender, atol=1e-5)


def test_scaling_a_bone_object_changes_its_length(project: None) -> None:
    obj = bpy.data.objects["upper bone"]
    obj.scale = (1.0, 1.2, 1.0)
    addon.rebuild()
    s = addon.STATE.session
    assert s is not None
    assert s.base.upper.length == pytest.approx(0.36)
    assert bpy.context.scene.proctune_params.upper__length == pytest.approx(0.36)
    assert "upper bone" not in s.edits
    assert np.allclose(np.array(obj.matrix_basis), np.eye(4))


def test_a_scale_shows_relative_to_the_default_and_survives(project: None) -> None:
    """Scaling the muscle 1.2 along itself keeps Scale at 1.2 in the object's fields, through later rebuilds."""
    obj = bpy.data.objects["muscle"]
    obj.scale = (1.0, 1.2, 1.0)
    addon.rebuild()
    bpy.context.scene.proctune_params.upper__length = 0.4
    addon.rebuild()
    assert tuple(obj.scale) == pytest.approx((1.0, 1.2, 1.0))
    s = addon.STATE.session
    assert s is not None
    assert s.edits["muscle"].tweak.scale == pytest.approx([1.0, 1.2, 1.0])


def test_moving_the_skin_becomes_a_sculpt_not_an_error_loop(project: None) -> None:
    """The skin follows the limb and has no place of its own: scaling it is kept as a sculpt, reported nowhere as
    an error, and does not fail the next rebuilds."""
    skin = bpy.data.objects["skin"]
    before = _verts("skin")
    skin.scale = (1.1, 1.1, 1.1)
    addon.rebuild()
    assert addon.STATE.error is None
    s = addon.STATE.session
    assert s is not None
    assert s.edits["skin"].displacement is not None
    assert np.allclose(np.array(skin.matrix_basis), np.eye(4))
    np.testing.assert_allclose(_verts("skin"), before * 1.1, atol=1e-5)
    addon.rebuild()
    assert addon.STATE.error is None


def test_an_edit_that_cannot_be_kept_is_undone_once(project: None) -> None:
    """A sculpt that changed the vertex count cannot be kept: it is reported, the mesh is put back, and the next
    rebuild is clean (no repeated failure)."""
    from proctune.errors import EditError  # noqa: PLC0415

    obj = bpy.data.objects["muscle"]
    me = obj.data
    import bmesh  # noqa: PLC0415

    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:10], cuts=1)
    bm.to_mesh(me)
    bm.free()
    with pytest.raises(EditError, match="vertices"):
        addon.rebuild()
    s = addon.STATE.session
    assert s is not None
    assert len(me.vertices) == len(s.evaluate().elements["muscle"].verts)
    addon.rebuild()  # clean


def test_reset_puts_one_parameter_back_to_its_default(project: None) -> None:
    props = bpy.context.scene.proctune_params
    props.upper__length = 0.36
    props.joint_deg = 40.0
    addon.rebuild()
    bpy.ops.proctune.reset_param(key="upper__length")
    s = addon.STATE.session
    assert s is not None
    assert s.base.upper.length == pytest.approx(0.30)
    assert s.base.joint_deg == pytest.approx(40.0)  # the others stay
    assert props.upper__length == pytest.approx(0.30)


def test_rotating_the_elbow_handle_bends_the_joint(project: None) -> None:
    import math  # noqa: PLC0415

    h = bpy.data.objects["handle: elbow"]
    h.rotation_euler = (0.0, 0.0, math.radians(-20.0))
    addon.rebuild()
    s = addon.STATE.session
    assert s is not None
    assert s.base.joint_deg == pytest.approx(35.0, abs=1e-4)
    assert tuple(h.rotation_euler) == pytest.approx((0.0, 0.0, 0.0))  # upright again, the slider shows the result
    assert bpy.context.scene.proctune_params.joint_deg == pytest.approx(35.0, abs=1e-4)


def test_handles_without_a_turn_are_locked(project: None) -> None:
    w = bpy.data.objects["handle: wrist"]
    assert tuple(w.lock_rotation) == (True, True, True)
    assert tuple(bpy.data.objects["handle: elbow"].lock_rotation) == (False, False, False)


def test_a_turn_in_progress_is_not_applied_twice(project: None) -> None:
    """Rebuilds during a rotation (the handle being dragged is not reset) apply only what was turned since."""
    import math  # noqa: PLC0415

    h = bpy.data.objects["handle: elbow"]
    addon.STATE.dragged = frozenset({"elbow"})
    h.rotation_euler = (0.0, 0.0, math.radians(-10.0))
    addon.rebuild()
    addon.STATE.dragged = frozenset({"elbow"})
    addon.rebuild()  # same rotation, still in progress: nothing more
    addon.STATE.dragged = frozenset({"elbow"})
    h.rotation_euler = (0.0, 0.0, math.radians(-25.0))
    addon.rebuild()
    s = addon.STATE.session
    assert s is not None
    assert s.base.joint_deg == pytest.approx(40.0, abs=1e-4)
