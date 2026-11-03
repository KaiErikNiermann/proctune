"""Leaks, scaling and errors under load: what proctune itself allocates, registers and writes, in plain Python and in
headless Blender."""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportOptionalMemberAccess=false

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import limb
import numpy as np
import psutil
import pytest

from proctune.core import Element, Frame, Range, Resolved, Session, Stage
from proctune.errors import MissingDependenciesError
from proctune.shapes import revolution

EXAMPLE = Path(limb.__file__).parent
ROOT_USER = hasattr(os, "geteuid") and os.geteuid() == 0


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


# ------------------------------------------------------------------------------------------------- plain Python


def test_rebuilding_does_not_grow_python_memory() -> None:
    s = Session.new(limb.MODEL, "limb:MODEL", (str(EXAMPLE),))
    r = s.evaluate().elements["muscle"]
    s.capture_sculpt("muscle", r.verts + 0.001)
    for _ in range(20):  # warm caches
        s.evaluate()
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    for i in range(300):
        s.set(("upper", "length"), 0.25 + 0.1 * (i % 3))
        s.set(("growth", "stage"), (i % 10) / 10)
        s.evaluate()
    gc.collect()
    grown = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    assert grown < 2 * 1024 * 1024, f"{grown / 1e6:.1f} MB kept after 300 rebuilds"


@dataclass(frozen=True)
class Many:
    count: Annotated[int, Range(1, 2000)] = 10
    size: Annotated[float, Range(0.1, 2.0)] = 1.0


def _tubes(p: Many, _: Mapping[str, Resolved]) -> tuple[Element, ...]:
    local, faces = revolution(np.full(limb.RINGS, 1.0))
    return tuple(
        Element(
            f"tube {i}",
            "bone",
            Frame(np.array([i * 0.1, 0.0, 0.0]), np.eye(3), np.full(3, 0.02 * p.size)),
            local,
            faces,
        )
        for i in range(p.count)
    )


@dataclass(frozen=True)
class ManyModel:
    name: str = "many"
    stages: tuple[Stage[Many], ...] = (Stage("tubes", _tubes),)
    composites: tuple[()] = field(default_factory=tuple)

    def defaults(self) -> Many:
        return Many()

    def handles(self, p: Many) -> tuple[()]:
        return ()


def _per_element_ms(n: int) -> float:
    s = Session.new(ManyModel())
    s.set(("count",), n)
    s.evaluate()
    t = time.perf_counter()
    for _ in range(3):
        s.evaluate()
    return (time.perf_counter() - t) / 3 / n * 1000


def test_rebuild_time_is_linear_in_the_element_count() -> None:
    small, large = _per_element_ms(20), _per_element_ms(400)
    assert large < small * 3, f"{small:.3f} ms per element at 20, {large:.3f} at 400"


# ------------------------------------------------------------------------------------------------- Blender

bpy = pytest.importorskip("bpy")

from proctune.blender import addon, scene  # noqa: E402

COUNTED = ("objects", "meshes", "materials", "node_groups", "collections", "images")


def _counts() -> dict[str, int]:
    return {k: len(getattr(bpy.data, k)) for k in COUNTED}


@pytest.fixture
def project() -> Iterator[None]:
    bpy.ops.wm.read_homefile(use_empty=True)
    addon.register()
    addon.new_project("limb:MODEL", EXAMPLE)
    yield
    addon.unregister()
    addon.STATE.__init__()


def test_rebuilds_leak_no_blender_data_or_memory(project: None) -> None:
    props = bpy.context.scene.proctune_params
    for _ in range(20):
        addon.rebuild()
    counts = _counts()
    gc.collect()
    rss = _rss_mb()
    for i in range(300):
        props.upper__length = 0.25 + 0.1 * (i % 3)
        props.fat__adiposity = (i % 7) / 7
        addon.rebuild()
    gc.collect()
    assert _counts() == counts
    grown = _rss_mb() - rss
    assert grown < 40, f"RSS grew {grown:.0f} MB over 300 rebuilds"


def test_reopening_projects_leaks_no_meshes(project: None, tmp_path: Path) -> None:
    addon.save_project(tmp_path / "p.json")
    addon.open_project(tmp_path / "p.json")
    counts = _counts()
    for _ in range(15):
        addon.open_project(tmp_path / "p.json")
    assert _counts() == counts


def test_register_cycles_leave_no_handlers_or_timers() -> None:
    h = bpy.app.handlers
    before = (len(h.depsgraph_update_post), len(h.load_post), len(h.frame_change_post))
    for _ in range(10):
        addon.register()
        addon.unregister()
    assert (len(h.depsgraph_update_post), len(h.load_post), len(h.frame_change_post)) == before
    assert not bpy.app.timers.is_registered(addon._poll_curves)  # pyright: ignore[reportPrivateUsage]


def test_a_burst_of_slider_changes_schedules_one_rebuild(project: None) -> None:
    addon.STATE.pending = False
    props = bpy.context.scene.proctune_params
    for i in range(1000):
        props.joint_deg = float(i % 150)
    assert addon.STATE.pending
    assert bpy.app.timers.is_registered(addon._tick)  # pyright: ignore[reportPrivateUsage]
    bpy.app.timers.unregister(addon._tick)  # pyright: ignore[reportPrivateUsage]
    addon.STATE.pending = False


def test_a_crashing_model_shows_in_the_panel_and_keeps_the_mesh(project: None, monkeypatch: pytest.MonkeyPatch) -> None:
    before = scene.read_verts(bpy.data.objects["skin"])

    def broken(*_: object) -> tuple[()]:
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(limb, "fat", broken)
    addon.STATE.pending = True
    addon.STATE.changed = True
    addon._tick()  # pyright: ignore[reportPrivateUsage]  # what the timer runs; must not raise
    e = addon.STATE.error
    assert e is not None
    assert "stage 'fat' failed" in e.message
    np.testing.assert_array_equal(scene.read_verts(bpy.data.objects["skin"]), before)
    monkeypatch.undo()
    addon.rebuild(force=True)
    assert addon.STATE.error is None  # cleared by the next good rebuild


@pytest.mark.skipif(
    ROOT_USER or sys.platform == "win32", reason="root ignores permissions; Windows ignores read-only on folders"
)
def test_saving_to_a_read_only_folder_is_reported(project: None, tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        # an operator's error report becomes a RuntimeError when it is called from Python (in the UI: a red
        # message in the status bar), carrying the message and the remedy
        with pytest.raises(RuntimeError, match=r"permission denied.*save to a folder you can write to"):
            bpy.ops.proctune.save(filepath=str(ro / "p.json"))
    finally:
        ro.chmod(0o755)
    assert addon.STATE.error is not None
    assert "permission denied" in addon.STATE.error.message
    assert list(ro.iterdir()) == []


def test_missing_packages_offer_the_install_button(monkeypatch: pytest.MonkeyPatch) -> None:
    from proctune import deps  # noqa: PLC0415

    def only_scipy(reqs: Iterable[deps.Requirement]) -> tuple[deps.Requirement, ...]:
        return tuple(r for r in reqs if r.module == "scipy")

    monkeypatch.setattr(deps, "missing", only_scipy)
    bpy.ops.wm.read_homefile(use_empty=True)
    addon.register()
    try:
        e = addon.STATE.error
        assert isinstance(e, MissingDependenciesError)
        assert e.packages == ("scipy",)
    finally:
        addon.unregister()
        addon.STATE.__init__()


def test_installing_reopens_the_project_that_needed_the_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A project whose model imports a missing package shows the install button; installing reopens it."""
    from proctune import deps  # noqa: PLC0415

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "needs_pkg_xyz.py").write_text("import pkg_not_there_xyz\n")
    project = tmp_path / "p.json"
    project.write_text('{"format": 2, "model": "needs_pkg_xyz:MODEL", "import_paths": ["m"]}')
    bpy.ops.wm.read_homefile(use_empty=True)
    addon.register()
    try:
        addon.open_reporting(project)
        e = addon.STATE.error
        assert isinstance(e, MissingDependenciesError)
        assert e.packages == ("pkg_not_there_xyz",)
        assert addon.STATE.retry == project

        def fake_install(names: Iterable[str], target: Path | None = None) -> deps.InstallResult:
            # the package appears: the model module now imports, and defines the limb's model
            (model_dir / "pkg_not_there_xyz.py").write_text("")
            code = [
                "import sys",
                f"sys.path.insert(0, {str(EXAMPLE)!r})",
                "import pkg_not_there_xyz",
                "from limb import MODEL",
            ]
            (model_dir / "needs_pkg_xyz.py").write_text("\n".join(code) + "\n")
            return deps.InstallResult(tuple(names), (), tmp_path)

        monkeypatch.setattr(deps, "install_into_blender", fake_install)
        bpy.ops.proctune.install(packages="pkg_not_there_xyz")
        assert addon.STATE.error is None
        assert addon.STATE.session is not None
        assert "muscle" in bpy.data.objects
    finally:
        addon.unregister()
        addon.STATE.__init__()


def test_our_own_writes_do_not_trigger_rebuilds(project: None) -> None:
    """A rebuild writes every mesh, and Blender reports those writes as updates; rebuilding on each of those rebuilt
    without end (with the 0.4 s synth-hand model the whole editor lagged, panning included). The update may schedule
    a check, but the check must find nothing to do and build nothing."""
    bpy.context.scene.proctune_params.joint_deg = 20.0
    addon.rebuild()
    count = addon.STATE.rebuilds
    for _ in range(5):
        bpy.context.view_layer.update()  # Blender reports the writes
        if addon.STATE.pending:
            addon._tick()  # pyright: ignore[reportPrivateUsage]  # what the timer runs
    assert addon.STATE.rebuilds == count, "our own write led to another rebuild"
    bpy.data.objects["muscle"].location = (0.0, 0.01, 0.0)
    bpy.context.view_layer.update()
    assert addon.STATE.pending  # a real hand edit schedules
    addon._tick()  # pyright: ignore[reportPrivateUsage]
    assert addon.STATE.rebuilds == count + 1  # and builds
