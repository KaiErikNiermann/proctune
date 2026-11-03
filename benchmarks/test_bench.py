"""Benchmarks of the paths a rebuild takes: poetry run pytest benchmarks (not part of the default test run)."""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import limb
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from proctune.core import Curve, Session

EXAMPLE = Path(limb.__file__).parent


def _sculpted() -> Session[limb.Limb]:
    s = Session.new(limb.MODEL, "limb:MODEL", (str(EXAMPLE),))
    for name, r in s.evaluate().elements.items():
        s.capture_sculpt(name, r.verts + 0.0005)
    return s


def test_evaluate(benchmark: BenchmarkFixture) -> None:
    s = Session.new(limb.MODEL, "limb:MODEL", (str(EXAMPLE),))
    benchmark(s.evaluate)


def test_evaluate_with_every_element_sculpted(benchmark: BenchmarkFixture) -> None:
    benchmark(_sculpted().evaluate)


def test_capture_sculpt(benchmark: BenchmarkFixture) -> None:
    s = _sculpted()
    world = s.evaluate().elements["skin"].verts + 0.001
    benchmark(s.capture_sculpt, "skin", world)


def test_skin_stage(benchmark: BenchmarkFixture) -> None:
    s = _sculpted()
    r = s.evaluate()
    upstream = {n: e for n, e in r.elements.items() if n != "skin"}
    benchmark(limb.skin, s.effective, upstream)


def test_wrap_skin_stage(benchmark: BenchmarkFixture) -> None:
    s = _sculpted()
    r = s.evaluate()
    upstream = {n: e for n, e in r.elements.items() if n not in ("skin", "wrap skin")}
    benchmark(limb.wrap_skin, s.effective, upstream)


def test_curve_100k_points(benchmark: BenchmarkFixture) -> None:
    c = Curve.through((0.0, 0.0), (0.2, 0.7), (0.5, 0.8), (1.0, 1.0))
    x = np.linspace(0.0, 1.0, 100_000)
    benchmark(c.evaluate, x)


def test_save_and_load_with_sculpts(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    s = _sculpted()

    def round_trip() -> None:
        s.save(tmp_path / "p.json")
        Session.load(tmp_path / "p.json")

    benchmark(round_trip)


bpy = pytest.importorskip("bpy")

from proctune.blender import addon, scene  # noqa: E402


@pytest.fixture
def project() -> Iterator[None]:
    bpy.ops.wm.read_homefile(use_empty=True)
    addon.register()
    addon.new_project("limb:MODEL", EXAMPLE)
    yield
    addon.unregister()
    addon.STATE.__init__()


def test_blender_rebuild(benchmark: BenchmarkFixture, project: None) -> None:
    """Capture hand edits, evaluate, write every mesh, place handles, sync sliders."""
    benchmark(addon.rebuild)


def test_blender_hand_edit_detection(benchmark: BenchmarkFixture, project: None) -> None:
    benchmark(scene.hand_edits, addon.STATE.written)
