"""The core against the limb example: parameters, composites, handles, and hand edits surviving rebuilds."""

from __future__ import annotations

import math
from pathlib import Path

import limb
import numpy as np
import pytest
from numpy.typing import NDArray

from proctune.core import Session, specs
from proctune.core import params as P
from proctune.core.element import F64, Edit


def _session() -> Session[limb.Limb]:
    return Session.new(limb.MODEL, "limb:MODEL", (str(Path(limb.__file__).parent),))


def _muscle_radial(s: Session[limb.Limb], verts: F64) -> F64:
    """Each muscle vertex's distance from the muscle's line (world, m)."""
    start, end = limb.muscle_ends(s.effective)
    d = (end - start) / np.linalg.norm(end - start)
    rel = verts - start
    return np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)


def _bump(s: Session[limb.Limb], mm: float = 5.0) -> tuple[F64, NDArray[np.bool_]]:
    """Push the muscle's vertices on the belly's +x side outwards by `mm`; returns the sculpted vertices and mask."""
    r = s.evaluate().elements["muscle"]
    local = r.element.local
    mask = (np.abs(local[:, 1] - 0.45) < 0.08) & (local[:, 0] > 0.5)
    world = r.verts.copy()
    world[mask] += r.element.frame.axes[:, 0] * mm / 1000
    return world, mask


def test_specs_cover_every_value_with_its_range() -> None:
    sp = {s.key: s for s in specs(limb.Limb())}
    assert sp["upper__length"].default == 0.30
    assert sp["upper__length"].range is not None and sp["upper__length"].range.unit == "m"
    assert sp["joint_deg"].kind == "float"
    assert len(sp) == 3 + 3 + 1 + 7 + 3


def test_replace_and_round_trip() -> None:
    p = P.replace(limb.Limb(), ("lower", "radius"), 0.02)
    assert P.get(p, ("lower", "radius")) == 0.02
    assert P.get(p, ("upper", "radius")) == limb.Limb().upper.radius
    assert P.from_dict(limb.Limb, P.to_dict(p)) == p
    with pytest.raises(KeyError):
        P.from_dict(limb.Limb, {"nonsense": 1})


def test_composites_apply_on_top_of_the_base() -> None:
    s = _session()
    s.set_composite("flex", 0.5)
    assert s.effective.joint_deg == pytest.approx(limb.Limb().joint_deg + 60)
    assert s.base.joint_deg == limb.Limb().joint_deg
    s.set_composite("size", 1.0)
    assert s.effective.upper.length == pytest.approx(2 * limb.Limb().upper.length)


def test_topology_is_fixed_across_parameters() -> None:
    s = _session()
    before = {n: r.verts.shape for n, r in s.evaluate().elements.items()}
    s.set(("upper", "length"), 0.5)
    s.set_composite("flex", 1.0)
    assert {n: r.verts.shape for n, r in s.evaluate().elements.items()} == before


def test_sculpt_round_trips_exactly() -> None:
    s = _session()
    world, _ = _bump(s)
    s.capture_sculpt("muscle", world)
    np.testing.assert_allclose(s.evaluate().elements["muscle"].verts, world, atol=1e-12)


def test_sculpt_survives_a_longer_bone() -> None:
    """The bump stays half way along the muscle and keeps its height when the bone gets longer."""
    s = _session()
    world, mask = _bump(s, 5.0)
    plain = _muscle_radial(s, s.evaluate().elements["muscle"].verts)
    s.capture_sculpt("muscle", world)
    s.set(("upper", "length"), 0.45)
    r = s.evaluate().elements["muscle"]
    s2 = _session()
    s2.set(("upper", "length"), 0.45)
    unsculpted = _muscle_radial(s2, s2.evaluate().elements["muscle"].verts)
    grown = _muscle_radial(s, r.verts) - unsculpted
    assert np.all(np.abs(grown[~mask]) < 1e-12)
    # radial scale (the muscle's girth) did not change, so the bump is the same height (up to its slight tilt)
    assert grown[mask].max() == pytest.approx(0.005, rel=0.05)
    assert np.allclose(plain, _muscle_radial(_session(), _session().evaluate().elements["muscle"].verts))


def test_tweak_is_the_world_move_and_composes() -> None:
    s = _session()
    before = s.evaluate().elements["muscle"].verts
    a = math.radians(20)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    m[:3, 3] = [0.01, -0.02, 0.005]
    s.capture_transform("muscle", m)
    after = s.evaluate().elements["muscle"].verts
    np.testing.assert_allclose(after, before @ m[:3, :3].T + m[:3, 3], atol=1e-12)
    s.capture_transform("muscle", m)
    m2 = m @ m
    np.testing.assert_allclose(s.evaluate().elements["muscle"].verts, before @ m2[:3, :3].T + m2[:3, 3], atol=1e-12)


def test_sculpt_moves_with_the_tweak() -> None:
    s = _session()
    world, _ = _bump(s)
    s.capture_sculpt("muscle", world)
    m = np.eye(4)
    m[:3, 3] = [0.0, 0.03, 0.0]
    s.capture_transform("muscle", m)
    np.testing.assert_allclose(s.evaluate().elements["muscle"].verts, world + m[:3, 3], atol=1e-12)


def test_skin_follows_a_sculpted_muscle() -> None:
    s = _session()
    plain = s.evaluate().elements["skin"].verts
    world, _ = _bump(s, 10.0)
    s.capture_sculpt("muscle", world)
    sculpted = s.evaluate().elements["skin"].verts
    pos, _ = limb.centreline(s.effective)
    reach = np.linalg.norm(sculpted - plain, axis=1)
    assert reach.max() > 0.003
    assert np.isfinite(pos).all()


def test_skin_sculpt_follows_the_bend() -> None:
    """A skin sculpt is stored along and around the limb: bending the joint keeps its size."""
    s = _session()
    r = s.evaluate().elements["skin"]
    world = r.verts.copy()
    idx = 70 * limb.SKIN_SECTIONS + 5  # a vertex on the lower part
    world[idx] += r.element.vertex_axes[idx][:, 0] * 0.004  # pyright: ignore[reportOptionalSubscript]
    s.capture_sculpt("skin", world)
    s.set(("joint_deg",), 80.0)
    bent = s.evaluate().elements["skin"]
    s2 = _session()
    s2.set(("joint_deg",), 80.0)
    plain = s2.evaluate().elements["skin"]
    moved = np.linalg.norm(bent.verts - plain.verts, axis=1)
    assert moved.argmax() == idx
    assert moved[idx] == pytest.approx(0.004, rel=0.25)  # scaled by the local radius, which bending changes a little


def test_stale_sculpt_is_reported_not_applied() -> None:
    s = _session()
    s.edits["muscle"] = Edit(displacement=np.zeros((3, 3)))
    r = s.evaluate()
    assert r.stale == ("muscle",)


def test_handles_set_parameters() -> None:
    s = _session()
    joint = limb.chain(s.base).joint
    target = joint + np.array([math.sin(math.radians(60)), math.cos(math.radians(60)), 0.0]) * 0.2
    s.drag("wrist", target)
    assert s.base.joint_deg == pytest.approx(60.0)
    assert s.base.lower.length == pytest.approx(0.2)
    s.drag("elbow", np.array([0.0, 0.35, 0.01]))
    assert s.base.upper.length == pytest.approx(0.35)


def test_project_round_trip(tmp_path: Path) -> None:
    s = _session()
    s.set(("muscle", "belly"), 0.03)
    s.set_composite("flex", 0.3)
    world, _ = _bump(s)
    s.capture_sculpt("muscle", world)
    m = np.eye(4)
    m[:3, 3] = [0.0, 0.0, 0.01]
    s.capture_transform("lower bone", m)
    s.import_paths = (str(Path(limb.__file__).parent.resolve().relative_to(tmp_path.resolve(), walk_up=True)),)
    s.save(tmp_path / "p.json")
    t = Session.load(tmp_path / "p.json")
    assert P.to_dict(t.base) == P.to_dict(s.base)  # load reloads the model module: a new class, same values
    assert t.composites == s.composites
    a, b = s.evaluate(), t.evaluate()
    for name in a.elements:
        np.testing.assert_allclose(a.elements[name].verts, b.elements[name].verts, atol=1e-12)


def test_describe_reads_a_project_without_importing(tmp_path: Path) -> None:
    from proctune.core.session import describe  # noqa: PLC0415

    (tmp_path / "p.json").write_text('{"format": 1, "model": "evil:MODEL", "import_paths": ["../x"]}')
    info = describe(tmp_path / "p.json")
    assert info.model == "evil:MODEL"
    assert info.import_dirs == ((tmp_path / "../x").resolve(),)
