"""Curves, growth, fat curves, bone moves as parameters, and parameters over time."""

from __future__ import annotations

import json
import math
from pathlib import Path

import limb
import numpy as np
import pytest

from proctune.core import Curve, Session, allometric
from proctune.core import params as P
from proctune.core.element import F64


def _session() -> Session[limb.Limb]:
    return Session.new(limb.MODEL, "limb:MODEL", (str(Path(limb.__file__).parent),))


def _rot_z(deg: float) -> F64:
    a = math.radians(deg)
    m = np.eye(4)
    m[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return m


# ------------------------------------------------------------------------------------------------- curves


def test_curve_passes_through_its_points_without_overshooting() -> None:
    c = Curve.through((0.0, 0.0), (0.2, 0.8), (0.4, 0.85), (1.0, 1.0))
    for x, y in c.points:
        assert c(x) == pytest.approx(y)
    ys = c.evaluate(np.linspace(0.0, 1.0, 201))
    assert np.all(np.diff(ys) >= -1e-12)  # rising points, rising curve
    assert ys.max() <= 1.0 + 1e-12


def test_curve_holds_its_ends_and_rejects_bad_points() -> None:
    c = Curve.through((0.2, 0.3), (0.8, 0.9))
    assert c(-1.0) == pytest.approx(0.3)
    assert c(2.0) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="increasing x"):
        Curve.through((0.5, 0.0), (0.5, 1.0))


def test_curves_round_trip_through_json() -> None:
    p = limb.Limb()
    back = P.from_dict(limb.Limb, json.loads(json.dumps(P.to_dict(p))))
    assert back == p


def test_allometry() -> None:
    assert allometric(0.5, 1.0) == pytest.approx(0.5)
    assert allometric(0.5, 2.0) == pytest.approx(0.25)


# ------------------------------------------------------------------------------------------------- growth and fat


def test_growth_scales_bones_by_the_curve_and_the_muscle_faster() -> None:
    s = _session()
    s.set(("growth", "stage"), 0.0)
    q = limb.develop(s.effective)
    size = s.base.growth.curve(0.0)
    assert size == pytest.approx(0.3)
    assert q.upper.length == pytest.approx(0.30 * size)
    assert q.muscle.belly == pytest.approx(limb.Muscle().belly * size**1.4)
    r = s.evaluate()
    assert r.elements["upper bone"].verts[:, 1].max() == pytest.approx(0.30 * size, abs=1e-3)


def _fat_peak(s: Session[limb.Limb], name: str) -> float:
    e = s.evaluate().elements[name].element
    return float(e.frame.scale[0])  # the belly solid's girth: tip + peak


def test_each_fat_deposit_follows_its_own_curve() -> None:
    s = _session()
    lean, full = {}, {}
    for a, out in ((0.3, lean), (1.0, full)):
        s.set(("fat", "adiposity"), a)
        out["upper"], out["lower"] = _fat_peak(s, "upper fat"), _fat_peak(s, "lower fat")
    # the upper deposit fills in early (0.1 + linear-ish), the lower one late (0.12 at 0.6)
    assert lean["upper"] / full["upper"] > 0.3
    assert lean["lower"] / full["lower"] < 0.1


def test_skin_wraps_the_fat() -> None:
    s = _session()
    s.set(("fat", "adiposity"), 0.0)
    thin = s.evaluate().elements["skin"].verts
    s.set(("fat", "adiposity"), 1.0)
    thick = s.evaluate().elements["skin"].verts
    assert np.abs(thick - thin).max() > 0.004


def test_sculpt_survives_growth() -> None:
    s = _session()
    r = s.evaluate().elements["muscle"]
    world = r.verts.copy()
    world[100:140] += r.element.frame.axes[:, 0] * 0.003
    s.capture_sculpt("muscle", world)
    s.set(("growth", "stage"), 0.3)
    grown = s.evaluate().elements["muscle"]
    assert grown.edit.displacement is not None
    assert not grown.stale


# ------------------------------------------------------------------------------------------------- bone moves


def test_scaling_a_bone_along_itself_sets_its_length() -> None:
    s = _session()
    m = np.diag([1.0, 1.2, 1.0, 1.0])  # the upper bone lies along y
    s.capture_transform("upper bone", m)
    assert s.base.upper.length == pytest.approx(0.36)
    assert "upper bone" not in s.edits  # a parameter change, not a tweak


def test_turning_the_lower_bone_bends_the_joint() -> None:
    s = _session()
    s.capture_transform("lower bone", _rot_z(-10.0))  # towards +x is a negative turn about z
    assert s.base.joint_deg == pytest.approx(25.0)
    assert "lower bone" not in s.edits


def test_bone_moves_respect_parameter_ranges() -> None:
    s = _session()
    s.capture_transform("upper bone", np.diag([1.0, 5.0, 1.0, 1.0]))
    assert s.base.upper.length == pytest.approx(0.60)  # the slider's maximum


def test_other_elements_are_still_tweaked() -> None:
    s = _session()
    s.capture_transform("muscle", _rot_z(5.0))
    assert "muscle" in s.edits


def test_handles_account_for_growth() -> None:
    s = _session()
    s.set(("growth", "stage"), 0.0)
    elbow = next(h for h in s.handles() if h.name == "elbow")
    at = elbow.position(s.effective)
    assert at[1] == pytest.approx(0.30 * 0.3)
    s.drag("elbow", at)  # dragging it to where it is changes nothing
    assert s.base.upper.length == pytest.approx(0.30)


# ------------------------------------------------------------------------------------------------- over time


def test_tracks_interpolate_and_hold() -> None:
    s = _session()
    s.tracks = {"growth.stage": ((1.0, 0.0), (11.0, 1.0)), "composite:flex": ((1.0, 0.0), (11.0, 0.5))}
    mid = s.at_frame(6.0)
    assert mid.base.growth.stage == pytest.approx(0.5)
    assert mid.composites["flex"] == pytest.approx(0.25)
    assert s.at_frame(50.0).base.growth.stage == pytest.approx(1.0)
    assert s.base.growth.stage == 1.0  # the session itself is untouched


def test_tracks_round_trip(tmp_path: Path) -> None:
    s = _session()
    s.tracks = {"growth.stage": ((1.0, 0.0), (11.0, 1.0))}
    s.import_paths = (str(Path(limb.__file__).parent),)
    s.save(tmp_path / "p.json")
    t = Session.load(tmp_path / "p.json")
    assert t.tracks == s.tracks
