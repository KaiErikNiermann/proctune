"""Fit the limb to measurements with Optuna: python fit.py [TRIALS] [OUT.json]

The targets are made up but typical of what one would measure on a reference (a scan, a photo with a ruler):
the limb's total length, the skin's largest girth around the muscle, and the skin's volume. The fit tunes the bone
lengths, the muscle's belly and the skin's thickness within their slider ranges and saves the result as a project
that opens in Blender.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import limb
import numpy as np
import optuna
import trimesh

from proctune.core import Result, Session
from proctune.core.tune import Target, apply, loss, objective, report, space

log = logging.getLogger("fit")


def total_length(_: Result, p: limb.Limb) -> float:
    return p.upper.length + p.lower.length


def girth(r: Result, p: limb.Limb) -> float:
    """The skin's largest circumference, from its rings (m)."""
    rings = (
        r.elements["skin"].verts[: limb.SKIN_RINGS * limb.SKIN_SECTIONS].reshape(limb.SKIN_RINGS, limb.SKIN_SECTIONS, 3)
    )
    closed = np.concatenate([rings, rings[:, :1]], axis=1)
    return float(np.linalg.norm(np.diff(closed, axis=1), axis=2).sum(axis=1).max())


def volume_l(r: Result, _: limb.Limb) -> float:
    s = r.elements["skin"]
    return float(trimesh.Trimesh(s.verts, s.faces, process=False).volume) * 1000.0


TARGETS: tuple[Target[limb.Limb], ...] = (
    Target("total length (m)", total_length, 0.62),
    Target("largest girth (m)", girth, 0.31),
    Target("skin volume (l)", volume_l, 2.1),
)
TUNED = ("upper.length", "lower.length", "muscle.belly", "skin.thickness")


def main(trials: int = 150, out: Path | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    here = Path(__file__).parent
    s = Session.new(limb.MODEL, "limb:MODEL", (".",))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective(s, space(s, TUNED), loss(TARGETS)), n_trials=trials)
    apply(s, study.best_params)
    log.info("best loss %.2e after %d trials", study.best_value, trials)
    for name, (got, want) in report(s, TARGETS).items():
        log.info("%-20s %.4f (target %.4f)", name, got, want)
    path = out or here / "fitted.json"
    s.save(path)
    log.info("saved %s", path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150, Path(sys.argv[2]) if len(sys.argv) > 2 else None)
