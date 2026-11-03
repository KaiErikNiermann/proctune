"""The tuning harness: the search space, a deterministic trial, and a real Optuna fit."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import limb
import pytest

from proctune.core import Session
from proctune.core.tune import Target, apply, candidate, loss, objective, report, space


def _session() -> Session[limb.Limb]:
    return Session.new(limb.MODEL, "limb:MODEL", (str(Path(limb.__file__).parent),))


class MidTrial:
    """Suggests the middle of every range: a trial without an optimizer."""

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float:
        return (low + high) / 2

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        return (low + high) // 2

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        return choices[0]


def test_space_takes_the_slider_ranges() -> None:
    sp = space(_session(), ("upper.*", "composite:flex"), exclude=("upper.flare",))
    assert sp.names == ("upper.length", "upper.radius", "composite:flex")
    assert sp.dimensions[0].low == 0.10
    assert sp.dimensions[0].high == 0.60


def test_composites_only_when_asked_by_name() -> None:
    assert not any(n.startswith("composite:") for n in space(_session()).names)


def test_candidate_leaves_the_session_alone() -> None:
    s = _session()
    c = candidate(s, {"upper.length": 0.5, "composite:flex": 0.5})
    assert c.base.upper.length == 0.5
    assert s.base.upper.length == 0.30
    assert s.composites["flex"] == 0.0


def test_objective_scores_a_trial() -> None:
    s = _session()
    target = Target[limb.Limb]("upper", lambda _r, p: p.upper.length, 0.35)
    f = objective(s, space(s, ("upper.length",)), loss([target]))
    assert f(MidTrial()) == pytest.approx(((0.35 - 0.35) / 0.35) ** 2)  # the middle of 0.10-0.60 is 0.35


def test_optuna_fits_targets() -> None:
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = _session()
    targets = [
        Target[limb.Limb]("total", lambda _r, p: p.upper.length + p.lower.length, 0.7),
        Target[limb.Limb]("ratio", lambda _r, p: p.upper.length / p.lower.length, 1.2, scale=1.0),
    ]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective(s, space(s, ("upper.length", "lower.length")), loss(targets)), n_trials=80)
    apply(s, study.best_params)
    got = report(s, targets)
    assert got["total"][0] == pytest.approx(0.7, rel=0.03)
    assert got["ratio"][0] == pytest.approx(1.2, rel=0.05)
