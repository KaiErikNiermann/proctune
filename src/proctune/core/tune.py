r"""Automated tuning: fit a model's parameters to targets with any optimizer that suggests values the way Optuna's
``Trial`` does.

The pieces:

- `space` picks the parameters (and composites) to tune by dotted path patterns; their bounds are the parameters'
  own `Range`s, so the search space is the same one the sliders show.
- `Target` measures one number on an evaluated model and says what it should be; `loss` folds targets into one
  score, $\sum_i w_i \left(\frac{m_i - v_i}{s_i}\right)^2$ (measure $m_i$, target $v_i$, scale $s_i$, weight $w_i$).
- `objective` turns a session, a space and a score into ``f(trial) -> float`` for ``study.optimize``; `apply` writes
  the best values back into the session (and so into the Blender panel and the project file).

Hand edits (sculpts, tweaks) stay applied in every candidate, so a fit works on the edited model. Optuna is not a
dependency of the core: anything with `suggest_float`/`suggest_int`/`suggest_categorical` works, which is also how
the tests drive it without Optuna.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from proctune.core import params as P
from proctune.core.session import COMPOSITE, Result, Session


class Trial(Protocol):
    """What an optimizer's trial must provide (a subset of ``optuna.Trial``)."""

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float: ...

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int: ...

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any: ...


@dataclass(frozen=True)
class Dimension:
    """One tuned value: a parameter (dotted path) or a composite (``composite:NAME``), with its bounds."""

    name: str
    kind: P.Kind
    low: float
    high: float
    step: float | None = None


@dataclass(frozen=True)
class Space:
    dimensions: tuple[Dimension, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)


def space(
    session: Session[Any], include: Iterable[str] = ("*",), exclude: Iterable[str] = (), *, steps: bool = False
) -> Space:
    """Parameters and composites whose names match an `include` pattern and no `exclude` pattern (fnmatch on dotted
    paths, e.g. ``"muscle.*"``, ``"composite:flex"``). Every float or int needs a `Range`; booleans are tuned as a
    choice. `steps` keeps each range's slider step as the search grid (off: continuous)."""
    inc, exc = tuple(include), tuple(exclude)

    def wanted(name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, p) for p in inc) and not any(fnmatch.fnmatchcase(name, p) for p in exc)

    dims: list[Dimension] = []
    for spec in P.specs(session.base):
        name = ".".join(spec.path)
        if not wanted(name) or spec.kind == "curve":
            continue  # curves are drawn, not searched
        if spec.kind == "bool":
            dims.append(Dimension(name, "bool", 0, 1))
            continue
        if spec.range is None:
            raise ValueError(f"{name}: a tuned parameter needs a Range")
        dims.append(Dimension(name, spec.kind, spec.range.lo, spec.range.hi, spec.range.step if steps else None))
    for c in session.model.composites:
        name = COMPOSITE + c.name
        if wanted(name) and any(fnmatch.fnmatchcase(name, p) for p in inc if p.startswith(COMPOSITE)):
            dims.append(Dimension(name, "float", c.range.lo, c.range.hi, c.range.step if steps else None))
    if not dims:
        raise ValueError(f"no parameter matches {inc} (excluding {exc})")
    return Space(tuple(dims))


def sample(sp: Space, trial: Trial) -> dict[str, P.Value]:
    out: dict[str, P.Value] = {}
    for d in sp.dimensions:
        match d.kind:
            case "bool":
                out[d.name] = bool(trial.suggest_categorical(d.name, (False, True)))
            case "int":
                out[d.name] = trial.suggest_int(d.name, int(d.low), int(d.high), step=int(d.step or 1))
            case "float":
                out[d.name] = trial.suggest_float(d.name, d.low, d.high, step=d.step)
            case "curve":  # never in a space (see `space`)
                raise ValueError(f"{d.name}: curves are not searched")
    return out


def apply(session: Session[Any], values: Mapping[str, P.Value]) -> None:
    """Write tuned values (dotted paths, ``composite:NAME``) into the session, e.g. ``study.best_params``."""
    session.apply(values)


def candidate[T](session: Session[T], values: Mapping[str, P.Value]) -> Session[T]:
    """A copy of the session with `values` applied; the original is untouched (edits are shared, read only)."""
    c = dataclasses.replace(session, composites=dict(session.composites), last=None)
    apply(c, values)
    return c


type Measure[T] = Callable[[Result, T], float]
type Score[T] = Callable[[Result, T], float]


@dataclass(frozen=True)
class Target[T]:
    """A measured number and the value it should have. `scale` makes targets in different units comparable (by
    default the target's own size, so errors are relative)."""

    name: str
    measure: Measure[T]
    value: float
    weight: float = 1.0
    scale: float | None = None

    def error(self, result: Result, p: T) -> float:
        s = self.scale if self.scale is not None else max(abs(self.value), 1e-12)
        return (self.measure(result, p) - self.value) / s


def loss[T](targets: Sequence[Target[T]]) -> Score[T]:
    r"""$\sum_i w_i\,e_i^2$ over the targets' scaled errors $e_i$."""

    def score(result: Result, p: T) -> float:
        return sum(t.weight * t.error(result, p) ** 2 for t in targets)

    return score


def report[T](session: Session[T], targets: Sequence[Target[T]]) -> dict[str, tuple[float, float]]:
    """Each target's measured and wanted value, on the session as it is."""
    r = session.evaluate()
    p = session.effective
    return {t.name: (t.measure(r, p), t.value) for t in targets}


def objective[T](session: Session[T], sp: Space, score: Score[T]) -> Callable[[Trial], float]:
    """``f(trial)``: sample the space, evaluate a candidate, score it (lower is better: ``direction="minimize"``)."""

    def f(trial: Trial) -> float:
        c = candidate(session, sample(sp, trial))
        return score(c.evaluate(), c.effective)

    return f
