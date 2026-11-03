"""What a model provides: its parameters, the stages that generate its elements, composites and handles.

A model may also implement `Binds`, turning a hand move of an element into parameter changes, and give a
``layer_colours`` attribute (layer name to RGBA) for how its layers are drawn.

A model is built in stages, each seeing the elements of the stages before it *with their edits applied*: a scaffold
(a skeleton), structure placed on it (muscles, fat), then a surface over everything (a skin). So a sculpted muscle
shows through the skin, and the skin can be sculpted in turn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from proctune.core.element import Element, Resolved
from proctune.core.params import Path, Range

type F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Stage[P]:
    name: str
    build: Callable[[P, Mapping[str, Resolved]], tuple[Element, ...]]
    """Parameters and every earlier element (resolved, by name) to this stage's elements"""


@dataclass(frozen=True)
class Composite[P]:
    """One slider that moves several parameters together (a "flex", a "volume-preserving stretch"): a pure function
    applied on top of the base parameters, in declared order."""

    name: str
    range: Range
    apply: Callable[[P, float], P]
    default: float = 0.0


@dataclass(frozen=True)
class Handle[P]:
    """A point in the scene bound to parameters: drawn at `position(p)`, and dragging it to a world point `w` sets
    the parameters to `drag(p, w)` (a bone's tip that sets its length, say). If `turn` is given, turning the handle
    by a world-space rotation `r` (3x3) sets them to `turn(p, r)` (a joint's handle that bends it); a handle without
    one cannot be turned (its rotation is locked)."""

    name: str
    position: Callable[[P], F64]
    drag: Callable[[P, F64], P]
    edits: tuple[Path, ...] = ()
    """Parameters the handle changes, for the UI"""
    turn: Callable[[P, F64], P] | None = None


class Binds[P](Protocol):
    """A model whose elements can be moved by hand into parameter changes (a bone scaled along itself changes its
    length; the lower bone turned about the joint changes the joint's angle)."""

    def bind(self, p: P, element: str, matrix: F64) -> P | None:
        """The parameters after `element` was moved by the world-space 4x4 `matrix`, or None if that element's moves
        are not parameters (they are then kept as a tweak)."""
        ...


class Model[P](Protocol):
    @property
    def name(self) -> str: ...

    def defaults(self) -> P:
        """The parameters a new session starts from."""
        ...

    @property
    def stages(self) -> tuple[Stage[P], ...]: ...

    @property
    def composites(self) -> tuple[Composite[P], ...]: ...

    def handles(self, p: P) -> tuple[Handle[P], ...]: ...
