r"""Elements: the meshes a model generates, each with the local coordinates its edits are stored in.

An element's vertices are given in its own normalized coordinates $l$, mapped to the world through a `Frame`
(origin $o$, orthonormal axes $A$, per-axis scale $s$): $w = o + A\,(s \odot l)$. The scale is whatever the
element's size is measured in (a bone's length along it and its radius across, say), so a coordinate of 0.5 along
means half way along at any length.

Edits live in those coordinates and are re-applied on every rebuild, which is what makes them survive parameter
changes:

- a *tweak*: an affine map $T(l) = L\,l + t$ in normalized coordinates (the element moved, turned or scaled by
  hand);
- a *sculpt*: a displacement per vertex $d_i$. For a movable element it is in normalized coordinates, before the
  tweak, so it moves with the element: $w_i = F(T(l_i + d_i))$. An element that cannot be moved as a whole (a skin
  following a bending limb) gives each vertex its own frame instead (`vertex_axes`, `vertex_scale`, e.g. along and
  around the limb at that point), and $w_i = F(l_i) + B_i\,(\sigma_i \odot d_i)$.

A model must keep an element's topology fixed across parameter values (same vertex count and order), or its
sculpt no longer applies; a sculpt captured at another topology is reported stale, not misapplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

type F64 = NDArray[np.float64]
type I32 = NDArray[np.int32]


@dataclass(frozen=True)
class Frame:
    origin: F64
    """(3,)"""
    axes: F64
    """(3, 3), columns are the unit axes"""
    scale: F64
    """(3,), the element's size along each axis"""

    @staticmethod
    def world() -> Frame:
        return Frame(np.zeros(3), np.eye(3), np.ones(3))

    def to_world(self, local: F64) -> F64:
        return self.origin + (local * self.scale) @ self.axes.T

    def to_local(self, world: F64) -> F64:
        return ((world - self.origin) @ self.axes) / self.scale


@dataclass(frozen=True)
class Tweak:
    """An affine map in an element's normalized coordinates: $T(l) = L\\,l + t$."""

    linear: F64 = field(default_factory=lambda: np.eye(3))
    offset: F64 = field(default_factory=lambda: np.zeros(3))

    def apply(self, local: F64) -> F64:
        return local @ self.linear.T + self.offset

    def invert(self, local: F64) -> F64:
        return np.linalg.solve(self.linear, (local - self.offset).T).T

    def then(self, other: Tweak) -> Tweak:
        """This tweak followed by `other`."""
        return Tweak(other.linear @ self.linear, other.linear @ self.offset + other.offset)

    @property
    def identity(self) -> bool:
        return bool(np.allclose(self.linear, np.eye(3)) and np.allclose(self.offset, 0.0))


@dataclass(frozen=True)
class Element:
    """A generated mesh in normalized coordinates, with the layer it belongs to (bone, muscle, skin, ...)."""

    name: str
    layer: str
    frame: Frame
    local: F64
    """(n, 3) vertices in normalized coordinates"""
    faces: I32
    vertex_axes: F64 | None = None
    """(n, 3, 3) a frame per vertex for its sculpt (columns are axes), for an element that is not movable"""
    vertex_scale: F64 | None = None
    """(n, 3) the sculpt's scale per vertex, with `vertex_axes`"""

    def __post_init__(self) -> None:
        if (self.vertex_axes is None) != (self.vertex_scale is None):
            raise ValueError(f"{self.name}: vertex_axes and vertex_scale go together")

    @property
    def tweakable(self) -> bool:
        """Whether the element can be moved as a whole: only if its sculpt is in its own frame."""
        return self.vertex_axes is None


@dataclass(frozen=True)
class Edit:
    """What was done to one element by hand."""

    tweak: Tweak = field(default_factory=Tweak)
    displacement: F64 | None = None
    """(n, 3) per-vertex sculpt in vertex frames, or None"""


@dataclass(frozen=True)
class Resolved:
    """An element with its edits applied: world vertices, and the edit that made them."""

    element: Element
    verts: F64
    """(n, 3) world"""
    edit: Edit
    stale: bool = False
    """A sculpt was stored for a different topology and is not applied"""

    @property
    def faces(self) -> I32:
        return self.element.faces


def resolve(e: Element, edit: Edit | None) -> Resolved:
    edit = edit or Edit()
    d = edit.displacement
    stale = d is not None and d.shape != e.local.shape
    if d is None or stale:
        local = edit.tweak.apply(e.local) if e.tweakable else e.local
        return Resolved(e, e.frame.to_world(local), edit, stale)
    if e.tweakable:
        return Resolved(e, e.frame.to_world(edit.tweak.apply(e.local + d)), edit)
    assert e.vertex_axes is not None and e.vertex_scale is not None
    offset = np.einsum("nij,nj->ni", e.vertex_axes, d * e.vertex_scale)
    return Resolved(e, e.frame.to_world(e.local) + offset, edit)


def capture_sculpt(e: Element, tweak: Tweak, world: F64) -> F64:
    """The sculpt that turns the element (with `tweak`) into `world` (n, 3): the inverse of `resolve`."""
    if world.shape != e.local.shape:
        raise ValueError(f"{e.name}: {len(world)} vertices, the element has {len(e.local)}")
    if e.tweakable:
        return tweak.invert(e.frame.to_local(world)) - e.local
    assert e.vertex_axes is not None and e.vertex_scale is not None
    return np.einsum("nji,nj->ni", e.vertex_axes, world - e.frame.to_world(e.local)) / e.vertex_scale


def capture_tweak(e: Element, matrix: F64) -> Tweak:
    r"""A world-space affine move of the element (4x4 `matrix`, e.g. its object's transform) as a tweak in its
    normalized coordinates: $l \mapsto S^{-1} A^\top (M (o + A S l) - o)$."""
    m3, mt = matrix[:3, :3], matrix[:3, 3]
    f = e.frame
    to_local = f.axes.T / f.scale[:, None]
    linear = to_local @ m3 @ f.axes * f.scale[None, :]
    offset = to_local @ (m3 @ f.origin + mt - f.origin)
    return Tweak(linear, offset)
