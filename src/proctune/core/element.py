r"""Elements: the meshes a model generates, each with the local coordinates its edits are stored in.

An element's vertices are given in its own normalized coordinates $l$, mapped to the world through a `Frame`
(origin $o$, orthonormal axes $A$, per-axis scale $s$): $w = o + A\,(s \odot l)$. The scale is whatever the
element's size is measured in (a bone's length along it and its radius across, say), so a coordinate of 0.5 along
means half way along at any length.

Edits live in those coordinates and are re-applied on every rebuild, which is what makes them survive parameter
changes:

- a *tweak*: the element moved, turned or scaled by hand (`Tweak`: translation, rotation and scale in its frame);
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

from proctune.errors import EditError

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

    def rigid(self) -> F64:
        """4x4: the frame's place and orientation, without its scale (where its element's object is parented)."""
        m = np.eye(4)
        m[:3, :3] = self.axes
        m[:3, 3] = self.origin
        return m


@dataclass(frozen=True)
class Tweak:
    r"""A hand move of an element relative to where it was generated, in the terms Blender shows an object's transform:
    a translation, a rotation and a scale along the element's own axes. In the element's frame, in metres
    ($p = s \odot l$ for its size $s$): $p' = R\,(k \odot p) + s \odot t$. The translation $t$ is in element sizes, so
    a muscle moved by a tenth of its length stays a tenth of its length away when it grows; rotation and scale are
    exactly what the object's Rotation and Scale fields show."""

    offset: F64 = field(default_factory=lambda: np.zeros(3))
    """Translation, in element sizes along its axes"""
    rotation: F64 = field(default_factory=lambda: np.eye(3))
    scale: F64 = field(default_factory=lambda: np.ones(3))

    def matrix(self, size: F64) -> F64:
        """4x4 in the element's frame, metres: what its object's transform is relative to its generated place."""
        m = np.eye(4)
        m[:3, :3] = self.rotation * self.scale[None, :]
        m[:3, 3] = size * self.offset
        return m

    @staticmethod
    def from_matrix(m: F64, size: F64) -> Tweak:
        """The tweak of a transform in the element's frame (polar decomposition: rotation times a stretch; a stretch
        off the element's axes, which Blender's location/rotation/scale cannot make, keeps only its diagonal)."""
        from scipy.linalg import polar  # noqa: PLC0415

        rot, stretch = polar(m[:3, :3])
        k = np.diag(stretch).copy()
        if np.linalg.det(rot) < 0:  # a mirror: keep it in the scale, as Blender does
            rot[:, 2] *= -1
            k[2] *= -1
        return Tweak(m[:3, 3] / size, np.asarray(rot, np.float64), k)

    def apply(self, local: F64, size: F64) -> F64:
        p = local * size
        return ((p * self.scale) @ self.rotation.T + size * self.offset) / size

    def invert(self, local: F64, size: F64) -> F64:
        p = local * size - size * self.offset
        return (p @ self.rotation) / self.scale / size

    @property
    def identity(self) -> bool:
        return bool(
            np.allclose(self.offset, 0.0) and np.allclose(self.rotation, np.eye(3)) and np.allclose(self.scale, 1.0)
        )

    def describe(self, size: F64) -> str:
        """In words, for the panel: how far it moved (mm), how far it turned (degrees), how it was scaled."""
        parts: list[str] = []
        moved = float(np.linalg.norm(size * self.offset)) * 1000
        if moved > 0.05:
            parts.append(f"moved {moved:.1f} mm")
        angle = float(np.degrees(np.arccos(np.clip((np.trace(self.rotation) - 1) / 2, -1.0, 1.0))))
        if angle > 0.05:
            parts.append(f"turned {angle:.1f} deg")
        if not np.allclose(self.scale, 1.0):
            parts.append("scaled " + " x ".join(f"{k:.2f}" for k in self.scale))
        return ", ".join(parts) or "no move"


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
        local = edit.tweak.apply(e.local, e.frame.scale) if e.tweakable else e.local
        return Resolved(e, e.frame.to_world(local), edit, stale)
    if e.tweakable:
        return Resolved(e, e.frame.to_world(edit.tweak.apply(e.local + d, e.frame.scale)), edit)
    assert e.vertex_axes is not None and e.vertex_scale is not None
    offset = np.einsum("nij,nj->ni", e.vertex_axes, d * e.vertex_scale)
    return Resolved(e, e.frame.to_world(e.local) + offset, edit)


def capture_sculpt(e: Element, tweak: Tweak, world: F64) -> F64:
    """The sculpt that turns the element (with `tweak`) into `world` (n, 3): the inverse of `resolve`."""
    if world.shape != e.local.shape:
        raise EditError(
            f"{e.name}: the edited mesh has {len(world)} vertices, the element {len(e.local)}",
            "a sculpt cannot add or remove vertices (no Dyntopo, Remesh or Decimate on generated meshes); undo it",
        )
    if e.tweakable:
        return tweak.invert(e.frame.to_local(world), e.frame.scale) - e.local
    assert e.vertex_axes is not None and e.vertex_scale is not None
    return np.einsum("nji,nj->ni", e.vertex_axes, world - e.frame.to_world(e.local)) / e.vertex_scale


def capture_tweak(e: Element, matrix: F64, current: Tweak) -> Tweak:
    """The element moved by a world-space 4x4 `matrix` on top of its `current` tweak."""
    rigid = e.frame.rigid()
    local = np.linalg.inv(rigid) @ matrix @ rigid @ current.matrix(e.frame.scale)
    return Tweak.from_matrix(local, e.frame.scale)
