"""Meshes with a fixed vertex layout for common primitives, in normalized coordinates, for models to build elements
from (a sculpt is kept per vertex, so a primitive's mesh must have the same layout at every size). Needs trimesh.

Each returns ``(local, faces)``: vertices in the element's normalized coordinates (its frame's scale is the size of
the primitive) and triangles.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

type F64 = NDArray[np.float64]
type I32 = NDArray[np.int32]


def revolution(profile: F64, along: F64 | None = None, sections: int = 32) -> tuple[F64, I32]:
    """A closed solid of revolution about the y axis: radii `profile` at positions `along` on y (default: evenly from
    0 to 1), zero at both ends (the poles). trimesh revolves about z; the result is turned onto y. The layout depends
    only on the counts."""
    import trimesh  # noqa: PLC0415

    t = np.linspace(0.0, 1.0, len(profile)) if along is None else along
    to_y = cast(F64, trimesh.transformations.rotation_matrix(-math.pi / 2, [1.0, 0.0, 0.0]))  # pyright: ignore[reportUnknownMemberType]
    revolve = cast(Callable[..., trimesh.Trimesh], trimesh.creation.revolve)  # pyright: ignore[reportUnknownMemberType]  # untyped
    m = revolve(np.column_stack([profile, t]), sections=sections, transform=to_y)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


@cache
def _sphere(subdivisions: int) -> tuple[F64, I32]:
    import trimesh  # noqa: PLC0415

    icosphere = cast(Callable[..., trimesh.Trimesh], trimesh.creation.icosphere)  # pyright: ignore[reportUnknownMemberType]
    m = icosphere(subdivisions=subdivisions, radius=1.0)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


def sphere(subdivisions: int = 3) -> tuple[F64, I32]:
    """The unit sphere (an icosphere): an ellipsoid when its frame's scale is the radii."""
    v, f = _sphere(subdivisions)
    return v.copy(), f.copy()


def round_cone(ra: float, rb: float, length: float, rings: int = 24, sections: int = 24) -> tuple[F64, I32]:
    r"""Spheres of radius `ra` at 0 and `rb` at `length` on y joined by their tangent cone, in coordinates normalized
    by $(r_{max}, \ell, r_{max})$. The radius along y is the larger of the two caps and the straight taper between
    them, sampled at `rings` positions (the layout depends only on the counts)."""
    r_max = max(ra, rb)
    y = np.linspace(-ra, length + rb, rings)
    taper = ra + (rb - ra) * np.clip(y / length, 0.0, 1.0)
    cap_a = np.sqrt(np.clip(ra**2 - y**2, 0.0, None))
    cap_b = np.sqrt(np.clip(rb**2 - (y - length) ** 2, 0.0, None))
    r = np.where((y >= 0) & (y <= length), taper, np.maximum(cap_a, cap_b))
    r[[0, -1]] = 0.0
    return revolution(r / r_max, y / length, sections)


def prism(a: F64, b: F64, c: F64, half: float) -> tuple[F64, F64, I32]:
    """A triangle thickened by `half` on both sides: ``(axes, local, faces)`` with axes x along a-b, z the triangle's
    normal (columns), and local in metres from `a` (a frame scale of 1)."""
    x = (b - a) / np.linalg.norm(b - a)
    z = np.cross(b - a, c - a)
    z /= np.linalg.norm(z)
    axes = np.column_stack([x, np.cross(z, x), z])
    pts = np.vstack([a, b, c])
    world = np.vstack([pts - z * half, pts + z * half])
    faces = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]], np.int32)
    return axes, (world - a) @ axes, faces
