r"""A minimal proctune model: two bones and one joint (scaffold), one muscle across the joint (structure), and a skin
swept over both (surface).

- Bones: solids of revolution along each bone, flared at the ends.
- Muscle: a solid of revolution between its origin on the upper bone and its insertion on the lower one, with a
  bell-shaped belly $r(t) = r_t + b \, e^{-(t - c)^2 / 2w^2} \sqrt{\sin \pi t}$ (tendon radius $r_t$, belly $b$,
  centre $c$, width $w$, $t \in [0, 1]$ along it). Bending the joint brings its ends together, so it shortens.
- Skin: a tube around a smoothed centreline through the bones. At each ring and angle its radius is the furthest
  point of the bones and muscle (resolved, so sculpts show through) in that direction, smoothed, plus a thickness.
  Every skin vertex has its own frame (out, along, around the limb), so a skin sculpt follows the bend.

Composites: `flex` bends the joint and bunches the muscle; `size` scales the whole limb. Handles: the elbow (sets
the upper bone's length) and the wrist (sets the joint angle and the lower bone's length).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Annotated, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import interpolate, ndimage
from scipy.spatial import KDTree

from proctune.core import Composite, Element, Frame, Handle, Range, Resolved, Stage

type F64 = NDArray[np.float64]

#: Mesh resolution (fixed: a sculpt is stored per vertex)
RINGS, SECTIONS = 48, 32
SKIN_RINGS, SKIN_SECTIONS = 96, 48
#: Plane normal of the joint's bend (the chain lies in the x-y plane, the upper bone along +y)
BEND_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Bone:
    length: Annotated[float, Range(0.10, 0.60, 0.005, "m", "Joint to joint")] = 0.30
    radius: Annotated[float, Range(0.008, 0.04, 0.001, "m", "Shaft radius")] = 0.016
    flare: Annotated[float, Range(0.0, 1.0, 0.01, "", "How much wider the ends are than the shaft")] = 0.35


@dataclass(frozen=True)
class Muscle:
    belly: Annotated[float, Range(0.0, 0.06, 0.001, "m", "How far the belly swells beyond the tendon")] = 0.024
    centre: Annotated[float, Range(0.2, 0.8, 0.01, "", "Where the belly peaks, along the muscle")] = 0.45
    width: Annotated[float, Range(0.05, 0.5, 0.01, "", "Belly width (sigma, share of the muscle's length)")] = 0.2
    tendon: Annotated[float, Range(0.002, 0.012, 0.0005, "m", "Tendon radius at both ends")] = 0.004
    origin: Annotated[float, Range(0.0, 0.6, 0.01, "", "Where it starts, share of the upper bone")] = 0.15
    insertion: Annotated[float, Range(0.01, 0.15, 0.005, "m", "Where it ends, along the lower bone")] = 0.04
    lift: Annotated[float, Range(0.0, 0.04, 0.001, "m", "Distance of its line from the bones' axes")] = 0.02


@dataclass(frozen=True)
class Skin:
    thickness: Annotated[float, Range(0.0, 0.02, 0.0005, "m", "Fat and skin over the outermost tissue")] = 0.004
    smoothing: Annotated[float, Range(0.0, 6.0, 0.1, "", "Gaussian sigma, in rings and sections")] = 2.0
    corner: Annotated[float, Range(0.0, 0.05, 0.001, "", "How much the centreline rounds the joint")] = 0.004


@dataclass(frozen=True)
class Limb:
    upper: Bone = field(default_factory=Bone)
    lower: Bone = field(default_factory=lambda: Bone(length=0.26, radius=0.013))
    joint_deg: Annotated[float, Range(0.0, 150.0, 1.0, "deg", "Flexion at the joint")] = 15.0
    muscle: Muscle = field(default_factory=Muscle)
    skin: Skin = field(default_factory=Skin)


# ------------------------------------------------------------------------------------------------- geometry


def _unit(v: F64) -> F64:
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class Chain:
    root: F64
    joint: F64
    tip: F64
    d1: F64
    d2: F64


def chain(p: Limb) -> Chain:
    a = math.radians(p.joint_deg)
    d1 = np.array([0.0, 1.0, 0.0])
    d2 = np.array([math.sin(a), math.cos(a), 0.0])  # flexion turns the lower bone towards +x, the muscle's side
    root = np.zeros(3)
    joint = root + d1 * p.upper.length
    return Chain(root, joint, joint + d2 * p.lower.length, d1, d2)


def _frame(origin: F64, along: F64, scale: F64) -> Frame:
    """y along the bone, z the bend axis, x = y cross z (towards the flexion side)."""
    y = _unit(along)
    z = _unit(BEND_AXIS - y * float(BEND_AXIS @ y))
    return Frame(origin, np.column_stack([np.cross(y, z), y, z]), scale)


def revolution(profile: F64, sections: int = SECTIONS) -> tuple[F64, NDArray[np.int32]]:
    """A closed solid of revolution about the y axis from radii `profile` at evenly spaced t in [0, 1] (zero at both
    ends: poles). trimesh revolves about z; it is turned onto y. Topology depends only on the counts."""
    t = np.linspace(0.0, 1.0, len(profile))
    to_y = cast(F64, trimesh.transformations.rotation_matrix(-math.pi / 2, [1.0, 0.0, 0.0]))  # pyright: ignore[reportUnknownMemberType]
    revolve = cast(Callable[..., trimesh.Trimesh], trimesh.creation.revolve)  # pyright: ignore[reportUnknownMemberType]  # untyped
    m = revolve(np.column_stack([profile, t]), sections=sections, transform=to_y)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


def _bone(name: str, origin: F64, along: F64, b: Bone) -> Element:
    t = np.linspace(0.0, 1.0, RINGS)
    ends = np.exp(-((t / 0.08) ** 2)) + np.exp(-(((1.0 - t) / 0.08) ** 2))
    profile = (1.0 + b.flare * ends) * np.sin(np.pi * t).clip(0.0) ** 0.15  # rounded ends
    profile[[0, -1]] = 0.0
    local, faces = revolution(profile)
    return Element(name, "bone", _frame(origin, along, np.array([b.radius, b.length, b.radius])), local, faces)


def bones(p: Limb, _: Mapping[str, Resolved]) -> tuple[Element, ...]:
    c = chain(p)
    return (_bone("upper bone", c.root, c.d1, p.upper), _bone("lower bone", c.joint, c.d2, p.lower))


def muscle_ends(p: Limb) -> tuple[F64, F64]:
    c = chain(p)
    side = _frame(c.root, c.d1, np.ones(3)).axes[:, 0]
    side2 = _frame(c.joint, c.d2, np.ones(3)).axes[:, 0]
    start = c.root + c.d1 * p.upper.length * p.muscle.origin + side * p.muscle.lift
    end = c.joint + c.d2 * p.muscle.insertion + side2 * p.muscle.lift * 0.6
    return start, end


def muscles(p: Limb, _: Mapping[str, Resolved]) -> tuple[Element, ...]:
    m = p.muscle
    start, end = muscle_ends(p)
    t = np.linspace(0.0, 1.0, RINGS)
    r = m.tendon + m.belly * np.exp(-((t - m.centre) ** 2) / (2 * m.width**2)) * np.sqrt(np.sin(np.pi * t).clip(0.0))
    girth = m.tendon + m.belly
    profile = r / girth
    profile[[0, -1]] = 0.0
    local, faces = revolution(profile)
    scale = np.array([girth, float(np.linalg.norm(end - start)), girth])
    return (Element("muscle", "muscle", _frame(start, end - start, scale), local, faces),)


def centreline(p: Limb, n: int = SKIN_RINGS) -> tuple[F64, F64]:
    """Points and unit tangents of a smoothing spline through the bone chain (it rounds the joint)."""
    c = chain(p)
    pts = np.vstack([np.linspace(c.root, c.joint, 40, endpoint=False), np.linspace(c.joint, c.tip, 40)])
    spline, _ = interpolate.make_splprep(pts.T, s=p.skin.corner * len(pts) * 1e-2, k=3)
    u = np.linspace(0.0, 1.0, n)
    pos = np.asarray(spline(u), np.float64).T  # the spline gives (ndim, n)
    tan = np.asarray(spline.derivative()(u), np.float64).T
    return pos, tan / np.linalg.norm(tan, axis=1, keepdims=True)


def skin(p: Limb, done: Mapping[str, Resolved]) -> tuple[Element, ...]:
    pos, tan = centreline(p)
    out = np.cross(tan, BEND_AXIS)  # in the bend plane, towards +x at rest
    around = np.cross(out, tan)
    # every tissue vertex to its nearest centreline point, then to an angle and a distance in that ring's plane
    tissue = np.vstack([r.verts for r in done.values()])
    ring = np.asarray(KDTree(pos).query(tissue)[1], np.intp)
    rel: F64 = tissue - pos[ring]
    rel -= tan[ring] * np.einsum("ij,ij->i", rel, tan[ring])[:, None]
    theta: F64 = np.arctan2(np.einsum("ij,ij->i", rel, around[ring]), np.einsum("ij,ij->i", rel, out[ring]))
    sector = np.round(theta / (2 * np.pi) * SKIN_SECTIONS).astype(np.intp) % SKIN_SECTIONS
    radius = np.zeros((SKIN_RINGS, SKIN_SECTIONS))
    dist: F64 = np.linalg.norm(rel, axis=1)
    np.maximum.at(radius, (ring, sector), dist)
    # fill sectors no vertex fell in, then smooth (wrapping around the limb)
    radius = ndimage.maximum_filter(radius, size=(3, 3), mode=("nearest", "wrap"))
    if p.skin.smoothing > 0:
        radius = ndimage.gaussian_filter(radius, p.skin.smoothing, mode=("nearest", "wrap"))
    radius += p.skin.thickness
    ang = np.linspace(0.0, 2 * np.pi, SKIN_SECTIONS, endpoint=False)
    radial = np.cos(ang)[None, :, None] * out[:, None, :] + np.sin(ang)[None, :, None] * around[:, None, :]
    ring_verts = pos[:, None, :] + radial * radius[:, :, None]
    caps = np.vstack([pos[0] - tan[0] * radius[0].mean() * 0.6, pos[-1] + tan[-1] * radius[-1].mean() * 0.6])
    verts = np.vstack([ring_verts.reshape(-1, 3), caps])
    faces = _tube_faces(SKIN_RINGS, SKIN_SECTIONS)
    # per-vertex sculpt frames: out, along, around; scaled by the local radius so a sculpt grows with the limb
    tangent = np.repeat(tan, SKIN_SECTIONS, axis=0)
    rad = radial.reshape(-1, 3)
    axes = np.stack([rad, tangent, np.cross(rad, tangent)], axis=2)
    cap_axes = np.stack([np.column_stack([-tan[0], *_perp(tan[0])]), np.column_stack([tan[-1], *_perp(tan[-1])])])
    scale = np.repeat(
        np.vstack([radius.reshape(-1, 1), radius[[0], :].mean(keepdims=True), radius[[-1], :].mean(keepdims=True)]),
        3,
        axis=1,
    )
    return (Element("skin", "skin", Frame.world(), verts, faces, np.concatenate([axes, cap_axes]), scale),)


def _perp(v: F64) -> tuple[F64, F64]:
    a = _unit(np.cross(v, BEND_AXIS))
    return a, np.cross(v, a)


def _tube_faces(rings: int, sections: int) -> NDArray[np.int32]:
    """Triangles of a ring grid (rings x sections, vertex index ring * sections + section) closed by two poles."""
    i, j = np.meshgrid(np.arange(rings - 1), np.arange(sections), indexing="ij")
    a = i * sections + j
    b = i * sections + (j + 1) % sections
    c, d = a + sections, b + sections
    sides = np.vstack(
        [np.column_stack([a.ravel(), b.ravel(), d.ravel()]), np.column_stack([a.ravel(), d.ravel(), c.ravel()])]
    )
    j1 = np.arange(sections)
    p0, p1 = rings * sections, rings * sections + 1
    last = (rings - 1) * sections
    cap0 = np.column_stack([np.full(sections, p0), (j1 + 1) % sections, j1])
    cap1 = np.column_stack([np.full(sections, p1), last + j1, last + (j1 + 1) % sections])
    return np.ascontiguousarray(np.vstack([sides, cap0, cap1])[:, ::-1], np.int32)  # outward winding


# ------------------------------------------------------------------------------------------------- tuning


def flex(p: Limb, s: float) -> Limb:
    """Bend the joint by up to 120 degrees and bunch the muscle: its belly swells and narrows as it shortens."""
    m = replace(p.muscle, belly=p.muscle.belly * (1.0 + 0.7 * s), width=p.muscle.width * (1.0 - 0.3 * s))
    return replace(p, joint_deg=min(p.joint_deg + 120.0 * s, 150.0), muscle=m)


def size(p: Limb, k: float) -> Limb:
    """Scale every length in the limb by $2^k$ (so -1 halves it and +1 doubles it)."""
    f = 2.0**k

    def bone(b: Bone) -> Bone:
        return replace(b, length=b.length * f, radius=b.radius * f)

    m = p.muscle
    m = replace(m, belly=m.belly * f, tendon=m.tendon * f, insertion=m.insertion * f, lift=m.lift * f)
    return replace(
        p, upper=bone(p.upper), lower=bone(p.lower), muscle=m, skin=replace(p.skin, thickness=p.skin.thickness * f)
    )


def _drag_elbow(p: Limb, w: F64) -> Limb:
    c = chain(p)
    return replace(p, upper=replace(p.upper, length=float(np.clip((w - c.root) @ c.d1, 0.1, 0.6))))


def _drag_wrist(p: Limb, w: F64) -> Limb:
    c = chain(p)
    v = w - c.joint
    v -= BEND_AXIS * float(v @ BEND_AXIS)
    angle = math.degrees(math.atan2(float(v @ np.array([1.0, 0.0, 0.0])), float(v @ c.d1)))
    lower = replace(p.lower, length=float(np.clip(np.linalg.norm(v), 0.1, 0.6)))
    return replace(p, joint_deg=float(np.clip(angle, 0.0, 150.0)), lower=lower)


@dataclass(frozen=True)
class LimbModel:
    name: str = "limb"

    def defaults(self) -> Limb:
        return Limb()

    @property
    def stages(self) -> tuple[Stage[Limb], ...]:
        return (Stage("scaffold", bones), Stage("structure", muscles), Stage("surface", skin))

    @property
    def composites(self) -> tuple[Composite[Limb], ...]:
        return (
            Composite("flex", Range(0.0, 1.0, 0.01, "", "Bend the joint and bunch the muscle"), flex),
            Composite("size", Range(-1.0, 1.0, 0.01, "", "Scale the whole limb by 2^size"), size),
        )

    def handles(self, p: Limb) -> tuple[Handle[Limb], ...]:
        return (
            Handle("elbow", lambda q: chain(q).joint, _drag_elbow, (("upper", "length"),)),
            Handle("wrist", lambda q: chain(q).tip, _drag_wrist, (("joint_deg",), ("lower", "length"))),
        )


MODEL = LimbModel()
