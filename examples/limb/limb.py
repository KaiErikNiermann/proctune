r"""A minimal proctune model: two bones and one joint (scaffold), one muscle across the joint (structure), and a skin
swept over both (surface).

- Bones: solids of revolution along each bone, flared at the ends.
- Muscle: a solid of revolution between its origin on the upper bone and its insertion on the lower one, with a
  bell-shaped belly $r(t) = r_t + b \, e^{-(t - c)^2 / 2w^2} \sqrt{\sin \pi t}$ (tendon radius $r_t$, belly $b$,
  centre $c$, width $w$, $t \in [0, 1]$ along it). Bending the joint brings its ends together, so it shortens.
- Skin: a tube around a smoothed centreline through the bones. At each ring and angle its radius is the furthest
  point of the bones and muscle (resolved, so sculpts show through) in that direction, smoothed, plus a thickness.
  Every skin vertex has its own frame (out, along, around the limb), so a skin sculpt follows the bend.
- Wrap skin, for comparison: the same tissue covered the way Blender's Shrinkwrap and Corrective Smooth would: a
  loose tube snapped onto the nearest tissue surface with an offset, relaxed, in a few rounds (`wrap_skin`).

- Fat: a deposit along each bone, its thickness set by overall adiposity through its own curve (one fills in early
  and mildly, the other late and steeply).

Growth: `growth.stage` (0 to 1, think age) drives the overall size through a free-form growth curve $s$; bones scale
by $s$, the muscle's belly by $s^{b_m}$ and the fat by $s^{b_f}$ (allometric exponents: above 1 grows faster than the
bones). All parameters describe the grown limb; `develop` gives the limb at its stage.

Composites: `flex` bends the joint and bunches the muscle; `size` scales the whole limb. Handles: the elbow (sets
the upper bone's length) and the wrist (sets the joint angle and the lower bone's length). Moving a bone by hand is a
parameter change too (`LimbModel.bind`): scaling it along itself sets its length, across itself its radius, and
turning the lower bone about the joint sets the joint's angle.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Annotated, cast

import igl  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import interpolate, ndimage
from scipy.spatial import KDTree

from proctune.core import Composite, Curve, Element, Frame, Handle, Range, Resolved, Stage, allometric
from proctune.core.params import clamp

type F64 = NDArray[np.float64]

#: Mesh resolution (fixed: a sculpt is stored per vertex)
RINGS, SECTIONS = 48, 32
SKIN_RINGS, SKIN_SECTIONS = 96, 48
#: The wrapped skin's tube: coarser, as each round queries the nearest tissue point for every vertex
WRAP_RINGS, WRAP_SECTIONS = 64, 32
#: Layers the skins cover
TISSUE = ("bone", "muscle", "fat")
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
class Wrap:
    """The second skin, made the way Blender's Shrinkwrap (Nearest Surface Point, with an Offset) followed by
    Corrective Smooth would make it, for comparison with the sweep."""

    enabled: bool = True
    offset: Annotated[float, Range(0.0, 0.02, 0.0005, "m", "Distance kept from the tissue (Shrinkwrap's Offset)")] = (
        0.004
    )
    passes: Annotated[int, Range(1, 8, 1, "", "Snap-then-relax rounds")] = 3
    relax: Annotated[int, Range(0, 40, 1, "", "Taubin smoothing iterations per round (0: pure Shrinkwrap)")] = 10
    slack: Annotated[float, Range(1.0, 2.5, 0.05, "", "Start tube radius, times the furthest tissue")] = 1.3


@dataclass(frozen=True)
class FatDeposit:
    thickness: Annotated[float, Range(0.0, 0.03, 0.0005, "m", "Thickness at full adiposity")] = 0.012
    centre: Annotated[float, Range(0.1, 0.9, 0.01, "", "Where along the bone it is thickest")] = 0.5
    width: Annotated[float, Range(0.05, 0.5, 0.01, "", "Spread along the bone (sigma, share of its length)")] = 0.22
    around_deg: Annotated[float, Range(-180.0, 180.0, 1.0, "deg", "Side of the bone (0: the muscle's side)")] = 180.0
    curve: Curve = field(default_factory=lambda: Curve.through((0.0, 0.1), (0.5, 0.5), (1.0, 1.0)))
    """Thickness share against adiposity"""


@dataclass(frozen=True)
class Fat:
    adiposity: Annotated[float, Range(0.0, 1.0, 0.01, "", "Overall body fat, the driver of every deposit's curve")] = (
        0.4
    )
    upper: FatDeposit = field(default_factory=FatDeposit)
    lower: FatDeposit = field(
        default_factory=lambda: FatDeposit(
            thickness=0.009, centre=0.35, curve=Curve.through((0.0, 0.0), (0.6, 0.12), (1.0, 1.0))
        )
    )


@dataclass(frozen=True)
class Growth:
    stage: Annotated[float, Range(0.0, 1.0, 0.01, "", "Growth stage, 1 is fully grown (the stated sizes)")] = 1.0
    curve: Curve = field(
        default_factory=lambda: Curve.through((0.0, 0.3), (0.25, 0.62), (0.6, 0.92), (1.0, 1.0), lo=0.0, hi=1.2)
    )
    """Overall size against stage: fast early, then a plateau"""
    muscle_exponent: Annotated[float, Range(0.5, 2.5, 0.05, "", "Muscle girth grows as size^b")] = 1.4
    fat_exponent: Annotated[float, Range(0.5, 2.5, 0.05, "", "Fat grows as size^b")] = 1.0


@dataclass(frozen=True)
class Limb:
    upper: Bone = field(default_factory=Bone)
    lower: Bone = field(default_factory=lambda: Bone(length=0.26, radius=0.013))
    joint_deg: Annotated[float, Range(0.0, 150.0, 1.0, "deg", "Flexion at the joint")] = 15.0
    muscle: Muscle = field(default_factory=Muscle)
    fat: Fat = field(default_factory=Fat)
    growth: Growth = field(default_factory=Growth)
    skin: Skin = field(default_factory=Skin)
    wrap: Wrap = field(default_factory=Wrap)


def size_factor(p: Limb) -> float:
    return p.growth.curve(p.growth.stage)


def develop(p: Limb) -> Limb:
    """The limb at its growth stage: bones scaled by the growth curve's size, the muscle allometrically."""
    s = size_factor(p)
    mg = allometric(s, p.growth.muscle_exponent)

    def bone(b: Bone) -> Bone:
        return replace(b, length=b.length * s, radius=b.radius * s)

    m = p.muscle
    muscle = replace(m, belly=m.belly * mg, tendon=m.tendon * s, insertion=m.insertion * s, lift=m.lift * s)
    return replace(p, upper=bone(p.upper), lower=bone(p.lower), muscle=muscle)


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
    q = develop(p)
    c = chain(q)
    return (_bone("upper bone", c.root, c.d1, q.upper), _bone("lower bone", c.joint, c.d2, q.lower))


def muscle_ends(p: Limb) -> tuple[F64, F64]:
    c = chain(p)
    side = _frame(c.root, c.d1, np.ones(3)).axes[:, 0]
    side2 = _frame(c.joint, c.d2, np.ones(3)).axes[:, 0]
    start = c.root + c.d1 * p.upper.length * p.muscle.origin + side * p.muscle.lift
    end = c.joint + c.d2 * p.muscle.insertion + side2 * p.muscle.lift * 0.6
    return start, end


@dataclass(frozen=True)
class Bell:
    r"""A bell-shaped radius profile along $t \in [0, 1]$: $r_t + b\,e^{-(t-c)^2/2w^2}\sqrt{\sin \pi t}$."""

    tip: float
    peak: float
    centre: float
    width: float

    def radii(self, t: F64) -> F64:
        bell = np.exp(-((t - self.centre) ** 2) / (2 * self.width**2)) * np.sqrt(np.sin(np.pi * t).clip(0.0))
        return self.tip + self.peak * bell


def belly(name: str, layer: str, start: F64, end: F64, shape: Bell) -> Element:
    """A solid of revolution from `start` to `end` with a bell-shaped profile."""
    girth = max(shape.tip + shape.peak, 1e-6)
    profile = shape.radii(np.linspace(0.0, 1.0, RINGS)) / girth
    profile[[0, -1]] = 0.0
    local, faces = revolution(profile)
    scale = np.array([girth, float(np.linalg.norm(end - start)), girth])
    return Element(name, layer, _frame(start, end - start, scale), local, faces)


def muscles(p: Limb, _: Mapping[str, Resolved]) -> tuple[Element, ...]:
    q = develop(p)
    m = q.muscle
    start, end = muscle_ends(q)
    return (belly("muscle", "muscle", start, end, Bell(m.tendon, m.belly, m.centre, m.width)),)


def fat(p: Limb, _: Mapping[str, Resolved]) -> tuple[Element, ...]:
    """A deposit along each bone, on its side, as thick as its curve says at this adiposity."""
    q = develop(p)
    c = chain(q)
    grown = allometric(size_factor(p), p.growth.fat_exponent)
    out: list[Element] = []
    for name, origin, d, b, dep in (
        ("upper fat", c.root, c.d1, q.upper, p.fat.upper),
        ("lower fat", c.joint, c.d2, q.lower, p.fat.lower),
    ):
        thick = dep.thickness * grown * max(dep.curve(p.fat.adiposity), 0.0)
        axes = _frame(origin, d, np.ones(3)).axes
        a = math.radians(dep.around_deg)
        side = axes[:, 0] * math.cos(a) + axes[:, 2] * math.sin(a)
        offset = side * (b.radius + 0.4 * thick)
        start, end = origin + d * b.length * 0.05 + offset, origin + d * b.length * 0.95 + offset
        out.append(belly(name, "fat", start, end, Bell(0.0005, thick, dep.centre, dep.width)))
    return tuple(out)


def centreline(p: Limb, n: int = SKIN_RINGS) -> tuple[F64, F64]:
    """Points and unit tangents of a smoothing spline through the bone chain (it rounds the joint)."""
    c = chain(p)
    pts = np.vstack([np.linspace(c.root, c.joint, 40, endpoint=False), np.linspace(c.joint, c.tip, 40)])
    spline, _ = interpolate.make_splprep(pts.T, s=p.skin.corner * len(pts) * 1e-2, k=3)
    u = np.linspace(0.0, 1.0, n)
    pos = np.asarray(spline(u), np.float64).T  # the spline gives (ndim, n)
    tan = np.asarray(spline.derivative()(u), np.float64).T
    return pos, tan / np.linalg.norm(tan, axis=1, keepdims=True)


@dataclass(frozen=True)
class Tube:
    """A closed tube around a centreline: rings of vertices at given radii, two caps, and a frame per vertex (out,
    along, around the limb; scaled by the local radius) so a sculpt follows the limb as it bends and grows."""

    verts: F64
    faces: NDArray[np.int32]
    axes: F64
    scale: F64


def tube(pos: F64, tan: F64, radius: F64) -> Tube:
    rings, sections = radius.shape
    out = np.cross(tan, BEND_AXIS)  # in the bend plane, towards +x at rest
    around = np.cross(out, tan)
    ang = np.linspace(0.0, 2 * np.pi, sections, endpoint=False)
    radial = np.cos(ang)[None, :, None] * out[:, None, :] + np.sin(ang)[None, :, None] * around[:, None, :]
    ring_verts = pos[:, None, :] + radial * radius[:, :, None]
    caps = np.vstack([pos[0] - tan[0] * radius[0].mean() * 0.6, pos[-1] + tan[-1] * radius[-1].mean() * 0.6])
    tangent = np.repeat(tan, sections, axis=0)
    rad = radial.reshape(-1, 3)
    axes = np.stack([rad, tangent, np.cross(rad, tangent)], axis=2)
    cap_axes = np.stack([np.column_stack([-tan[0], *_perp(tan[0])]), np.column_stack([tan[-1], *_perp(tan[-1])])])
    sizes = np.concatenate([radius.reshape(-1), [radius[0].mean(), radius[-1].mean()]])
    return Tube(
        np.vstack([ring_verts.reshape(-1, 3), caps]),
        _tube_faces(rings, sections),
        np.concatenate([axes, cap_axes]),
        np.repeat(sizes[:, None], 3, axis=1),
    )


def _tissue(done: Mapping[str, Resolved]) -> list[Resolved]:
    return [r for r in done.values() if r.element.layer in TISSUE]


def skin(p: Limb, done: Mapping[str, Resolved]) -> tuple[Element, ...]:
    """The sweep: at each ring and angle, the furthest tissue in that direction, smoothed, plus a thickness."""
    pos, tan = centreline(develop(p))
    out = np.cross(tan, BEND_AXIS)
    around = np.cross(out, tan)
    # every tissue vertex to its nearest centreline point, then to an angle and a distance in that ring's plane
    tissue = np.vstack([r.verts for r in _tissue(done)])
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
    t = tube(pos, tan, radius + p.skin.thickness)
    return (Element("skin", "skin", Frame.world(), t.verts, t.faces, t.axes, t.scale),)


def wrap_skin(p: Limb, done: Mapping[str, Resolved]) -> tuple[Element, ...]:
    """Shrinkwrap-style: a loose tube around everything, then rounds of snapping every vertex to the nearest point
    on the tissue plus `offset` along that surface's normal (Shrinkwrap, Nearest Surface Point, Offset) and relaxing
    the mesh (Taubin smoothing, which does not shrink it, standing in for Corrective Smooth). libigl finds the
    nearest points (an AABB tree; trimesh's own query was 25 times slower here), trimesh smooths."""
    w = p.wrap
    if not w.enabled:
        return ()
    pos, tan = centreline(develop(p), WRAP_RINGS)
    parts = _tissue(done)
    # trimesh is untyped: its results are cast to what it documents
    concatenate = cast(Callable[[list[trimesh.Trimesh]], trimesh.Trimesh], trimesh.util.concatenate)  # pyright: ignore[reportUnknownMemberType]
    tissue = concatenate([trimesh.Trimesh(r.verts, r.faces, process=False) for r in parts])
    tissue_verts = np.asarray(tissue.vertices, np.float64)
    nearest = np.asarray(KDTree(pos).query(tissue_verts)[1], np.intp)
    reach = float(np.linalg.norm(tissue_verts - pos[nearest], axis=1).max())
    t = tube(pos, tan, np.full((WRAP_RINGS, WRAP_SECTIONS), reach * w.slack))
    mesh = trimesh.Trimesh(t.verts, t.faces, process=False)
    nearest_on = cast(
        Callable[[F64, F64, NDArray[np.int64]], tuple[F64, NDArray[np.int64], F64]], igl.point_mesh_squared_distance
    )
    taubin = cast(Callable[..., None], trimesh.smoothing.filter_taubin)  # pyright: ignore[reportUnknownMemberType]
    normals = np.asarray(tissue.face_normals, np.float64)
    verts = t.verts
    faces = np.asarray(tissue.faces, np.int64)
    for _ in range(w.passes):
        _, tri, closest = nearest_on(verts, tissue_verts, faces)
        verts = np.asarray(closest, np.float64) + normals[np.asarray(tri, np.intp)] * w.offset
        if w.relax:
            mesh.vertices = verts
            taubin(mesh, iterations=w.relax)
            verts = np.asarray(mesh.vertices, np.float64)
    return (Element("wrap skin", "wrap skin", Frame.world(), verts, t.faces, t.axes, t.scale),)


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
    """The elbow at `w`: the upper bone's (grown) length is its projection on the bone, over the growth size."""
    c = chain(develop(p))
    length = float((w - c.root) @ c.d1) / size_factor(p)
    return replace(p, upper=replace(p.upper, length=clamp(p.upper, "length", length)))


def _drag_wrist(p: Limb, w: F64) -> Limb:
    c = chain(develop(p))
    v = w - c.joint
    v -= BEND_AXIS * float(v @ BEND_AXIS)
    angle = math.degrees(math.atan2(float(v @ np.array([1.0, 0.0, 0.0])), float(v @ c.d1)))
    lower = replace(p.lower, length=clamp(p.lower, "length", float(np.linalg.norm(v)) / size_factor(p)))
    return replace(p, joint_deg=clamp(p, "joint_deg", angle), lower=lower)


def _turn_elbow(p: Limb, rotation: F64) -> Limb:
    """Turning the elbow's handle about the bend axis bends the joint by as much (the lower bone turns with it)."""
    d2 = chain(develop(p)).d2
    return replace(p, joint_deg=clamp(p, "joint_deg", p.joint_deg + _bend_deg(d2, rotation @ d2)))


def _bend_deg(before: F64, after: F64) -> float:
    """How far `before` turned into `after` about the bend axis, in the joint's flexion sense (towards +x)."""
    return -math.degrees(math.atan2(float(np.cross(before, after) @ BEND_AXIS), float(before @ after)))


def bind(p: Limb, element: str, matrix: F64) -> Limb | None:
    """A bone moved by hand as parameters: scaled along itself (its length), across (its radius), and for the lower
    bone, turned about the joint (the joint's angle). Moving a bone sideways has no parameter; it springs back."""
    if element not in ("upper bone", "lower bone"):
        return None
    m3 = matrix[:3, :3]
    upper = element == "upper bone"
    c = chain(develop(p))
    d = c.d1 if upper else c.d2
    across = _frame(c.root, d, np.ones(3)).axes[:, 0]
    b = p.upper if upper else p.lower
    b = replace(
        b,
        length=clamp(b, "length", b.length * float(np.linalg.norm(m3 @ d))),
        radius=clamp(b, "radius", b.radius * float(np.linalg.norm(m3 @ across))),
    )
    if upper:
        return replace(p, upper=b)
    return replace(p, lower=b, joint_deg=clamp(p, "joint_deg", p.joint_deg + _bend_deg(d, m3 @ d)))


@dataclass(frozen=True)
class LimbModel:
    name: str = "limb"
    layer_colours: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {"wrap skin": (0.55, 0.68, 0.85, 0.45)}
    )

    def defaults(self) -> Limb:
        return Limb()

    @property
    def stages(self) -> tuple[Stage[Limb], ...]:
        return (
            Stage("scaffold", bones),
            Stage("structure", muscles),
            Stage("fat", fat),
            Stage("surface", skin),
            Stage("wrap", wrap_skin),
        )

    @property
    def composites(self) -> tuple[Composite[Limb], ...]:
        return (
            Composite("flex", Range(0.0, 1.0, 0.01, "", "Bend the joint and bunch the muscle"), flex),
            Composite("size", Range(-1.0, 1.0, 0.01, "", "Scale the whole limb by 2^size"), size),
        )

    def handles(self, p: Limb) -> tuple[Handle[Limb], ...]:
        return (
            Handle(
                "elbow",
                lambda q: chain(develop(q)).joint,
                _drag_elbow,
                (("upper", "length"), ("joint_deg",)),
                _turn_elbow,
            ),
            Handle("wrist", lambda q: chain(develop(q)).tip, _drag_wrist, (("joint_deg",), ("lower", "length"))),
        )

    def bind(self, p: Limb, element: str, matrix: F64) -> Limb | None:
        return bind(p, element, matrix)


MODEL = LimbModel()
