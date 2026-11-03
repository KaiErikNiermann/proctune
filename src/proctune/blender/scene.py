"""Writing a session's result into the Blender scene and reading hand edits back out of it.

Each element is one object (named as the element) in a collection per layer under ``proctune``. A movable element's
object is parented to a hidden empty placed where the element was generated (its frame, without scale), its mesh is
written in that frame, and its own transform is its hand move: Location, Rotation and Scale read 0, 0 and 1 at the
generated place, and Scale 1.2 means 1.2 times the generated size along that axis. A surface without a place of its
own (a skin) is written in world space, unparented. A transform that differs from what was written is a hand move, a
vertex that differs is a sculpt. Handles are empties.
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import bpy
import numpy as np
from mathutils import Matrix

from proctune.core import Result, Session
from proctune.core.element import F64
from proctune.errors import ProctuneError

ROOT = "proctune"
HANDLES = "proctune handles"
ELEMENT_KEY = "proctune_element"
HANDLE_KEY = "proctune_handle"
FRAMES = "proctune frames"
FRAME_KEY = "proctune_frame"
#: Viewport colours per layer name (RGBA); other layers get grey
COLOURS: dict[str, tuple[float, float, float, float]] = {
    "bone": (0.91, 0.86, 0.77, 1.0),
    "muscle": (0.71, 0.27, 0.24, 1.0),
    "fat": (0.91, 0.76, 0.35, 1.0),
    "skin": (0.85, 0.66, 0.55, 0.45),
}
TOLERANCE = 1e-7


@dataclass
class Written:
    """What was last written, to tell hand edits from our own writes."""

    verts: dict[str, F64] = field(default_factory=dict[str, F64])
    """Object-space vertices"""
    basis: dict[str, F64] = field(default_factory=dict[str, F64])
    """The object's own transform (relative to its frame)"""
    frames: dict[str, F64] = field(default_factory=dict[str, F64])
    """Where the frame empty was put (4x4, world); identity for unparented elements"""
    handles: dict[str, F64] = field(default_factory=dict[str, F64])
    handle_turns: dict[str, F64] = field(default_factory=dict[str, F64])
    """Each handle's rotation already applied (3x3): a turn is relative, and a rotation in progress is applied in
    steps, so only the change since the last step counts"""


def _collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
    if c.name not in parent.children:
        parent.children.link(c)
    return c


def root() -> bpy.types.Collection:
    return _collection(ROOT, bpy.context.scene.collection)


def layer_collection(layer: str) -> bpy.types.Collection:
    return _collection(f"{ROOT}: {layer}", root())


def _material(layer: str, colours: dict[str, tuple[float, float, float, float]]) -> bpy.types.Material:
    """A viewport material per layer: the model's colour for it (`layer_colours`), else proctune's default."""
    name = f"{ROOT} {layer}"
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = colours.get(layer) or COLOURS.get(layer, (0.6, 0.6, 0.6, 1.0))
    return m


def element_objects() -> dict[str, bpy.types.Object]:
    return {str(o[ELEMENT_KEY]): o for o in bpy.data.objects if ELEMENT_KEY in o}


def handle_objects() -> dict[str, bpy.types.Object]:
    return {str(o[HANDLE_KEY]): o for o in bpy.data.objects if HANDLE_KEY in o}


def read_verts(obj: bpy.types.Object) -> F64:
    """The object's vertices in object space (synced from edit mode first if it is being edited)."""
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    me = cast(bpy.types.Mesh, obj.data)
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def world_verts(obj: bpy.types.Object) -> F64:
    """The object's vertices in world space, from its frame and its own transform (both as set, no depsgraph lag)."""
    m = np.array(obj.matrix_basis, np.float64)
    if obj.parent is not None:
        m = np.array(obj.parent.matrix_world, np.float64) @ m
    v = read_verts(obj)
    return v @ m[:3, :3].T + m[:3, 3]


def _world(obj: bpy.types.Object) -> F64:
    """The object's own transform (relative to its frame, if it has one), from its location/rotation/scale:
    matrix_world lags until the next depsgraph update."""
    return np.array(obj.matrix_basis, np.float64)


def frame_objects() -> dict[str, bpy.types.Object]:
    return {str(o[FRAME_KEY]): o for o in bpy.data.objects if FRAME_KEY in o}


def _frame_object(name: str) -> bpy.types.Object:
    obj = frame_objects().get(name)
    if obj is None:
        obj = bpy.data.objects.new(f"{name} frame", None)
        obj.empty_display_type = "PLAIN_AXES"
        obj.empty_display_size = 0.01
        obj[FRAME_KEY] = name
        _collection(FRAMES, root()).objects.link(obj)
        obj.hide_select = True
    obj.hide_set(True)
    return obj


def write(
    result: Result,
    written: Written,
    skip: frozenset[str] = frozenset(),
    colours: dict[str, tuple[float, float, float, float]] | None = None,
) -> None:
    """Create or update an object per element; remove objects of elements that no longer exist."""
    objs = element_objects()
    for name, r in result.elements.items():
        if name in skip:
            continue
        e = r.element
        obj = objs.get(name)
        if obj is None:
            me = bpy.data.meshes.new(name)
            obj = bpy.data.objects.new(name, me)
            obj[ELEMENT_KEY] = name
            layer_collection(e.layer).objects.link(obj)
            me.materials.append(_material(e.layer, colours or {}))
        me = cast(bpy.types.Mesh, obj.data)
        if obj.mode == "EDIT":
            continue  # its edit mesh would overwrite this; it is written after leaving edit mode
        if e.tweakable:
            frame = e.frame.rigid()
            parent = _frame_object(name)
            parent.matrix_world = Matrix(frame.tolist())
            d = r.edit.displacement if r.edit.displacement is not None and not r.stale else 0.0
            verts = (e.local + d) * e.frame.scale
            obj.parent = parent
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix(r.edit.tweak.matrix(e.frame.scale).tolist())
        else:
            frame = np.eye(4)
            verts = r.verts
            obj.parent = None
            obj.matrix_basis = Matrix.Identity(4)
        if len(me.vertices) == len(verts) and len(me.polygons) == len(e.faces):
            me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        else:
            me.clear_geometry()
            me.from_pydata(verts.tolist(), [], e.faces.tolist())
            me.shade_smooth()
        me.update()
        written.verts[name] = verts.astype(np.float32).astype(np.float64)  # as stored in the mesh
        written.basis[name] = np.array(obj.matrix_basis, np.float64)  # as Blender keeps it (location/rotation/scale)
        written.frames[name] = frame
    for name, obj in objs.items():
        if name not in result.elements:
            remove(obj)
            for d in (written.verts, written.basis, written.frames):
                d.pop(name, None)


def remove(obj: bpy.types.Object) -> None:
    """Remove an element's object, its mesh and its frame (removing only the object would leave the others behind,
    unused, in the file)."""
    data = obj.data
    frame = frame_objects().get(str(obj.get(ELEMENT_KEY, "")))
    bpy.data.objects.remove(obj)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)
    if frame is not None:
        bpy.data.objects.remove(frame)


def write_handles(session: Session[object], written: Written, skip: frozenset[str] = frozenset()) -> None:
    objs = handle_objects()
    coll = _collection(HANDLES, root())
    for h in session.handles():
        obj = objs.get(h.name)
        if obj is None:
            obj = bpy.data.objects.new(f"handle: {h.name}", None)
            obj.empty_display_type = "SPHERE"
            obj.empty_display_size = 0.012
            obj.show_in_front = True
            obj[HANDLE_KEY] = h.name
            coll.objects.link(obj)
        # a handle that means nothing when turned or scaled is locked against it, so R visibly does nothing
        obj.lock_rotation = (h.turn is None,) * 3
        obj.lock_scale = (True, True, True)
        if h.name in skip:
            continue
        pos = h.position(session.effective)
        obj.location = pos.tolist()
        obj.rotation_euler = (0.0, 0.0, 0.0)
        written.handles[h.name] = np.asarray(pos, np.float64).copy()
        written.handle_turns[h.name] = np.eye(3)


def turned_handles(written: Written) -> dict[str, F64]:
    """Handles turned since their rotation was last applied: name to that further rotation (3x3, world). Records
    the rotation as applied."""
    out: dict[str, F64] = {}
    for name, obj in handle_objects().items():
        now = np.array(obj.rotation_euler.to_matrix(), np.float64)
        done = written.handle_turns.get(name, np.eye(3))
        if not np.allclose(now, done, atol=1e-6):
            out[name] = now @ done.T
            written.handle_turns[name] = now
    return out


def moved_handles(written: Written) -> dict[str, F64]:
    """Handles whose empties are not where they were placed: name to world position."""
    out: dict[str, F64] = {}
    for name, obj in handle_objects().items():
        pos = np.array(obj.location, np.float64)  # unparented: location is world; matrix_world lags a depsgraph update
        last = written.handles.get(name)
        if last is not None and np.linalg.norm(pos - last) > 1e-6:
            out[name] = pos
    return out


@dataclass(frozen=True)
class HandEdits:
    moved: dict[str, tuple[F64, F64]]
    """Element name to its object's transform as written and as it is now (relative to its frame)"""
    sculpted: dict[str, F64]
    """Element name to its vertices (object space), where they differ from what was written"""

    def __bool__(self) -> bool:
        return bool(self.moved or self.sculpted)


def hand_edits(written: Written) -> HandEdits:
    moved: dict[str, tuple[F64, F64]] = {}
    sculpted: dict[str, F64] = {}
    for name, obj in element_objects().items():
        before = written.basis.get(name)
        if before is None:
            continue
        now = _world(obj)
        if not np.allclose(now, before, atol=1e-6):
            moved[name] = (before, now)
        v = read_verts(obj)
        last = written.verts[name]
        if v.shape != last.shape or np.abs(v - last).max() > 1e-6:
            sculpted[name] = v
    return HandEdits(moved, sculpted)


def apply_hand_edits(session: Session[object], edits: HandEdits, written: Written) -> list[ProctuneError]:
    """Moves first (they change what a sculpt is measured against), then sculpts. Each is captured on its own: one
    that cannot be kept is returned as an error and the next write puts that object back, so it is reported once
    instead of failing every rebuild after it."""
    errors: list[ProctuneError] = []
    for name, (before, now) in edits.moved.items():
        frame = written.frames.get(name, np.eye(4))
        world_move = frame @ now @ np.linalg.inv(before) @ np.linalg.inv(frame)
        try:
            session.capture_transform(name, world_move)
        except ProctuneError as e:
            errors.append(e)
    if edits.moved:
        session.evaluate()
    for name, v in edits.sculpted.items():
        to_world = written.frames.get(name, np.eye(4)) @ edits.moved.get(name, (None, written.basis[name]))[1]
        try:
            session.capture_sculpt(name, v @ to_world[:3, :3].T + to_world[:3, 3])
        except ProctuneError as e:
            errors.append(e)
    return errors
