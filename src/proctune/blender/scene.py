"""Writing a session's result into the Blender scene and reading hand edits back out of it.

Each element is one object (named as the element) in a collection per layer under ``proctune``; the object's
transform is kept at identity and the vertices are written in world coordinates, so a transform on the object is a
hand move and a vertex that differs from what was written is a sculpt. Handles are empties.
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

ROOT = "proctune"
HANDLES = "proctune handles"
ELEMENT_KEY = "proctune_element"
HANDLE_KEY = "proctune_handle"
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
    handles: dict[str, F64] = field(default_factory=dict[str, F64])


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


def _material(layer: str) -> bpy.types.Material:
    name = f"{ROOT} {layer}"
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = COLOURS.get(layer, (0.6, 0.6, 0.6, 1.0))
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


def _world(obj: bpy.types.Object) -> F64:
    """The (unparented) object's transform, from its own location/rotation/scale: matrix_world lags until the next
    depsgraph update."""
    return np.array(obj.matrix_basis, np.float64)


def write(result: Result, written: Written, skip: frozenset[str] = frozenset()) -> None:
    """Create or update an object per element; remove objects of elements that no longer exist."""
    objs = element_objects()
    for name, r in result.elements.items():
        if name in skip:
            continue
        obj = objs.get(name)
        verts = r.verts
        faces = r.faces
        if obj is None:
            me = bpy.data.meshes.new(name)
            obj = bpy.data.objects.new(name, me)
            obj[ELEMENT_KEY] = name
            layer_collection(r.element.layer).objects.link(obj)
            me.materials.append(_material(r.element.layer))
        me = cast(bpy.types.Mesh, obj.data)
        if obj.mode == "EDIT":
            continue  # its edit mesh would overwrite this; it is written after leaving edit mode
        if len(me.vertices) == len(verts) and len(me.polygons) == len(faces):
            me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        else:
            me.clear_geometry()
            me.from_pydata(verts.tolist(), [], faces.tolist())
            me.shade_smooth()
        me.update()
        obj.matrix_world = Matrix.Identity(4)
        written.verts[name] = verts.astype(np.float32).astype(np.float64)  # as stored in the mesh
    for name, obj in objs.items():
        if name not in result.elements:
            remove(obj)
            written.verts.pop(name, None)


def remove(obj: bpy.types.Object) -> None:
    """Remove an object and its mesh (removing only the object would leave the mesh behind, unused, in the file)."""
    data = obj.data
    bpy.data.objects.remove(obj)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)


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
        if h.name in skip:
            continue
        pos = h.position(session.effective)
        obj.location = pos.tolist()
        written.handles[h.name] = np.asarray(pos, np.float64).copy()


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
    moved: dict[str, F64]
    """Element name to its object's 4x4 world transform, where it is not identity"""
    sculpted: dict[str, F64]
    """Element name to its vertices (object space), where they differ from what was written"""

    def __bool__(self) -> bool:
        return bool(self.moved or self.sculpted)


def hand_edits(written: Written) -> HandEdits:
    moved: dict[str, F64] = {}
    sculpted: dict[str, F64] = {}
    for name, obj in element_objects().items():
        m = _world(obj)
        if not np.allclose(m, np.eye(4), atol=TOLERANCE):
            moved[name] = m
        last = written.verts.get(name)
        if last is None:
            continue
        v = read_verts(obj)
        if v.shape != last.shape or np.abs(v - last).max() > 1e-6:
            sculpted[name] = v
    return HandEdits(moved, sculpted)


def apply_hand_edits(session: Session[object], edits: HandEdits) -> None:
    """Move tweaks first (they change what a sculpt is measured against), then sculpts. A sculpted object that was
    also moved is read in world space: its transform applied to its vertices."""
    for name, m in edits.moved.items():
        session.capture_transform(name, m)
    if edits.moved:
        session.evaluate()
    for name, v in edits.sculpted.items():
        m = edits.moved.get(name, np.eye(4))
        world = v @ m[:3, :3].T + m[:3, 3]
        session.capture_sculpt(name, world)
