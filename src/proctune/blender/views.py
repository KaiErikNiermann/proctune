"""Two 3D views side by side with their navigation kept in step: the model as built on the left (every layer, in its
colours), its surface on the right (only the surface layers, opaque in one colour, so the real outer shape shows).

Blender has no view sync between two 3D views, so a draw callback does it: whenever one of the pair is drawn and its
view (location, rotation, distance, projection) differs from what was last synced, that view is the one being moved,
and the other copies it and redraws. The right view hides the inner layers with Blender's per-viewport collection
visibility (Local Collections), which leaves the left view as it is.
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportOptionalMemberAccess=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import bpy

from proctune.blender import scene

#: The right view's colour for the surface (a skin tone)
SURFACE_COLOUR = (0.85, 0.66, 0.55)


@dataclass
class Pair:
    left: int = 0
    right: int = 0
    """The two areas, by pointer (areas are not kept between draws)"""
    last: tuple[Any, ...] = ()
    """The view as last synced"""
    handle: Any = None
    syncing: bool = False
    surface: tuple[str, ...] = field(default_factory=tuple)


PAIR = Pair()


def _areas() -> dict[int, bpy.types.Area]:
    out: dict[int, bpy.types.Area] = {}
    wm = bpy.context.window_manager
    for w in wm.windows if wm else ():
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                out[a.as_pointer()] = a
    return out


def region_3d(area: bpy.types.Area) -> bpy.types.RegionView3D:
    return cast(bpy.types.RegionView3D, cast(bpy.types.SpaceView3D, area.spaces[0]).region_3d)


def _state(r: bpy.types.RegionView3D) -> tuple[Any, ...]:
    return (
        tuple(round(v, 6) for v in r.view_location),
        tuple(round(v, 6) for v in r.view_rotation),
        round(r.view_distance, 6),
        r.view_perspective,
    )


def _copy(src: bpy.types.RegionView3D, dst: bpy.types.RegionView3D) -> None:
    dst.view_location = src.view_location.copy()
    dst.view_rotation = src.view_rotation.copy()
    dst.view_distance = src.view_distance
    if dst.view_perspective != src.view_perspective:
        dst.view_perspective = src.view_perspective


def _sync() -> None:
    """Draw callback (every 3D view): keep the pair's views equal, the moved one leading."""
    if PAIR.syncing:
        return
    areas = _areas()
    left, right = areas.get(PAIR.left), areas.get(PAIR.right)
    if left is None or right is None:
        return
    ls, rs = _state(region_3d(left)), _state(region_3d(right))
    if ls == rs:
        PAIR.last = ls
        return
    src, dst = (left, right) if ls != PAIR.last else (right, left)
    PAIR.syncing = True
    try:
        _copy(region_3d(src), region_3d(dst))
        PAIR.last = _state(region_3d(src))
        dst.tag_redraw()
    finally:
        PAIR.syncing = False


def _surface_only(area: bpy.types.Area, surface: tuple[str, ...]) -> None:
    space = cast(bpy.types.SpaceView3D, area.spaces[0])
    space.show_region_ui = False  # the proctune panel is in the left view's sidebar
    space.use_local_collections = True
    shading = space.shading
    shading.type = "SOLID"
    shading.color_type = "SINGLE"
    shading.single_color = SURFACE_COLOUR
    view_layer = bpy.context.view_layer
    root = view_layer.layer_collection.children.get(scene.ROOT)
    if root is None:
        return
    for lc in root.children:
        _set_local(area, lc, lc.name.removeprefix(f"{scene.ROOT}: ") in surface)


def _set_local(area: bpy.types.Area, lc: bpy.types.LayerCollection, visible: bool) -> None:
    """Show or hide a collection in this viewport only (Local Collections), through the operator Blender uses."""
    index = _collection_index(lc)
    if index is None:
        return
    region = next(r for r in area.regions if r.type == "WINDOW")
    with bpy.context.temp_override(area=area, region=region):
        if bool(lc.visible_get()) != visible:
            bpy.ops.object.hide_collection(collection_index=index, toggle=True)


def _collection_index(target: bpy.types.LayerCollection) -> int | None:
    """The layer collection's index as `object.hide_collection` counts them (depth-first, the scene's root is 0)."""
    i = 0

    def walk(lc: bpy.types.LayerCollection) -> int | None:
        nonlocal i
        if lc == target:
            return i
        for child in lc.children:
            i += 1
            if (found := walk(child)) is not None:
                return found
        return None

    return walk(bpy.context.view_layer.layer_collection)


def enabled() -> bool:
    areas = _areas()
    return PAIR.handle is not None and PAIR.left in areas and PAIR.right in areas


def compare(surface: tuple[str, ...]) -> bool:
    """Split the (largest) 3D view into the pair; the right half shows only the `surface` layers, opaque. Returns
    whether there is a pair (no 3D view, e.g. in the background: no)."""
    areas = sorted(_areas().values(), key=lambda a: a.width * a.height, reverse=True)
    if not areas:
        return False
    if enabled():
        right = _areas()[PAIR.right]
    else:
        left = areas[0]
        before = set(_areas())
        region = next(r for r in left.regions if r.type == "WINDOW")
        with bpy.context.temp_override(window=bpy.context.window_manager.windows[0], area=left, region=region):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
        new = [a for p, a in _areas().items() if p not in before]
        if not new:
            return False
        right = new[0]
        # which of the two is on the right
        if right.x < left.x:
            left, right = right, left
        PAIR.left, PAIR.right = left.as_pointer(), right.as_pointer()
        _copy(region_3d(left), region_3d(right))
        PAIR.last = _state(region_3d(left))
    PAIR.surface = surface
    _surface_only(right, surface)
    if PAIR.handle is None:
        PAIR.handle = bpy.types.SpaceView3D.draw_handler_add(_sync, (), "WINDOW", "POST_PIXEL")
    return True


def refresh() -> None:
    """After a rebuild (new layer collections may exist): keep the right view showing only the surface."""
    if enabled():
        _surface_only(_areas()[PAIR.right], PAIR.surface)


def stop() -> None:
    """Stop syncing (the views stay as they are)."""
    if PAIR.handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(PAIR.handle, "WINDOW")
    PAIR.handle = None
    PAIR.left = PAIR.right = 0
