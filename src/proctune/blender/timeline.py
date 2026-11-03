"""Parameters over time: keyframe any proctune slider (or composite) and the model follows the timeline.

Blender evaluates the keyframes on a frame change without calling the properties' update callbacks, so a
``frame_change_post`` handler copies the animated values into the session and rebuilds. On save the animated values
are sampled once per frame into the project's tracks, so a headless run (`Session.at_frame`) gives the same model at
every frame without Blender's F-curves.
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from typing import Any

import bpy
from bpy_extras import anim_utils

from proctune.core.params import Path as ParamPath
from proctune.core.session import COMPOSITE, Track

PREFIX = "proctune_params."


def animated(keys: dict[str, ParamPath]) -> dict[str, bpy.types.FCurve]:
    """Property key to its F-curve, for every keyframed proctune property of the scene."""
    sc = bpy.context.scene
    ad = sc.animation_data
    if ad is None or ad.action is None:
        return {}
    bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    if bag is None:
        return {}
    out: dict[str, bpy.types.FCurve] = {}
    for fc in bag.fcurves:
        key = fc.data_path.removeprefix(PREFIX)
        if fc.data_path.startswith(PREFIX) and key in keys:
            out[key] = fc
    return out


def name_of(path: ParamPath) -> str:
    """The track name of a property path: dotted, or ``composite:NAME``."""
    return COMPOSITE + path[1] if path[0] == "@composite" else ".".join(path)


def current_values(keys: dict[str, ParamPath]) -> dict[str, Any]:
    """The animated properties' values at the current frame, by track name."""
    props = bpy.context.scene.proctune_params
    return {name_of(keys[k]): getattr(props, k) for k in animated(keys)}


def bake(keys: dict[str, ParamPath]) -> dict[str, Track]:
    """Every animated property sampled at each frame of the scene's range."""
    sc = bpy.context.scene
    frames = range(sc.frame_start, sc.frame_end + 1)
    return {
        name_of(keys[k]): tuple((float(f), float(fc.evaluate(f))) for f in frames) for k, fc in animated(keys).items()
    }
