"""Curve parameters edited with Blender's curve widget.

Blender's curve editor (``CurveMapping``) cannot be an add-on property, so each curve lives on a Float Curve node in a
node group no material uses (``proctune curves``), named after the parameter, and the panel draws that node's widget.
Editing the widget fires no update, so a timer compares the points with the session's curves. The widget only edits
points: the curve is evaluated by `proctune.core.curves.Curve` (PCHIP) everywhere, and the widget draws the points with
Blender's auto-clamped handles, which also do not overshoot, so the drawn line is close to the evaluated one.
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportOptionalMemberAccess=false

from __future__ import annotations

from typing import Any, cast

import bpy
import numpy as np

from proctune.core import Curve, Session, specs
from proctune.core import params as P
from proctune.core.params import Path as ParamPath

GROUP = "proctune curves"
#: Points closer than this in x are moved apart (a curve needs strictly increasing x)
MIN_DX = 1e-3


def _group() -> bpy.types.NodeTree:
    g = bpy.data.node_groups.get(GROUP)
    if g is None:
        g = bpy.data.node_groups.new(GROUP, "ShaderNodeTree")
        g.use_fake_user = True  # kept in the .blend although nothing uses it
    return cast(bpy.types.NodeTree, g)


def curve_specs(session: Session[Any]) -> dict[str, ParamPath]:
    return {s.key: s.path for s in specs(session.base) if s.kind == "curve"}


def node(key: str) -> bpy.types.ShaderNodeFloatCurve:
    g = _group()
    n = g.nodes.get(key)
    if n is None:
        n = g.nodes.new("ShaderNodeFloatCurve")
        n.name = key
    return cast(bpy.types.ShaderNodeFloatCurve, n)


def write(session: Session[Any]) -> None:
    """Put the session's curves into their widgets."""
    for key, path in curve_specs(session).items():
        c = cast(Curve, P.get(session.base, path))
        m = node(key).mapping
        cm = m.curves[0]
        while len(cm.points) > len(c.points):
            cm.points.remove(cm.points[-1])
        while len(cm.points) < len(c.points):
            cm.points.new(0.5, 0.5)
        for pt, (x, y) in zip(cm.points, c.points, strict=True):
            pt.location = (x, y)
            pt.handle_type = "AUTO_CLAMPED"
        m.use_clip = True
        m.clip_min_x, m.clip_max_x = 0.0, 1.0
        m.clip_min_y, m.clip_max_y = c.lo, c.hi
        m.update()


def read(key: str, like: Curve) -> Curve:
    """The widget's points as a curve (sorted, and nudged apart where two share an x)."""
    pts = sorted((float(p.location[0]), float(p.location[1])) for p in node(key).mapping.curves[0].points)
    out: list[tuple[float, float]] = []
    for x, y in pts:
        out.append((max(x, out[-1][0] + MIN_DX) if out else x, y))
    return Curve(tuple(out), like.lo, like.hi)


def changed(session: Session[Any]) -> dict[ParamPath, Curve]:
    """Curves whose widgets differ from the session."""
    out: dict[ParamPath, Curve] = {}
    for key, path in curve_specs(session).items():
        now = cast(Curve, P.get(session.base, path))
        drawn = read(key, now)
        if len(drawn.points) != len(now.points) or not np.allclose(drawn.points, now.points, atol=1e-6):
            out[path] = drawn
    return out


def draw(layout: bpy.types.UILayout, session: Session[Any]) -> None:
    for key, path in curve_specs(session).items():
        box = layout.box()
        box.label(text=" / ".join(path), icon="FCURVE")
        box.template_curve_mapping(node(key), "mapping")
