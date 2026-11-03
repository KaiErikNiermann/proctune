"""Inside Blender, at start (`proctune open`): register the add-on, clear Blender's default cube, open the project
(a failure shows in the panel, with what to do), and split the view to compare the model with its surface.

    blender --python-expr "...; from proctune.blender.launch import main; main()" -- PROJECT [--no-compare]
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import bpy

from proctune import config, deps


def remove_default_cube() -> None:
    """Blender started without a file shows its default cube in the middle of the model. Remove it, but only the
    untouched default (named Cube, the 8-vertex default mesh, at the origin, unscaled): a cube of your own stays."""
    cube = bpy.data.objects.get("Cube")
    if bpy.data.filepath or cube is None or cube.type != "MESH":
        return
    me = cast(bpy.types.Mesh, cube.data)
    default = len(me.vertices) == 8 and tuple(cube.location) == (0, 0, 0) and tuple(cube.scale) == (1, 1, 1)
    if default and me.users == 1:
        bpy.data.objects.remove(cube)
        bpy.data.meshes.remove(me)


def main() -> None:
    from proctune.blender import addon  # noqa: PLC0415 -- after the paths below are in place

    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    cfg = config.load()
    deps.add_paths(cfg.paths)
    if cfg.packages is not None:
        deps.ensure_on_path(cfg.packages)
    addon.register()
    projects = [a for a in args if not a.startswith("--")]
    if not projects:
        return
    remove_default_cube()
    addon.open_reporting(Path(projects[0]).resolve())
    if "--no-compare" not in args and not bpy.app.background:
        # once the window exists: the model on the left, its surface on the right

        def open_compare() -> None:
            addon.compare()

        bpy.app.timers.register(open_compare, first_interval=0.5)
