"""Run inside Blender (scripts/gui.py): register proctune from this checkout and open the project given after
``--``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_env

blender_env.setup()
from proctune.blender import addon  # noqa: E402


def remove_default_cube() -> None:
    """Blender started without a file shows its default cube in the middle of the model. Remove it, but only the
    untouched default (named Cube, the 8-vertex default mesh, at the origin, unscaled): a cube of your own stays."""
    import bpy  # noqa: PLC0415

    cube = bpy.data.objects.get("Cube")
    if bpy.data.filepath or cube is None or cube.type != "MESH":
        return
    me = cube.data
    default = len(me.vertices) == 8 and tuple(cube.location) == (0, 0, 0) and tuple(cube.scale) == (1, 1, 1)
    if default and me.users == 1:
        bpy.data.objects.remove(cube)
        bpy.data.meshes.remove(me)


addon.register()
args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if args:
    remove_default_cube()
    addon.open_reporting(Path(args[0]).resolve())
