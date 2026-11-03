"""proctune: live sliders, handles and regeneration-proof sculpting for procedural models in Blender.

`proctune.core` has no Blender dependency (models, sessions and projects work in plain Python); `proctune.blender`
is the add-on. As a Blender extension this package's `register`/`unregister` load the add-on.
"""


def register() -> None:
    from proctune.blender import addon  # noqa: PLC0415 -- bpy only exists inside Blender

    addon.register()


def unregister() -> None:
    from proctune.blender import addon  # noqa: PLC0415

    addon.unregister()
