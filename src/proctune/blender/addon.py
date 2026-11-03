"""The Blender side: a sidebar panel generated from the model's parameters, live rebuilds, handles, and hand edits
captured back into the session.

The loop: a slider (or composite, or a dragged handle) changes the session's parameters and schedules a rebuild. A
rebuild first captures what was done by hand since the last one (objects moved, vertices sculpted, in any mode),
then evaluates the model and writes every element. Hand edits are noticed without a button: a moved object or a
sculpt stroke schedules a rebuild once no modal operator (a grab, a stroke) is running, so the skin follows a
sculpted muscle stroke by stroke.
"""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false, reportIncompatibleVariableOverride=false, reportUntypedFunctionDecorator=false

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from proctune.blender import scene
from proctune.core import Session, Spec, specs
from proctune.core.params import Path as ParamPath
from proctune.core.session import describe

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
else:
    OperatorReturnItems = str

type Ret = set[Literal[OperatorReturnItems]]
log = logging.getLogger("proctune")
PROJECT_KEY = "proctune_project"
#: Wait this long after the last change before rebuilding (s); a running modal operator postpones it further
DEBOUNCE_S = 0.03


@dataclass
class State:
    session: Session[Any] | None = None
    path: Path | None = None
    written: scene.Written = field(default_factory=scene.Written)
    group: type[bpy.types.PropertyGroup] | None = None
    keys: dict[str, ParamPath] = field(default_factory=dict[str, ParamPath])
    syncing: bool = False
    pending: bool = False
    dragged: frozenset[str] = frozenset()
    last_ms: float = 0.0
    message: str = ""
    stored: Path | None = None
    """The project a loaded .blend names, offered in the panel; never opened by itself"""


STATE = State()


# ------------------------------------------------------------------------------------------------- properties


def _prop(spec: Spec, update: Any) -> Any:
    r = spec.range
    desc = (r.doc + (f" ({r.unit})" if r.unit else "")) if r else ""
    match spec.kind:
        case "bool":
            return BoolProperty(name=spec.label, default=bool(spec.default), description=desc, update=update)
        case "int":
            lo, hi = (int(r.lo), int(r.hi)) if r else (-(2**31), 2**31 - 1)
            return IntProperty(
                name=spec.label, default=int(spec.default), min=lo, max=hi, description=desc, update=update
            )
        case "float":
            lo, hi = (r.lo, r.hi) if r else (-1e9, 1e9)
            step = (r.step * 100.0) if r and r.step else 1.0  # Blender's step is in hundredths
            return FloatProperty(
                name=spec.label, default=float(spec.default), min=lo, max=hi, soft_min=lo, soft_max=hi, step=step,
                precision=4, description=desc, update=update,
            )  # fmt: skip


def _on_param(key: str) -> Any:
    def update(self: bpy.types.PropertyGroup, _ctx: bpy.types.Context) -> None:
        s = STATE.session
        if STATE.syncing or s is None:
            return
        value = getattr(self, key)
        path = STATE.keys[key]
        if path[0] == "@composite":
            s.set_composite(path[1], float(value))
        else:
            s.set(path, value)
        schedule()

    return update


def _build_group(session: Session[Any]) -> type[bpy.types.PropertyGroup]:
    STATE.keys.clear()
    annotations: dict[str, Any] = {}
    for spec in specs(session.base):
        annotations[spec.key] = _prop(spec, _on_param(spec.key))
        STATE.keys[spec.key] = spec.path
    for c in session.model.composites:
        key = f"composite__{c.name}"
        annotations[key] = _prop(Spec(("@composite", c.name), "float", c.default, c.range), _on_param(key))
        STATE.keys[key] = ("@composite", c.name)
    return type("ProctuneParams", (bpy.types.PropertyGroup,), {"__annotations__": annotations})


def _install_group(session: Session[Any]) -> None:
    if STATE.group is not None:
        del bpy.types.Scene.proctune_params
        bpy.utils.unregister_class(STATE.group)
    STATE.group = _build_group(session)
    bpy.utils.register_class(STATE.group)
    bpy.types.Scene.proctune_params = PointerProperty(type=STATE.group)
    sync_properties()


def sync_properties() -> None:
    """Show the session's values on the sliders without triggering their updates."""
    s = STATE.session
    if s is None:
        return
    props = bpy.context.scene.proctune_params
    STATE.syncing = True
    try:
        for spec in specs(s.base):
            setattr(props, spec.key, spec.default)
        for c in s.model.composites:
            setattr(props, f"composite__{c.name}", s.composites.get(c.name, c.default))
    finally:
        STATE.syncing = False


# ------------------------------------------------------------------------------------------------- rebuilding


def _modal_running() -> bool:
    wm = bpy.context.window_manager
    return wm is not None and any(len(w.modal_operators) > 0 for w in wm.windows)


def schedule() -> None:
    if not STATE.pending:
        STATE.pending = True
        bpy.app.timers.register(_tick, first_interval=DEBOUNCE_S)


def _tick() -> float | None:
    if _modal_running() and not STATE.dragged:
        return 0.1  # a grab or a stroke is running: capture when it ends
    STATE.pending = False
    rebuild()
    return None


def rebuild() -> None:
    """Capture hand edits, evaluate, write. Safe to call directly (tests, the Rebuild button)."""
    s = STATE.session
    if s is None:
        return
    t = time.perf_counter()
    for name, pos in scene.moved_handles(STATE.written).items():
        s.drag(name, pos)
    edits = scene.hand_edits(STATE.written)
    if edits:
        if s.last is None:
            s.evaluate()
        scene.apply_hand_edits(s, edits)
    result = s.evaluate()
    scene.write(result, STATE.written)
    scene.write_handles(s, STATE.written, skip=STATE.dragged)
    STATE.dragged = frozenset()
    sync_properties()
    STATE.last_ms = (time.perf_counter() - t) * 1000.0
    stale = result.stale
    STATE.message = f"stale sculpt (topology changed): {', '.join(stale)}" if stale else ""


@persistent
def _on_depsgraph(_scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
    if STATE.session is None or STATE.pending:
        return
    for u in depsgraph.updates:
        obj = getattr(u.id, "original", None)
        if not isinstance(obj, bpy.types.Object):
            continue
        if scene.HANDLE_KEY in obj and u.is_updated_transform:
            name = str(obj[scene.HANDLE_KEY])
            last = STATE.written.handles.get(name)
            if last is not None and np.linalg.norm(np.array(obj.location) - last) > 1e-6:
                STATE.dragged = STATE.dragged | {name}
                schedule()
                return
        elif scene.ELEMENT_KEY in obj and (u.is_updated_transform or u.is_updated_geometry):
            schedule()
            return


@persistent
def _on_load(_dummy: object) -> None:
    """Forget the previous file's session. The project this file names is only offered: opening it imports and runs
    the model's Python, which must not happen just because a .blend was opened (Blender's auto-run protection would
    not apply to it)."""
    STATE.session, STATE.path, STATE.written = None, None, scene.Written()
    path = bpy.context.scene.get(PROJECT_KEY)
    STATE.stored = Path(str(path)) if path and Path(str(path)).is_absolute() else None


# ------------------------------------------------------------------------------------------------- projects


def _start(session: Session[Any], path: Path | None) -> None:
    STATE.session, STATE.path, STATE.written = session, path, scene.Written()
    for obj in [*scene.element_objects().values(), *scene.handle_objects().values()]:
        bpy.data.objects.remove(obj)
    _install_group(session)
    if path is not None:
        bpy.context.scene[PROJECT_KEY] = str(path)
    rebuild()


def open_project(path: Path) -> Session[Any]:
    session = Session.load(path)
    _start(session, path)
    return session


def new_project(source: str, import_dir: Path) -> Session[Any]:
    from proctune.core import load_model  # noqa: PLC0415

    session = Session.new(load_model(source, (str(import_dir),)), source, (str(import_dir),))
    _start(session, None)
    return session


def save_project(path: Path) -> None:
    s = STATE.session
    if s is None:
        raise RuntimeError("no project open")
    rebuild()  # capture anything not yet captured
    if STATE.path is None or path.parent != STATE.path.parent:
        # import paths are relative to the project file
        s.import_paths = tuple(
            str(Path(p).resolve().relative_to(path.parent.resolve(), walk_up=True)) for p in _absolute_imports(s)
        )
    s.save(path)
    STATE.path = path
    bpy.context.scene[PROJECT_KEY] = str(path)


def _absolute_imports(s: Session[Any]) -> tuple[Path, ...]:
    base = STATE.path.parent if STATE.path else Path.cwd()
    return tuple((base / p).resolve() for p in s.import_paths)


# ------------------------------------------------------------------------------------------------- operators


class PROCTUNE_OT_open(bpy.types.Operator, ImportHelper):  # pyright: ignore[reportIncompatibleMethodOverride]  # the stubs' ImportHelper.invoke returns set[str]
    """Open a proctune project (.json)"""

    bl_idname = "proctune.open"
    bl_label = "Open project"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: bpy.types.Context) -> Ret:
        open_project(Path(self.filepath))
        return {"FINISHED"}


class PROCTUNE_OT_open_stored(bpy.types.Operator):
    """Open the project this .blend names, after showing what it would import and run"""

    bl_idname = "proctune.open_stored"
    bl_label = "Open this file's project?"

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> Ret:
        if STATE.stored is None:
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=520, confirm_text="Open and run its model")

    def draw(self, context: bpy.types.Context) -> None:
        lay = self.layout
        if lay is None or STATE.stored is None:
            return
        try:
            info = describe(STATE.stored)
        except (OSError, ValueError) as e:
            lay.label(text=f"cannot read {STATE.stored}: {e}", icon="ERROR")
            return
        lay.label(text="Opening a project imports and runs its model's Python.", icon="ERROR")
        lay.label(text=f"Project: {info.path}")
        lay.label(text=f"Model: {info.model}")
        for d in info.import_dirs:
            lay.label(text=f"Imported from: {d}")

    def execute(self, context: bpy.types.Context) -> Ret:
        if STATE.stored is None:
            return {"CANCELLED"}
        open_project(STATE.stored)
        return {"FINISHED"}


class PROCTUNE_OT_new(bpy.types.Operator):
    """Start from a model's defaults: module:attribute, imported from a directory"""

    bl_idname = "proctune.new"
    bl_label = "New from model"
    source: StringProperty(name="Model", description="module:attribute")  # pyright: ignore[reportInvalidTypeForm]
    directory: StringProperty(name="Import from", subtype="DIR_PATH")  # pyright: ignore[reportInvalidTypeForm]

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> Ret:
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context: bpy.types.Context) -> Ret:
        new_project(self.source, Path(bpy.path.abspath(self.directory)))
        return {"FINISHED"}


class PROCTUNE_OT_save(bpy.types.Operator, ExportHelper):  # pyright: ignore[reportIncompatibleMethodOverride]  # the stubs' ExportHelper.invoke returns set[str]
    """Save the project: parameters and tweaks as JSON, sculpts as .npz beside it"""

    bl_idname = "proctune.save"
    bl_label = "Save project"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # pyright: ignore[reportInvalidTypeForm]

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> Ret:  # pyright: ignore[reportIncompatibleMethodOverride]
        if STATE.path is not None:
            self.filepath = str(STATE.path)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> Ret:
        save_project(Path(self.filepath))
        self.report({"INFO"}, f"saved {self.filepath}")
        return {"FINISHED"}


class PROCTUNE_OT_rebuild(bpy.types.Operator):
    """Capture hand edits and rebuild every element"""

    bl_idname = "proctune.rebuild"
    bl_label = "Rebuild"

    def execute(self, context: bpy.types.Context) -> Ret:
        rebuild()
        return {"FINISHED"}


class PROCTUNE_OT_clear(bpy.types.Operator):
    """Remove an element's hand edits"""

    bl_idname = "proctune.clear"
    bl_label = "Clear edits"
    element: StringProperty()  # pyright: ignore[reportInvalidTypeForm]
    sculpt: BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]
    tweak: BoolProperty(default=True)  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context: bpy.types.Context) -> Ret:
        s = STATE.session
        if s is None:
            return {"CANCELLED"}
        s.clear(self.element, sculpt=self.sculpt, tweak=self.tweak)
        # what is in the scene now is the edit being removed: overwrite it, do not capture it again
        STATE.written = scene.Written()
        result = s.evaluate()
        scene.write(result, STATE.written)
        scene.write_handles(s, STATE.written)
        return {"FINISHED"}


class PROCTUNE_OT_defaults(bpy.types.Operator):
    """Reset every parameter and composite to the model's defaults (hand edits are kept)"""

    bl_idname = "proctune.defaults"
    bl_label = "Reset parameters"

    def execute(self, context: bpy.types.Context) -> Ret:
        s = STATE.session
        if s is None:
            return {"CANCELLED"}
        s.base = s.model.defaults()
        s.composites = {c.name: c.default for c in s.model.composites}
        rebuild()
        return {"FINISHED"}


# ------------------------------------------------------------------------------------------------- panel


class PROCTUNE_PT_panel(bpy.types.Panel):
    bl_idname = "PROCTUNE_PT_panel"
    bl_label = "proctune"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "proctune"

    def draw(self, context: bpy.types.Context) -> None:
        lay = self.layout
        if lay is None:
            return
        s = STATE.session
        row = lay.row(align=True)
        row.operator("proctune.open", text="Open", icon="FILEBROWSER")
        row.operator("proctune.new", text="New", icon="ADD")
        row.operator("proctune.save", text="Save", icon="FILE_TICK")
        if s is None:
            if STATE.stored is not None:
                lay.label(text=f"This file used {STATE.stored}")
                lay.operator("proctune.open_stored", text="Open it (runs its model's code)", icon="ERROR")
            else:
                lay.label(text="No project open")
            return
        lay.label(text=f"{s.model.name}: {STATE.path.name if STATE.path else 'unsaved'}")
        row = lay.row(align=True)
        row.operator("proctune.rebuild", icon="FILE_REFRESH")
        row.operator("proctune.defaults", icon="LOOP_BACK")
        lay.label(text=f"last rebuild {STATE.last_ms:.0f} ms")
        if STATE.message:
            lay.label(text=STATE.message, icon="ERROR")
        props = context.scene.proctune_params
        if s.model.composites:
            box = lay.box()
            box.label(text="Composites", icon="DRIVER")
            for c in s.model.composites:
                box.prop(props, f"composite__{c.name}", slider=True, text=c.name)
        groups: dict[ParamPath, list[Spec]] = {}
        for spec in specs(s.base):
            groups.setdefault(spec.path[:-1], []).append(spec)
        for prefix, items in groups.items():
            box = lay.box()
            box.label(text=" / ".join(prefix) if prefix else "Parameters", icon="PROPERTIES")
            for spec in items:
                box.prop(props, spec.key, slider=spec.kind == "float")
        self._layers(lay)
        self._edits(lay, s)
        self._handles(lay, s)

    @staticmethod
    def _layers(lay: bpy.types.UILayout) -> None:
        box = lay.box()
        box.label(text="Layers", icon="OUTLINER_COLLECTION")
        row = box.row(align=True)
        for c in scene.root().children:
            if c.name != scene.HANDLES:
                row.prop(
                    c, "hide_viewport", text=c.name.removeprefix(f"{scene.ROOT}: "), toggle=True, invert_checkbox=True
                )

    @staticmethod
    def _edits(lay: bpy.types.UILayout, s: Session[Any]) -> None:
        box = lay.box()
        box.label(text="Hand edits (move or sculpt an element)", icon="SCULPTMODE_HLT")
        if not s.edits:
            box.label(text="none")
        for name, e in s.edits.items():
            what = [w for w, on in (("sculpt", e.displacement is not None), ("moved", not e.tweak.identity)) if on]
            row = box.row(align=True)
            row.label(text=f"{name}: {', '.join(what)}")
            op = row.operator("proctune.clear", text="", icon="X")
            if op is not None:
                op.element = name

    @staticmethod
    def _handles(lay: bpy.types.UILayout, s: Session[Any]) -> None:
        hs = s.handles()
        if not hs:
            return
        box = lay.box()
        box.label(text="Handles (drag the empties)", icon="EMPTY_DATA")
        for h in hs:
            box.label(text=f"{h.name}: {', '.join('.'.join(p) for p in h.edits)}")


CLASSES: tuple[type, ...] = (
    PROCTUNE_OT_open,
    PROCTUNE_OT_open_stored,
    PROCTUNE_OT_new,
    PROCTUNE_OT_save,
    PROCTUNE_OT_rebuild,
    PROCTUNE_OT_clear,
    PROCTUNE_OT_defaults,
    PROCTUNE_PT_panel,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.load_post.append(_on_load)


def unregister() -> None:
    for h, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph), (bpy.app.handlers.load_post, _on_load)):
        if fn in h:
            h.remove(fn)
    if STATE.group is not None:
        del bpy.types.Scene.proctune_params
        bpy.utils.unregister_class(STATE.group)
        STATE.group = None
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
