"""A model being tuned: its base parameters, composite values and hand edits, evaluated on demand and saved as a
project (``NAME.json`` with the values, ``NAME.npz`` beside it with the sculpts).

The project names its model as ``module:attribute`` plus directories to import it from, relative to the project
file, so a project opens anywhere the model's code is.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
import json
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path as FsPath
from typing import Any, cast

import numpy as np

from proctune import deps
from proctune.core import files
from proctune.core import params as P
from proctune.core.element import Edit, Element, Resolved, Tweak, capture_sculpt, capture_tweak, resolve
from proctune.core.model import F64, Binds, Handle, Model
from proctune.errors import EditError, MissingDependenciesError, ModelError, ProctuneError, ProjectError

FORMAT = 2
type Track = tuple[tuple[float, float], ...]
"""(frame, value) samples"""
COMPOSITE = "composite:"


@dataclass(frozen=True)
class Result:
    """Every element, resolved, in build order; and by stage."""

    elements: dict[str, Resolved]
    stages: dict[str, tuple[str, ...]]

    @property
    def stale(self) -> tuple[str, ...]:
        return tuple(n for n, r in self.elements.items() if r.stale)


@dataclass
class Session[T]:
    model: Model[T]
    base: T
    composites: dict[str, float] = field(default_factory=dict[str, float])
    edits: dict[str, Edit] = field(default_factory=dict[str, Edit])
    tracks: dict[str, tuple[tuple[float, float], ...]] = field(
        default_factory=dict[str, tuple[tuple[float, float], ...]]
    )
    """Values over time: a dotted parameter path (or ``composite:NAME``) to (frame, value) samples, e.g. baked from
    Blender keyframes; `at_frame` interpolates them"""
    source: str = ""
    """``module:attribute`` of the model, kept for saving"""
    import_paths: tuple[str, ...] = ()
    last: Result | None = None

    @staticmethod
    def new[U](model: Model[U], source: str = "", import_paths: tuple[str, ...] = ()) -> Session[U]:
        deps.require(deps.CORE, "proctune")
        return Session(
            model,
            model.defaults(),
            {c.name: c.default for c in model.composites},
            source=source,
            import_paths=import_paths,
        )

    # ------------------------------------------------------------------------------------------- values

    @property
    def effective(self) -> T:
        """The base parameters with every composite applied, in declared order."""
        return reduce(lambda p, c: c.apply(p, self.composites.get(c.name, c.default)), self.model.composites, self.base)

    def set(self, path: P.Path, value: P.Value) -> None:
        self.base = P.replace(self.base, path, value)

    def set_composite(self, name: str, value: float) -> None:
        if name not in {c.name for c in self.model.composites}:
            raise KeyError(f"no composite {name!r}")
        self.composites[name] = value

    def apply(self, values: Mapping[str, P.Value]) -> None:
        """Set values by dotted path (``upper.length``) or ``composite:NAME``."""
        for name, v in values.items():
            if name.startswith(COMPOSITE):
                self.set_composite(name.removeprefix(COMPOSITE), float(cast(float, v)))
            else:
                self.set(tuple(name.split(".")), v)

    def at_frame(self, frame: float) -> Session[T]:
        """A copy with every track's value at `frame` (linear between samples, held beyond them)."""
        c = dataclasses.replace(self, composites=dict(self.composites), last=None)
        c.apply({n: float(np.interp(frame, [f for f, _ in t], [v for _, v in t])) for n, t in self.tracks.items()})
        return c

    def handles(self) -> tuple[Handle[T], ...]:
        """The handles, placed at the effective parameters (where the model is drawn)."""
        return self.model.handles(self.effective)

    def drag(self, handle: str, world: F64) -> None:
        """Move a handle; it sets base parameters (composites stay applied on top)."""
        h = next(h for h in self.model.handles(self.base) if h.name == handle)
        self.base = h.drag(self.base, world)

    # ------------------------------------------------------------------------------------------- evaluation

    def evaluate(self) -> Result:
        p = self.effective
        done: dict[str, Resolved] = {}
        stages: dict[str, tuple[str, ...]] = {}
        for stage in self.model.stages:
            try:
                built: tuple[Element, ...] = stage.build(p, dict(done))
            except ProctuneError:
                raise
            except Exception as e:
                raise ModelError(
                    f"{self.model.name}: stage {stage.name!r} failed: {type(e).__name__}: {e}",
                    "the model's code raised; the traceback is in the log (Blender: Window > Toggle System Console "
                    "on Windows, the terminal on Linux). The last good mesh stays in the scene",
                ) from e
            for e in built:
                if e.name in done:
                    raise ModelError(
                        f"stage {stage.name}: element name {e.name!r} is taken", "give every element a unique name"
                    )
                done[e.name] = resolve(e, self.edits.get(e.name))
            stages[stage.name] = tuple(e.name for e in built)
        self.last = Result(done, stages)
        return self.last

    def _resolved(self, name: str) -> Resolved:
        r = (self.last or self.evaluate()).elements.get(name)
        if r is None:
            raise KeyError(f"no element {name!r}")
        return r

    # ------------------------------------------------------------------------------------------- hand edits

    def capture_sculpt(self, name: str, world: F64) -> None:
        """Store `world` (n, 3), the element's vertices as sculpted, as its sculpt."""
        r = self._resolved(name)
        edit = self.edits.get(name, Edit())
        self.edits[name] = Edit(edit.tweak, capture_sculpt(r.element, edit.tweak, world))

    def capture_transform(self, name: str, matrix: F64) -> None:
        """A world-space move of the element (4x4): parameter changes if the model binds that element, otherwise
        composed into its tweak."""
        r = self._resolved(name)
        bind = getattr(self.model, "bind", None)
        if bind is not None:
            bound = cast(Binds[T], self.model).bind(self.base, name, matrix)
            if bound is not None:
                self.base = bound
                return
        edit = self.edits.get(name, Edit())
        if not r.element.tweakable:
            # a surface rebuilt around others (a skin) has no place of its own to move: its moved vertices are
            # kept as what they are, a sculpt in its per-vertex frames
            self.capture_sculpt(name, r.verts @ matrix[:3, :3].T + matrix[:3, 3])
            return
        self.edits[name] = Edit(capture_tweak(r.element, matrix, edit.tweak), edit.displacement)

    def set_tweak(self, name: str, tweak: Tweak) -> None:
        """Set an element's move outright (from its object's transform relative to its generated place)."""
        r = self._resolved(name)
        if not r.element.tweakable:
            raise EditError(f"{name} has no place of its own to move", "sculpt it instead")
        edit = self.edits.get(name, Edit())
        self.edits[name] = Edit(tweak, edit.displacement)
        if tweak.identity and edit.displacement is None:
            del self.edits[name]

    def clear(self, name: str, *, sculpt: bool = True, tweak: bool = True) -> None:
        edit = self.edits.get(name)
        if edit is None:
            return
        kept = Edit(Tweak() if tweak else edit.tweak, None if sculpt else edit.displacement)
        if kept.tweak.identity and kept.displacement is None:
            del self.edits[name]
        else:
            self.edits[name] = kept

    # ------------------------------------------------------------------------------------------- files

    def save(self, path: FsPath) -> None:
        """Write ``path`` (JSON) and, if anything is sculpted, ``path.npz``. Each file is written to a temporary file
        beside it and moved into place, so a failed save never leaves a half-written project."""
        sculpts = {n: e.displacement for n, e in self.edits.items() if e.displacement is not None}
        doc = {
            "format": FORMAT,
            "model": self.source,
            "import_paths": list(self.import_paths),
            "params": P.to_dict(self.base),
            "composites": self.composites,
            "tracks": {n: [list(fv) for fv in t] for n, t in self.tracks.items()},
            "tweaks": {
                n: {
                    "offset": e.tweak.offset.tolist(),
                    "rotation": e.tweak.rotation.tolist(),
                    "scale": e.tweak.scale.tolist(),
                }
                for n, e in self.edits.items()
                if not e.tweak.identity
            },
        }
        npz = path.with_suffix(".npz")
        try:
            files.write_atomic(path, (json.dumps(doc, indent=2) + "\n").encode())
            if sculpts:
                buf = io.BytesIO()
                np.savez_compressed(buf, **cast(dict[str, Any], sculpts))
                files.write_atomic(npz, buf.getvalue())
            elif npz.exists():
                npz.unlink()
        except OSError as e:
            raise files.write_error(path, e) from e

    @staticmethod
    def load(path: FsPath) -> Session[Any]:
        doc = files.read_json(path)
        if doc.get("format") == 1 and not doc.get("tweaks"):
            doc["format"] = FORMAT  # format 2 changed only how tweaks are stored
        if doc.get("format") != FORMAT:
            raise ProjectError(
                f"{path}: project format {doc.get('format')!r}, this proctune reads {FORMAT}",
                "open it with the proctune version that wrote it",
            )
        source = doc.get("model")
        if not isinstance(source, str):
            raise ProjectError(f"{path}: no model named", 'the file needs "model": "module:attribute"')
        paths = tuple(cast(list[str], doc.get("import_paths", [])))
        model = load_model(source, tuple(str((path.parent / p).resolve()) for p in paths), path)
        s = Session.new(model, source, paths)
        try:
            s.base = P.from_dict(cast(type[Any], type(s.base)), doc.get("params", {}))
            P.validate(s.base)
        except (KeyError, TypeError, ValueError) as e:
            raise ProjectError(
                f"{path}: parameters do not fit model {source}: {e}",
                "fix the value in the file, or remove it to use the model's default",
            ) from e
        s.composites.update(cast(dict[str, float], doc.get("composites", {})))
        tracks = cast(dict[str, list[list[float]]], doc.get("tracks", {}))
        s.tracks = {n: tuple((float(f), float(v)) for f, v in t) for n, t in tracks.items()}
        tweaks = cast(dict[str, dict[str, list[Any]]], doc.get("tweaks", {}))
        try:
            for n, t in tweaks.items():
                arrays = (
                    np.array(t["offset"], np.float64),
                    np.array(t["rotation"], np.float64),
                    np.array(t["scale"], np.float64),
                )
                s.edits[n] = Edit(Tweak(*arrays))
        except (KeyError, TypeError, ValueError) as e:
            raise ProjectError(
                f"{path}: a move (tweaks) cannot be read: {e}", "remove the broken entry under tweaks"
            ) from e
        npz = path.with_suffix(".npz")
        if npz.exists():
            try:
                with np.load(npz) as z:
                    for n in z.files:
                        s.edits[n] = Edit(s.edits.get(n, Edit()).tweak, np.asarray(z[n], np.float64))
            except (OSError, ValueError, zipfile.BadZipFile) as e:
                raise ProjectError(
                    f"{npz}: the sculpts cannot be read ({e})",
                    f"restore {npz.name} from a backup, or move it away to open the project without its sculpts",
                ) from e
        return s


@dataclass(frozen=True)
class ProjectInfo:
    """What opening a project would import, read without importing anything."""

    path: FsPath
    model: str
    import_dirs: tuple[FsPath, ...]


def describe(path: FsPath) -> ProjectInfo:
    doc = files.read_json(path)
    dirs = tuple((path.parent / p).resolve() for p in cast(list[str], doc.get("import_paths", [])))
    return ProjectInfo(path.resolve(), str(doc.get("model", "")), dirs)


def load_model(source: str, import_paths: tuple[str, ...] = (), project: FsPath | None = None) -> Model[Any]:
    """Import ``module:attribute`` (after putting `import_paths` first on ``sys.path``) and return that model. A
    module that is already imported is reloaded, so edits to the model's code take effect on reopening."""
    where = f"{project}: " if project else ""
    module, _, attr = source.partition(":")
    if not module or not attr:
        raise ProjectError(f"{where}model {source!r} is not module:attribute", 'name it like "limb:MODEL"')
    for p in reversed(import_paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        mod = importlib.reload(sys.modules[module]) if module in sys.modules else importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name == module or (e.name and module.startswith(e.name + ".")):
            dirs = ", ".join(import_paths) or "(none)"
            raise ProjectError(
                f"{where}model module {module!r} not found (looked in {dirs} and Python's path)",
                "check the project's import_paths point at the folder holding the model's .py file",
            ) from e
        req = deps.requirement(e.name or "?", f"model {source}")
        raise MissingDependenciesError(
            f"{where}model {source} imports {req.package}, which this Python lacks",
            deps.install_hint([req.package]),
            (req.package,),
        ) from e
    except Exception as e:
        raise ModelError(
            f"{where}importing model module {module!r} failed: {type(e).__name__}: {e}", "fix the model's code"
        ) from e
    model = getattr(mod, attr, None)
    if model is None:
        raise ProjectError(f"{where}module {module!r} has no {attr!r}", f"the module must define {attr} (the model)")
    return cast(Model[Any], model)
