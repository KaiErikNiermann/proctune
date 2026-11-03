"""A model being tuned: its base parameters, composite values and hand edits, evaluated on demand and saved as a
project (``NAME.json`` with the values, ``NAME.npz`` beside it with the sculpts).

The project names its model as ``module:attribute`` plus directories to import it from, relative to the project
file, so a project opens anywhere the model's code is.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path as FsPath
from typing import Any, cast

import numpy as np

from proctune.core import params as P
from proctune.core.element import Edit, Element, Resolved, Tweak, capture_sculpt, capture_tweak, resolve
from proctune.core.model import F64, Handle, Model

FORMAT = 1


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
    source: str = ""
    """``module:attribute`` of the model, kept for saving"""
    import_paths: tuple[str, ...] = ()
    last: Result | None = None

    @staticmethod
    def new[U](model: Model[U], source: str = "", import_paths: tuple[str, ...] = ()) -> Session[U]:
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
            built: tuple[Element, ...] = stage.build(p, dict(done))
            for e in built:
                if e.name in done:
                    raise ValueError(f"stage {stage.name}: element name {e.name!r} is taken")
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
        """Compose a world-space move of the element (4x4) into its tweak."""
        r = self._resolved(name)
        if not r.element.tweakable:
            raise ValueError(f"{name} cannot be moved as a whole")
        edit = self.edits.get(name, Edit())
        self.edits[name] = Edit(edit.tweak.then(capture_tweak(r.element, matrix)), edit.displacement)

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
        sculpts = {n: e.displacement for n, e in self.edits.items() if e.displacement is not None}
        doc = {
            "format": FORMAT,
            "model": self.source,
            "import_paths": list(self.import_paths),
            "params": P.to_dict(self.base),
            "composites": self.composites,
            "tweaks": {
                n: {"linear": e.tweak.linear.tolist(), "offset": e.tweak.offset.tolist()}
                for n, e in self.edits.items()
                if not e.tweak.identity
            },
        }
        path.write_text(json.dumps(doc, indent=2) + "\n")
        npz = path.with_suffix(".npz")
        if sculpts:
            np.savez_compressed(npz, **cast(dict[str, Any], sculpts))
        elif npz.exists():
            npz.unlink()

    @staticmethod
    def load(path: FsPath) -> Session[Any]:
        doc = json.loads(path.read_text())
        if doc.get("format") != FORMAT:
            raise ValueError(f"{path}: format {doc.get('format')}, expected {FORMAT}")
        paths = tuple(cast(list[str], doc.get("import_paths", [])))
        model = load_model(doc["model"], tuple(str((path.parent / p).resolve()) for p in paths))
        s = Session.new(model, doc["model"], paths)
        s.base = P.from_dict(cast(type[Any], type(s.base)), doc.get("params", {}))
        s.composites.update(cast(dict[str, float], doc.get("composites", {})))
        tweaks = cast(dict[str, dict[str, list[Any]]], doc.get("tweaks", {}))
        for n, t in tweaks.items():
            s.edits[n] = Edit(Tweak(np.array(t["linear"], np.float64), np.array(t["offset"], np.float64)))
        npz = path.with_suffix(".npz")
        if npz.exists():
            with np.load(npz) as z:
                for n in z.files:
                    s.edits[n] = Edit(s.edits.get(n, Edit()).tweak, np.asarray(z[n], np.float64))
        return s


def load_model(source: str, import_paths: tuple[str, ...] = ()) -> Model[Any]:
    """Import ``module:attribute`` (after putting `import_paths` first on ``sys.path``) and return that model."""
    module, _, attr = source.partition(":")
    if not attr:
        raise ValueError(f"{source!r}: expected module:attribute")
    for p in reversed(import_paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.reload(sys.modules[module]) if module in sys.modules else importlib.import_module(module)
    return cast(Model[Any], getattr(mod, attr))
