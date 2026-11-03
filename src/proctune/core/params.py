"""Parameters as frozen dataclasses, instrumented by annotation.

A model's parameters are a frozen dataclass whose fields are numbers annotated with a `Range`
(``Annotated[float, Range(0.1, 0.6, unit="m")]``), booleans, `Curve`s, or nested parameter dataclasses (a group).
`specs` walks the tree into flat `Spec`s that a UI turns into sliders; `get` and `replace` address one
value by its path; `to_dict`/`from_dict` move the tree to and from JSON.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, Literal, cast

from proctune.core.curves import Curve

type Path = tuple[str, ...]
type Kind = Literal["float", "int", "bool", "curve"]
type Value = float | int | bool | Curve


@dataclass(frozen=True)
class Range:
    """A slider's bounds, step, unit and tooltip."""

    lo: float
    hi: float
    step: float | None = None
    unit: str = ""
    doc: str = ""


@dataclass(frozen=True)
class Spec:
    path: Path
    kind: Kind
    default: Value
    range: Range | None

    @property
    def key(self) -> str:
        """A flat name for the value (path joined by ``__``), usable as a property identifier."""
        return "__".join(self.path)

    @property
    def label(self) -> str:
        return self.path[-1].replace("_", " ")


def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls, include_extras=True)


def _kind(tp: Any) -> Kind | None:
    match tp:
        case _ if tp is bool:
            return "bool"
        case _ if tp is int:
            return "int"
        case _ if tp is float:
            return "float"
        case _:
            return None


def specs(obj: object, prefix: Path = ()) -> tuple[Spec, ...]:
    """Every leaf value of a parameter tree (an instance, so its defaults are the current values)."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"{obj!r} is not a dataclass instance")
    out: list[Spec] = []
    hints = _hints(type(obj))
    for f in dataclasses.fields(obj):
        value: object = getattr(obj, f.name)
        path = (*prefix, f.name)
        if isinstance(value, Curve):
            out.append(Spec(path, "curve", value, None))
            continue
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out += specs(value, path)
            continue
        tp = hints[f.name]
        rng: Range | None = None
        if typing.get_origin(tp) is typing.Annotated:
            args = typing.get_args(tp)
            tp = args[0]
            rng = next((a for a in args[1:] if isinstance(a, Range)), None)
        kind = _kind(tp)
        if kind is None:
            raise TypeError(f"{'.'.join(path)}: unsupported parameter type {tp!r}")
        out.append(Spec(path, kind, cast(Value, value), rng))
    return tuple(out)


def get(obj: object, path: Path) -> Value:
    cur = obj
    for name in path:
        cur = getattr(cur, name)
    return cast(Value, cur)


def replace[T](obj: T, path: Path, value: Value) -> T:
    """A copy of the tree with the value at `path` replaced."""
    head, *rest = path
    if not rest:
        return cast(T, dataclasses.replace(cast(Any, obj), **{head: value}))
    return cast(T, dataclasses.replace(cast(Any, obj), **{head: replace(getattr(obj, head), tuple(rest), value)}))


def to_dict(obj: object) -> dict[str, Any]:
    return dataclasses.asdict(cast(Any, obj))


def from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
    """Rebuild a parameter tree from `to_dict` output; missing keys keep the class's defaults, unknown keys fail."""
    hints = _hints(cls)
    names = {f.name for f in dataclasses.fields(cast(Any, cls))}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown parameters {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        tp = hints[name]
        if typing.get_origin(tp) is typing.Annotated:
            tp = typing.get_args(tp)[0]
        if tp is Curve:
            c = cast(dict[str, Any], value)
            pts = tuple((float(x), float(y)) for x, y in cast(list[list[float]], c["points"]))
            kwargs[name] = Curve(pts, float(c.get("lo", 0.0)), float(c.get("hi", 1.0)))
            continue
        is_group = isinstance(tp, type) and dataclasses.is_dataclass(tp)
        kwargs[name] = from_dict(cast(type, tp), cast(dict[str, Any], value)) if is_group else value
    return cls(**kwargs)


def range_of(cls: type, name: str) -> Range | None:
    """The `Range` a parameter class gives one of its fields."""
    tp = _hints(cls)[name]
    if typing.get_origin(tp) is typing.Annotated:
        return next((a for a in typing.get_args(tp)[1:] if isinstance(a, Range)), None)
    return None


def clamp(obj: object, name: str, value: float) -> float:
    """`value` held to the range of field `name` of `obj`'s class (unchanged if it has none)."""
    r = range_of(type(obj), name)
    return value if r is None else min(max(value, r.lo), r.hi)


def validate(obj: object) -> None:
    """Raise `ValueError` naming every value outside its `Range` (a hand-edited project, say)."""
    bad = [
        f"{'.'.join(s.path)} = {s.default} (allowed {s.range.lo} to {s.range.hi})"
        for s in specs(obj)
        if s.range is not None
        and s.kind in ("float", "int")
        and not s.range.lo <= float(cast(float, s.default)) <= s.range.hi
    ]
    if bad:
        raise ValueError("out of range: " + "; ".join(bad))
