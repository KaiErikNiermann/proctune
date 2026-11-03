"""Where proctune finds things, settable per machine: a TOML file, overridden by environment variables.

The file is ``$PROCTUNE_CONFIG`` if set, else ``%APPDATA%\\proctune\\config.toml`` on Windows and
``$XDG_CONFIG_HOME/proctune/config.toml`` (``~/.config/...``) elsewhere. Every key is optional::

    blender = "/opt/blender-5.2/blender"     # the Blender executable (dev scripts and tests)
    paths = ["~/venvs/blender/lib/python3.14/site-packages"]
                                             # folders added to Blender's import path, after its own (packages from
                                             # your own environment; must be for Blender's Python version)
    packages = "/data/blender-packages"      # where missing packages are installed (default: Blender's user
                                             # scripts/modules)
    dev_site = "/home/me/proctune-site"      # packages for the dev scripts and tests (default: .blender-site)
    xvfb_run = "/usr/bin/xvfb-run"           # the virtual display wrapper (end-to-end tests, Linux)
    magick = "/usr/bin/magick"               # ImageMagick 7 (end-to-end screenshots); ImageMagick 6: "import"

Environment variables win over the file: ``PROCTUNE_BLENDER``, ``PROCTUNE_PACKAGES``, ``PROCTUNE_DEV_SITE``,
``PROCTUNE_XVFB_RUN``, ``PROCTUNE_MAGICK``, and ``PROCTUNE_PATHS`` (folders separated by ``os.pathsep``: ``:`` on
Linux, ``;`` on Windows).
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

from proctune.errors import ProctuneError

ROOT = Path(__file__).resolve().parents[2]
"""The checkout (for the dev defaults); meaningless in an installed add-on, where the dev keys are not used"""


@dataclass(frozen=True)
class Config:
    blender: str = "blender"
    paths: tuple[Path, ...] = ()
    packages: Path | None = None
    dev_site: Path = ROOT / ".blender-site"
    xvfb_run: str = "xvfb-run"
    magick: str = "magick"


def config_path() -> Path:
    if env := os.environ.get("PROCTUNE_CONFIG"):
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "proctune" / "config.toml"


def load() -> Config:
    path = config_path()
    values: dict[str, object] = {}
    if path.exists():
        try:
            values = dict(tomllib.loads(path.read_text(encoding="utf-8")))
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as e:
            raise ProctuneError(f"{path}: cannot read the proctune config: {e}", "fix the file or remove it") from e
    known = {f.name for f in fields(Config)}
    if unknown := set(values) - known:
        raise ProctuneError(f"{path}: unknown keys {sorted(unknown)}", f"the keys are {sorted(known)}")
    for name in known:
        if env := os.environ.get(f"PROCTUNE_{name.upper()}"):
            values[name] = env.split(os.pathsep) if name == "paths" else env
    paths = values.pop("paths", [])
    if not isinstance(paths, list) or not all(isinstance(x, str) for x in cast(list[object], paths)):
        raise ProctuneError(
            f"{path}: paths must be a list of folder paths, got {paths!r}", 'write paths = ["/a", "/b"]'
        )
    for name in ("packages", "dev_site"):
        if name in values:
            values[name] = Path(str(values[name])).expanduser()
    for name, v in values.items():
        if not isinstance(v, (str, Path)):
            raise ProctuneError(f"{path}: {name} must be a string path, got {v!r}", "quote the path")
    folders = tuple(Path(x).expanduser() for x in cast(list[str], paths) if x)
    return Config(paths=folders, **values)  # pyright: ignore[reportArgumentType]
