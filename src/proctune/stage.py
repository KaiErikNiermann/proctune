"""Make proctune importable in Blender's own Python without anything else from this environment.

Blender runs its own interpreter, often another Python version than the environment proctune is installed in, so
putting this environment's site-packages on Blender's path would bring its numpy (and the rest) along, compiled for
the wrong Python. Instead the `proctune` package alone is copied into a cache folder (``$XDG_CACHE_HOME/proctune``
or ``~/.cache/proctune`` on Linux, ``%LOCALAPPDATA%\\proctune`` on Windows), and only that folder goes on Blender's
path. The copy is redone when the package's files change (an editable install picks up edits on the next launch); a
copy rather than a link, since links need extra rights on Windows.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import proctune

#: What a staged package keeps: the code and its data files
KEEP = (".py", ".toml", ".typed")


def cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "proctune"


def _package() -> Path:
    return Path(proctune.__file__).resolve().parent


def _files(package: Path) -> list[Path]:
    return sorted(p for p in package.rglob("*") if p.is_file() and p.suffix in KEEP and "__pycache__" not in p.parts)


def _fingerprint(package: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(f"{p.relative_to(package)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def stage() -> Path:
    """The folder to put on Blender's path: it holds only a copy of the `proctune` package, current with this one."""
    package = _package()
    files = _files(package)
    root = cache_dir() / "blender-path"
    target = root / proctune.__name__
    stamp = root / ".fingerprint"
    print_ = _fingerprint(package, files)
    if target.is_dir() and stamp.exists() and stamp.read_text() == print_:
        return root
    if target.exists():
        shutil.rmtree(target)
    for p in files:
        dest = target / p.relative_to(package)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    stamp.write_text(print_)
    return root
