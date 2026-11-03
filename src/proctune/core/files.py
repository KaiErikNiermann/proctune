"""Reading and writing project files with errors that name the file and the fix, on Linux and Windows.

Writes go to a temporary file in the same folder and are moved into place with `os.replace`, which is atomic on both
systems when source and destination are on the same volume (they are: same folder). A failed write leaves the old file
as it was.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from proctune.errors import ProjectError


def write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists() and not os.access(path, os.W_OK):
            # os.replace would silently swap a read-only file on Linux (the folder's permission is what counts);
            # refuse, as the file's own permission says it must not change
            raise PermissionError(errno.EACCES, "the file is read-only", str(path))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise


def write_error(path: Path, e: OSError) -> ProjectError:
    """A write failure as an error that says what to do."""
    folder = path.parent
    match e:
        case PermissionError():
            return ProjectError(
                f"cannot write {path}: permission denied ({e.strerror})",
                f"save to a folder you can write to, or make {path.name} and {folder} writable",
            )
        case FileNotFoundError():
            return ProjectError(f"cannot write {path}: the folder {folder} does not exist", "create it or pick another")
        case _ if e.errno == errno.ENOSPC:
            return ProjectError(f"cannot write {path}: the disk is full", "free space or save elsewhere")
        case _ if e.errno == errno.EROFS:
            return ProjectError(f"cannot write {path}: the file system is read-only", "save to a writable drive")
        case _:
            return ProjectError(f"cannot write {path}: {e.strerror or e}", "check the path and its folder")


def read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ProjectError(f"{path} does not exist", "check the path, or open another project") from e
    except PermissionError as e:
        raise ProjectError(f"cannot read {path}: permission denied", f"make {path.name} readable") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ProjectError(f"cannot read {path}: {e}", "check the file is a proctune project (.json)") from e
    try:
        doc: object = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProjectError(
            f"{path} is not valid JSON (line {e.lineno}, column {e.colno}: {e.msg})", "fix the file at that position"
        ) from e
    if not isinstance(doc, dict):
        raise ProjectError(f"{path} is not a proctune project (top level is not an object)", "open a project .json")
    return cast(dict[str, Any], doc)


def relative_or_absolute(target: Path, start: Path) -> str:
    """`target` relative to the folder `start` when possible (portable projects); absolute when no relative path
    exists, as between two drives on Windows."""
    try:
        return os.path.relpath(target, start)
    except ValueError:
        return str(target.resolve())
