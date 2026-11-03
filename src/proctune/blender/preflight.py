"""Inside Blender (in the background): load and build a project with Blender's own Python, report what is missing
and where it looked, and only when asked install what is missing (then try again). `proctune check` and
`proctune open` run this before showing anything.

The report is printed for people, and its last line is ``PROCTUNE_PREFLIGHT {json}`` for the command that started
Blender: ``ok``, ``missing`` (pip packages), ``error``, ``installed``, ``into``, ``environment`` (the lines of
`deps.Environment`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from proctune import config, deps
from proctune.core import Session
from proctune.core import files as F
from proctune.errors import MissingDependenciesError, ProctuneError

MARK = "PROCTUNE_PREFLIGHT "
ROUNDS = 6
"""Packages often turn up missing one at a time"""


def target_for(project: Path) -> Path:
    """The project's first package folder if it declares one, else proctune's default target."""
    folders = F.read_json(project).get("package_paths", [])
    return (project.parent / folders[0]).resolve() if folders else deps.default_target()


def run(project: Path, install: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "missing": [], "error": "", "installed": [], "into": "", "environment": []}
    deps.add_paths(config.load().paths)
    deps.ensure_on_path(deps.default_target())  # where the add-on looks too
    for _ in range(ROUNDS):
        try:
            Session.load(project).evaluate()
        except MissingDependenciesError as e:
            out["missing"] = list(e.packages)
            out["environment"] = list(deps.environment().lines())
            if not install:
                return out
            into = target_for(project)
            out["into"] = str(into)
            out["installed"] += list(deps.install(e.packages, into).installed)
            continue
        except ProctuneError as e:
            out["error"] = str(e)
            return out
        out.update(ok=True, missing=[])
        return out
    out["error"] = f"still missing packages after {ROUNDS} rounds of installing"
    return out


def main() -> None:
    """``blender -b --python-expr "...; from proctune.blender.preflight import main; main()" -- PROJECT [--install]``"""
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not args:
        result: dict[str, Any] = {"error": "no project given"}
    else:
        try:
            result = run(Path(args[0]).resolve(), "--install" in args)
        except ProctuneError as e:
            result = {"error": str(e)}
    print(MARK + json.dumps(result))
