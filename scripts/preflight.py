"""Run inside Blender (by scripts/gui.py): load and build a project with Blender's own Python, print a report of what
is missing and where it looked, and, only with --install-missing, install what is missing (then try again).

    blender -b --factory-startup --python scripts/preflight.py -- PROJECT.json [--install-missing]

The last line printed is ``PROCTUNE_PREFLIGHT {json}``: ``ok``, ``missing`` (pip packages), ``error``,
``installed`` and ``into``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_env

blender_env.setup()
from proctune import config, deps  # noqa: E402
from proctune.core import Session  # noqa: E402
from proctune.core import files as F  # noqa: E402
from proctune.errors import MissingDependenciesError, ProctuneError  # noqa: E402

ROUNDS = 6  # packages often turn up missing one at a time


def target_for(project: Path) -> Path:
    """The project's first package folder if it declares one, else proctune's default (config, then Blender's)."""
    folders = F.read_json(project).get("package_paths", [])
    return (project.parent / folders[0]).resolve() if folders else deps.default_target()


def main(project: Path, install: bool) -> dict[str, object]:
    out: dict[str, object] = {"ok": False, "missing": [], "error": "", "installed": [], "into": ""}
    deps.add_paths(config.load().paths)
    deps.ensure_on_path(deps.default_target())  # where the add-on looks too (and installs by default)
    for _ in range(ROUNDS):
        try:
            Session.load(project).evaluate()
        except MissingDependenciesError as e:
            out["missing"] = list(e.packages)
            env = deps.environment()
            print(f"\n{project.name} needs {', '.join(e.packages)}, which Blender's Python cannot import.")
            print("\n".join(env.lines()))
            if not install:
                return out
            into = target_for(project)
            out["into"] = str(into)
            print(f"installing {', '.join(e.packages)} into {into} (--install-missing)")
            r = deps.install(e.packages, into)
            out["installed"] = [*out["installed"], *r.installed]  # type: ignore[list-item]
            continue
        except ProctuneError as e:
            out["error"] = str(e)
            return out
        out["ok"], out["missing"] = True, []
        return out
    out["error"] = f"still missing packages after {ROUNDS} rounds of installing"
    return out


args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
result = main(Path(args[0]).resolve(), "--install-missing" in args) if args else {"error": "no project given"}
print("PROCTUNE_PREFLIGHT " + json.dumps(result))
