"""proctune: open a model's project in Blender with live sliders, or check that Blender's Python can build it.

    proctune open PROJECT.json [--install-missing] [-- blender args]
    proctune check PROJECT.json [--install-missing] [--json]

Blender is "blender" on the PATH, or the proctune config's "blender" (PROCTUNE_BLENDER), or --blender.
Nothing is installed unless asked: a missing package is reported with which Python looked, where, and where an
install would go.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

import proctune
from proctune import config
from proctune.stage import stage

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__, rich_markup_mode=None)
# no hard wrapping: paths and commands must stay copyable (the terminal wraps long lines itself)
out = Console(soft_wrap=True)
err = Console(stderr=True, soft_wrap=True)

Blender = Annotated[str | None, typer.Option("--blender", help="The Blender executable (default: from the config)")]
Install = Annotated[bool, typer.Option("--install-missing", help="Install what the project lacks, after saying where")]


def _blender(given: str | None) -> str:
    exe = given or config.load().blender
    if shutil.which(exe) is None:
        err.print(f"[red]Blender not found:[/] {exe!r}")
        err.print(
            "Put blender on the PATH, set `blender` in the proctune config (or PROCTUNE_BLENDER), or pass --blender."
        )
        raise typer.Exit(2)
    return exe


def _boot(module: str) -> str:
    """Python for Blender's --python-expr: only the staged proctune on its path, then the module's main()."""
    return f"import sys; sys.path.insert(0, {str(stage())!r}); from {module} import main; main()"


def _preflight(blender: str, project: Path, install: bool) -> dict[str, Any]:
    from proctune.blender.preflight import MARK  # noqa: PLC0415 -- the constant only; bpy is not imported

    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    cmd = [blender, "-b", "--factory-startup", "--python-expr", _boot("proctune.blender.preflight"), "--", str(project)]
    r = subprocess.run(
        [*cmd, *(["--install"] if install else [])], env=env, capture_output=True, text=True, check=False
    )
    last = next((line for line in reversed(r.stdout.splitlines()) if line.startswith(MARK)), None)
    if last is None:
        return {
            "error": f"Blender did not finish the check (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-1500:]}"
        }
    return json.loads(last.removeprefix(MARK))


def _report(project: Path, result: dict[str, Any]) -> None:
    if result.get("installed"):
        out.print(f"installed {', '.join(result['installed'])} into {result.get('into')}")
    if result.get("ok"):
        out.print(f"[green]{project.name}[/]: builds in Blender's Python")
        return
    if missing := result.get("missing"):
        out.print(f"[yellow]{project.name}[/] needs {', '.join(missing)}, which Blender's Python cannot import.")
        for line in result.get("environment", []):
            out.print(f"  {line}", highlight=False)
        out.print(
            "Install with --install-missing (into the folder above), add your own environment's site-packages to "
            "`paths` in the proctune config, or use the Install button in Blender's proctune panel."
        )
        return
    if error := result.get("error"):
        out.print(f"[red]{project.name}[/]: {error}")


def _project(path: Path) -> Path:
    if not path.is_file():
        err.print(f"[red]no project at[/] {path}")
        raise typer.Exit(2)
    return path.resolve()


@app.command()
def check(
    project: Annotated[Path, typer.Argument(help="The project file (.json)")],
    install_missing: Install = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON")] = False,
    blender: Blender = None,
) -> None:
    """Load and build the project with Blender's own Python; report what is missing and where it looked."""
    path = _project(project)
    result = _preflight(_blender(blender), path, install_missing)
    if as_json:
        sys.stdout.write(json.dumps(result) + "\n")
    else:
        _report(path, result)
    raise typer.Exit(0 if result.get("ok") else 1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def open(  # noqa: PLR0917 -- Typer's parameters
    ctx: typer.Context,
    project: Annotated[Path, typer.Argument(help="The project file (.json)")],
    install_missing: Install = False,
    compare: Annotated[bool, typer.Option(help="Split the view: the model, and its surface alone")] = True,
    skip_check: Annotated[bool, typer.Option("--skip-check", help="Start Blender without checking first")] = False,
    blender: Blender = None,
) -> None:
    """Start Blender with the add-on and the project (anything after -- goes to Blender)."""
    path = _project(project)
    exe = _blender(blender)
    if not skip_check:
        _report(path, _preflight(exe, path, install_missing))
    args = [path.as_posix(), *([] if compare else ["--no-compare"])]
    cmd = [exe, *ctx.args, "--python-expr", _boot("proctune.blender.launch"), "--", *args]
    raise typer.Exit(subprocess.run(cmd, check=False).returncode)


@app.command()
def version() -> None:
    """Print proctune's version."""
    from importlib.metadata import version as dist_version  # noqa: PLC0415

    out.print(f"proctune {dist_version(proctune.__name__)}")


if __name__ == "__main__":
    app()
