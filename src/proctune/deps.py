"""Python packages proctune and its models need, checked up front and installable into Blender's Python.

Blender embeds its own Python (on Arch the system's, in official builds a bundled one); the add-on runs in it, so
packages must be installed for that interpreter, not for whatever ``python`` is on the PATH. `install` runs pip with
``sys.executable`` from inside Blender, into a target folder (by default Blender's user ``scripts/modules``, which is
on Blender's path). Packages that interpreter already has (Blender's own numpy above all) are left out: pip resolves
the full set with ``--dry-run --report``, the ones already present are dropped, and the rest are installed with
``--no-deps``, so a second numpy never shadows Blender's.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from proctune.errors import MissingDependenciesError, ProctuneError

log = logging.getLogger("proctune.deps")


@dataclass(frozen=True)
class Requirement:
    module: str
    """What is imported"""
    package: str
    """What pip installs"""
    why: str


#: What proctune.core needs; models add their own
CORE: tuple[Requirement, ...] = (
    Requirement("numpy", "numpy", "arrays"),
    Requirement("scipy", "scipy", "curve parameters (PCHIP)"),
)
#: Import names that differ from their pip package
PACKAGE_OF: dict[str, str] = {
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "PIL": "Pillow",
    "igl": "libigl",
}


def requirement(module: str, why: str = "") -> Requirement:
    top = module.split(".", maxsplit=1)[0]
    return Requirement(top, PACKAGE_OF.get(top, top), why)


def missing(reqs: Iterable[Requirement]) -> tuple[Requirement, ...]:
    return tuple(r for r in reqs if importlib.util.find_spec(r.module) is None)


@dataclass(frozen=True)
class Environment:
    """Where this Python looks for packages and where an install would go: what a missing-package report says, so
    a setup proctune cannot guess (a venv, a custom Blender build) can be pointed at instead."""

    python: str
    version: str
    """major.minor"""
    searched: tuple[str, ...]
    """The folders on the import path, in order"""
    configured: tuple[str, ...]
    """Folders added by the proctune config (``paths``)"""
    target: str
    """Where the Install button and `install_into_blender` would put packages"""
    config: str

    def lines(self) -> tuple[str, ...]:
        out = [f"Python {self.version} at {self.python}", f"looked in {len(self.searched)} folders:"]
        out += [f"  {p}" for p in self.searched]
        out.append(f"would install into {self.target}")
        out.append(f"config: {self.config}" + ("" if Path(self.config).exists() else " (not present)"))
        return tuple(out)


def default_target() -> Path:
    """Where packages go unless told otherwise: the configured ``packages`` folder, else (in Blender) its user
    ``scripts/modules``, else this Python's user site-packages."""
    from proctune import config  # noqa: PLC0415

    cfg = config.load()
    if cfg.packages is not None:
        return cfg.packages
    if importlib.util.find_spec("bpy") is not None:
        return blender_modules()
    import site  # noqa: PLC0415

    return Path(site.getusersitepackages())


def environment() -> Environment:
    from proctune import config  # noqa: PLC0415

    searched = tuple(p for p in sys.path if p and Path(p).is_dir())
    try:
        configured = tuple(str(p) for p in config.load().paths)
        target = str(default_target())
    except ProctuneError as e:  # a broken config must not hide the report
        configured, target = (), f"(unknown: {e.message})"
    v = sys.version_info
    return Environment(sys.executable, f"{v.major}.{v.minor}", searched, configured, target, str(config.config_path()))


def version_mismatch(folder: Path) -> str | None:
    """A site-packages folder made for another Python version than this one (its compiled packages will not load)."""
    v = sys.version_info
    for part in folder.parts:
        if (
            part.startswith("python")
            and part[6:].replace(".", "").isdigit()
            and part[6:] not in (f"{v.major}.{v.minor}", "")
        ):
            return (
                f"{folder} is for Python {part[6:]}, this is {v.major}.{v.minor}: its compiled packages will not load"
            )
    return None


def install_hint(packages: Iterable[str]) -> str:
    """How to make packages importable, naming where this Python looked and where an install would go."""
    names = sorted(set(packages))
    env = environment()
    src = str(Path(__file__).resolve().parents[1])  # where proctune itself is imported from
    expr = f"import sys; sys.path.insert(0, {src!r}); from proctune import deps; deps.install_into_blender({names!r})"
    return (
        f"Python {env.version} at {env.python} searched {len(env.searched)} folders (listed in the log). "
        f"To use packages from your own environment, add its site-packages for Python {env.version} to `paths` in "
        f"{env.config} (or PROCTUNE_PATHS). To install them into {env.target}: proctune panel > Install missing "
        f'packages, or run `blender -b --python-expr "{expr}"`'
    )


def require(reqs: Iterable[Requirement], context: str) -> None:
    """Raise `MissingDependenciesError` naming every missing package and how to install it."""
    gone = missing(reqs)
    if gone:
        log.warning("missing %s; %s", [r.package for r in gone], "\n".join(environment().lines()))
        what = ", ".join(f"{r.package} ({r.why})" if r.why else r.package for r in gone)
        raise MissingDependenciesError(
            f"{context} needs {what}", install_hint(r.package for r in gone), tuple(r.package for r in gone)
        )


@dataclass(frozen=True)
class InstallResult:
    installed: tuple[str, ...]
    kept: tuple[str, ...]
    """Already present in this interpreter, not installed again"""
    target: Path


def _present(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _pip(args: list[str], python: str) -> str:
    cmd = [python, "-m", "pip", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        raise MissingDependenciesError(
            f"cannot run Python at {python}: {e.strerror or e}",
            "inside Blender this is Blender's own Python (sys.executable); elsewhere pass the Python to install for",
            tuple(a for a in args if not a.startswith("-")),
        ) from e
    if r.returncode != 0 and "No module named pip" in r.stderr and args[:1] != ["--version"]:
        # official Blender builds may ship Python without pip: bootstrap it once (standard library), then retry
        boot = subprocess.run([python, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True, check=False)
        if boot.returncode == 0:
            log.info("bootstrapped pip for %s", python)
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
        raise MissingDependenciesError(
            f"pip failed ({' '.join(cmd[:4])} ...): {' / '.join(tail)}",
            f"check the network and that {python} has pip (`{python} -m ensurepip`)",
            tuple(a for a in args if not a.startswith("-")),
        )
    return r.stdout


def install(packages: Iterable[str], target: Path, python: str | None = None) -> InstallResult:
    """Install `packages` and what they need into `target`, leaving out what `python` (by default this interpreter)
    already has."""
    py = python or sys.executable
    pkgs = sorted(set(packages))
    if not pkgs:
        return InstallResult((), (), target)
    report = json.loads(_pip(["install", "--dry-run", "--ignore-installed", "--quiet", "--report", "-", *pkgs], py))
    wanted = [(i["metadata"]["name"], i["metadata"]["version"]) for i in report["install"]]
    kept = tuple(n for n, _ in wanted if _present(n))
    todo = [f"{n}=={v}" for n, v in wanted if n not in kept]
    target.mkdir(parents=True, exist_ok=True)
    if todo:
        _pip(["install", "--quiet", "--no-deps", "--target", str(target), *todo], py)
    log.info("installed %s into %s (kept %s)", todo, target, kept)
    if str(target) not in sys.path:
        sys.path.append(str(target))
    importlib.invalidate_caches()
    return InstallResult(tuple(todo), kept, target)


def blender_modules() -> Path:
    """Blender's user ``scripts/modules`` (only inside Blender; not created here: `install` creates its target)."""
    import bpy  # noqa: PLC0415

    return Path(bpy.utils.user_resource("SCRIPTS", path="modules"))


def ensure_on_path(target: Path) -> None:
    """Blender puts ``scripts/modules`` on its path only if it existed at startup; last, so Blender's own wins."""
    if target.is_dir() and str(target) not in sys.path:
        sys.path.append(str(target))


def install_into_blender(packages: Iterable[str], target: Path | None = None) -> InstallResult:
    """Run inside Blender: install into `target`, else the configured folder (`proctune.config`, ``packages``), else
    Blender's user ``scripts/modules``."""
    return install(packages, target or default_target())


def add_paths(folders: Iterable[Path]) -> tuple[str, ...]:
    """Put folders on the import path (after what is there, so the interpreter's own packages win); returns warnings
    for folders that do not exist or are for another Python version."""
    warnings: list[str] = []
    for folder in folders:
        if not folder.is_dir():
            warnings.append(f"{folder} does not exist")
            continue
        if mismatch := version_mismatch(folder):
            warnings.append(mismatch)
        ensure_on_path(folder)
    for w in warnings:
        log.warning("%s", w)
    return tuple(warnings)


def main(argv: list[str] | None = None) -> int:
    """``python -m proctune.deps PACKAGE...``: install for this interpreter into ``--target`` (default: user site)."""
    import argparse  # noqa: PLC0415
    import site  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python -m proctune.deps", description=main.__doc__)
    ap.add_argument("packages", nargs="*", help="pip packages (default: what proctune.core needs and lacks)")
    ap.add_argument("--target", type=Path, default=Path(site.getusersitepackages()))
    args = ap.parse_args(argv)
    pkgs = args.packages or [r.package for r in missing(CORE)]
    if not pkgs:
        print("nothing missing")
        return 0
    r = install(pkgs, args.target)
    done, kept = ", ".join(r.installed) or "nothing", ", ".join(r.kept) or "none"
    print(f"installed {done} into {r.target}; already present: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
