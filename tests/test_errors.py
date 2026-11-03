"""Failures attributable to proctune: every one raises a proctune error that names the file or model and says what to
do, and none leaves a half-written file behind."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import limb
import numpy as np
import pytest

from proctune import config, deps
from proctune.core import Session
from proctune.core import files as F
from proctune.core.session import load_model
from proctune.errors import MissingDependenciesError, ModelError, ProctuneError, ProjectError

EXAMPLE = Path(limb.__file__).parent
ROOT_USER = hasattr(os, "geteuid") and os.geteuid() == 0
needs_permissions = pytest.mark.skipif(ROOT_USER, reason="root ignores file permissions")


def _session() -> Session[limb.Limb]:
    return Session.new(limb.MODEL, "limb:MODEL", (str(EXAMPLE),))


def _project(tmp_path: Path, **override: object) -> Path:
    doc: dict[str, object] = {"format": 1, "model": "limb:MODEL", "import_paths": [str(EXAMPLE)], "params": {}}
    doc.update(override)
    path = tmp_path / "p.json"
    path.write_text(json.dumps(doc))
    return path


def _remedy(e: pytest.ExceptionInfo[ProctuneError]) -> str:
    assert e.value.remedy, f"no remedy for: {e.value.message}"
    return e.value.remedy


@pytest.fixture
def readonly(tmp_path: Path) -> Iterator[Path]:
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o555)
    yield d
    d.chmod(0o755)


# ------------------------------------------------------------------------------------------------- writing


@needs_permissions
@pytest.mark.skipif(sys.platform == "win32", reason="Windows ignores the read-only flag on folders")
def test_save_into_a_read_only_folder(readonly: Path) -> None:
    with pytest.raises(ProjectError, match="permission denied") as e:
        _session().save(readonly / "p.json")
    assert "writable" in _remedy(e)
    assert list(readonly.iterdir()) == []  # no temporary file left


@needs_permissions
def test_save_over_a_read_only_file_keeps_it(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("old")
    path.chmod(0o444)
    try:
        with pytest.raises(ProjectError, match="permission denied"):
            _session().save(path)
        assert path.read_text() == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["p.json"]
    finally:
        path.chmod(0o644)


def test_save_into_a_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="does not exist"):
        _session().save(tmp_path / "nope" / "p.json")


def test_save_handles_spaces_and_unicode(tmp_path: Path) -> None:
    d = tmp_path / "Projekt äöü 手"
    d.mkdir()
    s = _session()
    s.save(d / "mein projekt.json")
    assert Session.load(d / "mein projekt.json").base.upper.length == s.base.upper.length


def test_a_failed_sculpt_write_keeps_the_old_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _session()
    r = s.evaluate().elements["muscle"]
    s.capture_sculpt("muscle", r.verts + 0.001)
    path = tmp_path / "p.json"
    s.save(path)
    before = (path.read_bytes(), path.with_suffix(".npz").read_bytes())
    real = F.write_atomic

    def fail_on_npz(p: Path, data: bytes) -> None:
        if p.suffix == ".npz":
            raise OSError(28, "No space left on device")
        real(p, data)

    monkeypatch.setattr(F, "write_atomic", fail_on_npz)
    s.set(("upper", "length"), 0.4)
    with pytest.raises(ProjectError, match="disk is full"):
        s.save(path)
    assert path.with_suffix(".npz").read_bytes() == before[1]


# ------------------------------------------------------------------------------------------------- reading


def test_load_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="does not exist"):
        Session.load(tmp_path / "nope.json")


def test_load_broken_json_says_where(tmp_path: Path) -> None:
    (tmp_path / "p.json").write_text('{"format": 1,\n "model": }')
    with pytest.raises(ProjectError, match="line 2, column"):
        Session.load(tmp_path / "p.json")


def test_load_not_a_project(tmp_path: Path) -> None:
    (tmp_path / "p.json").write_text("[1, 2]")
    with pytest.raises(ProjectError, match="not a proctune project"):
        Session.load(tmp_path / "p.json")


def test_load_another_format(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="format 99"):
        Session.load(_project(tmp_path, format=99))


def test_out_of_range_values_are_listed(tmp_path: Path) -> None:
    params = {"upper": {"length": 5.0}, "joint_deg": -40.0}
    with pytest.raises(ProjectError, match=r"upper\.length = 5\.0 \(allowed 0\.1 to 0\.6\).*joint_deg") as e:
        Session.load(_project(tmp_path, params=params))
    assert "default" in _remedy(e)


def test_unknown_parameters_are_named(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="unknown parameters"):
        Session.load(_project(tmp_path, params={"upper": {"lenght": 0.3}}))


def test_invalid_curve_points(tmp_path: Path) -> None:
    params = {"growth": {"curve": {"points": [[0.5, 0.0], [0.5, 1.0]]}}}
    with pytest.raises(ProjectError, match="increasing x"):
        Session.load(_project(tmp_path, params=params))


def test_corrupt_sculpt_file(tmp_path: Path) -> None:
    path = _project(tmp_path)
    path.with_suffix(".npz").write_bytes(b"not a zip")
    with pytest.raises(ProjectError, match="sculpts cannot be read") as e:
        Session.load(path)
    assert "move it away" in _remedy(e)


# ------------------------------------------------------------------------------------------------- models


def _module(tmp_path: Path, name: str, code: str) -> str:
    (tmp_path / f"{name}.py").write_text(code)
    return str(tmp_path)


def test_model_name_must_be_module_and_attribute() -> None:
    with pytest.raises(ProjectError, match="not module:attribute"):
        load_model("limb")


def test_missing_model_module_lists_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match=re.escape(f"not found (looked in {tmp_path}")) as e:
        load_model("no_such_model_xyz:MODEL", (str(tmp_path),))
    assert "import_paths" in _remedy(e)


def test_a_package_the_model_needs_is_named(tmp_path: Path) -> None:
    d = _module(tmp_path, "needs_pkg_model", "import yaml_not_installed_xyz\nMODEL = None\n")
    with pytest.raises(MissingDependenciesError, match="yaml_not_installed_xyz") as e:
        load_model("needs_pkg_model:MODEL", (d,))
    assert e.value.packages == ("yaml_not_installed_xyz",)
    assert "install_into_blender" in _remedy(e)


def test_a_model_that_crashes_on_import(tmp_path: Path) -> None:
    d = _module(tmp_path, "crashing_model", "raise RuntimeError('boom')\n")
    with pytest.raises(ModelError, match="RuntimeError: boom"):
        load_model("crashing_model:MODEL", (d,))


def test_a_module_without_the_model(tmp_path: Path) -> None:
    d = _module(tmp_path, "empty_model", "X = 1\n")
    with pytest.raises(ProjectError, match="has no 'MODEL'"):
        load_model("empty_model:MODEL", (d,))


def test_a_stage_that_crashes_names_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _session()

    def broken(*_: object) -> tuple[()]:
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(limb, "fat", broken)
    with pytest.raises(ModelError, match="stage 'fat' failed: ZeroDivisionError") as e:
        s.evaluate()
    assert "last good mesh" in _remedy(e)


# ------------------------------------------------------------------------------------------------- packages and config


def test_missing_packages_are_named_with_their_pip_names() -> None:
    reqs = (
        deps.requirement("yaml_xyz_missing"),
        deps.requirement("yaml"),
        deps.Requirement("np_missing_xyz", "numpy-x", "arrays"),
    )
    with pytest.raises(MissingDependenciesError) as e:
        deps.require(reqs, "a model")
    assert "numpy-x (arrays)" in e.value.message
    assert "yaml_xyz_missing" in e.value.packages
    assert deps.requirement("yaml.loader").package == "PyYAML"


def test_installing_drops_what_the_interpreter_has(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    report = {
        "install": [
            {"metadata": {"name": "numpy", "version": "2.0"}},
            {"metadata": {"name": "tiny-x", "version": "1.0"}},
        ]
    }

    def pip(args: list[str], _python: str) -> str:
        calls.append(args)
        return json.dumps(report) if "--dry-run" in args else ""

    monkeypatch.setattr(deps, "_pip", pip)
    r = deps.install(["tiny-x"], tmp_path / "site")
    assert r.installed == ("tiny-x==1.0",)
    assert r.kept == ("numpy",)
    assert "--no-deps" in calls[-1]
    sys.path.remove(str(tmp_path / "site"))


def test_a_missing_interpreter_is_explained(tmp_path: Path) -> None:
    with pytest.raises(MissingDependenciesError, match="cannot run Python at") as e:
        deps.install(["anything"], tmp_path, python=sys.executable + "-nope")
    assert "sys.executable" in _remedy(e)


def test_pip_failure_is_explained(tmp_path: Path) -> None:
    with pytest.raises(MissingDependenciesError, match="pip failed") as e:
        deps.install(["definitely-not-a-package-proctune-xyz"], tmp_path, python=sys.executable)
    assert "ensurepip" in _remedy(e)


def test_config_file_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('blender = "/opt/blender/blender"\npackages = "~/pkgs"\n')
    monkeypatch.setenv("PROCTUNE_CONFIG", str(cfg))
    monkeypatch.setenv("PROCTUNE_MAGICK", "/usr/local/bin/magick")
    c = config.load()
    assert c.blender == "/opt/blender/blender"
    assert c.packages == Path.home() / "pkgs"
    assert c.magick == "/usr/local/bin/magick"


def test_config_errors_name_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("PROCTUNE_CONFIG", str(cfg))
    cfg.write_text("blendr = 'x'\n")
    with pytest.raises(ProctuneError, match="unknown keys"):
        config.load()
    cfg.write_text("blender = \n")
    with pytest.raises(ProctuneError, match="cannot read the proctune config"):
        config.load()


def test_config_lives_in_appdata_on_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROCTUNE_CONFIG", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert config.config_path() == tmp_path / "Roaming" / "proctune" / "config.toml"


def test_paths_across_drives_fall_back_to_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert F.relative_or_absolute(tmp_path / "a" / "b", tmp_path / "c") == os.path.join("..", "a", "b")

    def no_relpath(*_: object) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, "relpath", no_relpath)
    assert F.relative_or_absolute(tmp_path / "a", tmp_path) == str((tmp_path / "a").resolve())


def test_numpy_is_never_installed_twice() -> None:
    """Blender's own numpy must stay the one imported: it is in CORE but always present where proctune runs."""
    assert deps.missing(deps.CORE[:1]) == ()
    assert np.__version__


# ------------------------------------------------------------------------------------------------- where packages are


def test_the_install_command_in_the_hint_runs_as_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(deps, "install_into_blender", lambda names, target=None: calls.append(list(names)))  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    hint = deps.install_hint(["scikit-image"])
    expr = re.search(r'--python-expr "(.+?)"`', hint)
    assert expr is not None
    exec(expr.group(1), {})
    assert calls == [["scikit-image"]]


def test_the_hint_says_which_python_looked_and_where_it_would_install() -> None:
    env = deps.environment()
    hint = deps.install_hint(["x"])
    assert env.python in hint
    assert env.version in hint
    assert env.target in hint
    assert "paths" in hint
    assert any(line.startswith("  ") for line in env.lines())  # the searched folders, one per line


def test_folders_for_another_python_are_flagged(tmp_path: Path) -> None:
    other = tmp_path / "venv" / "lib" / "python3.9" / "site-packages"
    other.mkdir(parents=True)
    v = sys.version_info
    same = tmp_path / "venv2" / "lib" / f"python{v.major}.{v.minor}" / "site-packages"
    same.mkdir(parents=True)
    assert deps.version_mismatch(other) is not None
    assert deps.version_mismatch(same) is None
    warnings = deps.add_paths([other, same, tmp_path / "nope"])
    assert any("3.9" in w for w in warnings)
    assert any("does not exist" in w for w in warnings)
    assert str(same) in sys.path
    for p in (str(other), str(same)):
        if p in sys.path:
            sys.path.remove(p)


def test_paths_from_the_config_file_and_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('paths = ["~/a", "/b"]\n')
    monkeypatch.setenv("PROCTUNE_CONFIG", str(cfg))
    assert config.load().paths == (Path.home() / "a", Path("/b"))
    monkeypatch.setenv("PROCTUNE_PATHS", os.pathsep.join(["/c", "/d"]))
    assert config.load().paths == (Path("/c"), Path("/d"))
    cfg.write_text('paths = "/not/a/list"\n')
    monkeypatch.delenv("PROCTUNE_PATHS")
    with pytest.raises(ProctuneError, match="list of folder paths"):
        config.load()


def test_a_project_says_where_its_packages_are(tmp_path: Path) -> None:
    pkgs = tmp_path / "pkgs"
    pkgs.mkdir()
    path = _project(tmp_path, package_paths=["pkgs"])
    s = Session.load(path)
    assert str(pkgs.resolve()) in sys.path
    assert s.package_paths == ("pkgs",)
    s.save(tmp_path / "again.json")
    assert json.loads((tmp_path / "again.json").read_text())["package_paths"] == ["pkgs"]
    sys.path.remove(str(pkgs.resolve()))


def test_a_package_missing_while_building_is_a_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _session()

    def needs_it(*_: object) -> tuple[()]:
        import pkg_imported_late_xyz  # noqa: F401, PLC0415  # pyright: ignore[reportMissingImports, reportUnusedImport]

        return ()

    monkeypatch.setattr(limb, "fat", needs_it)
    with pytest.raises(MissingDependenciesError, match="stage 'fat' imports pkg_imported_late_xyz") as e:
        s.evaluate()
    assert e.value.packages == ("pkg_imported_late_xyz",)


PREFLIGHT_BLENDER = config.load().blender


@pytest.mark.skipif(sys.platform != "linux" or shutil.which(PREFLIGHT_BLENDER) is None, reason="needs blender (Linux)")
def test_check_reports_and_installs_nothing_unless_asked(tmp_path: Path) -> None:
    """`proctune check` builds the project in Blender's own Python and reports; it installs nothing."""
    (tmp_path / "needs_late_pkg_model.py").write_text("import pkg_preflight_missing_xyz\nMODEL = None\n")
    project = tmp_path / "p.json"
    project.write_text(
        '{"format": 2, "model": "needs_late_pkg_model:MODEL", "import_paths": ["."], "package_paths": ["pkgs"]}'
    )
    (tmp_path / "pkgs").mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "proctune.cli", "check", str(project), "--json"],
        capture_output=True, text=True, timeout=180, check=False,
    )  # fmt: skip
    assert r.returncode == 1, r.stderr
    report = json.loads(r.stdout.strip().splitlines()[-1])
    assert report["missing"] == ["pkg_preflight_missing_xyz"]
    assert report["installed"] == []
    assert any(line.startswith("looked in") for line in report["environment"])
    assert any(str(tmp_path / "pkgs") in line for line in report["environment"])  # the project's folder was searched
    assert list((tmp_path / "pkgs").iterdir()) == []


def test_the_staged_package_is_proctune_alone() -> None:
    """What `proctune open` puts on Blender's path holds the proctune package and nothing else from this
    environment (no numpy compiled for another Python)."""
    from proctune.stage import stage  # noqa: PLC0415

    root = stage()
    assert sorted(p.name for p in root.iterdir() if not p.name.startswith(".")) == ["proctune"]
    assert (root / "proctune" / "blender" / "launch.py").is_file()
    assert stage() == root  # unchanged package: not copied again
