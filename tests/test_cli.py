"""The proctune command: what it hands to Blender, without starting one."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proctune import cli

PROJECT = Path(__file__).resolve().parents[1] / "examples" / "limb" / "limb.json"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def found(exe: str) -> str:
        return f"/usr/bin/{exe}"

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", found)
    return seen


def test_open_passes_blender_arguments_and_the_project(calls: list[list[str]]) -> None:
    r = CliRunner().invoke(cli.app, ["open", str(PROJECT), "--skip-check", "--no-compare", "--", "--factory-startup"])
    assert r.exit_code == 0, r.output
    cmd = calls[-1]
    expr = cmd.index("--python-expr")
    assert cmd[1:expr] == ["--factory-startup"]  # Blender's own arguments come first
    assert "from proctune.blender.launch import main" in cmd[expr + 1]
    assert cmd[cmd.index("--") + 1 :] == [PROJECT.as_posix(), "--no-compare"]


def test_open_checks_first_unless_told_not_to(calls: list[list[str]]) -> None:
    CliRunner().invoke(cli.app, ["open", str(PROJECT)])
    assert "-b" in calls[0]  # the check runs Blender in the background first
    assert "from proctune.blender.preflight import main" in calls[0][calls[0].index("--python-expr") + 1]
    assert "-b" not in calls[1]


def test_a_missing_project_or_blender_is_explained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    r = CliRunner().invoke(cli.app, ["check", str(tmp_path / "nope.json")])
    assert r.exit_code == 2
    assert "no project at" in r.output

    def not_found(_exe: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", not_found)
    r = CliRunner().invoke(cli.app, ["check", str(PROJECT), "--blender", "not-blender"])
    assert r.exit_code == 2
    assert "Blender not found" in r.output
