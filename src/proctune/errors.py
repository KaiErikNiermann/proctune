"""proctune's errors: each says what went wrong and what to do about it (`remedy`), so a UI can show both."""

from __future__ import annotations


class ProctuneError(Exception):
    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def __str__(self) -> str:
        return f"{self.message} ({self.remedy})" if self.remedy else self.message


class ProjectError(ProctuneError):
    """A project file cannot be read or written."""


class EditError(ProctuneError):
    """A hand edit cannot be kept as it is."""


class ModelError(ProctuneError):
    """The model's own code failed while building."""


class MissingDependenciesError(ProctuneError):
    """Python packages are missing from the interpreter (Blender's, inside Blender)."""

    def __init__(self, message: str, remedy: str, packages: tuple[str, ...]) -> None:
        super().__init__(message, remedy)
        self.packages = packages
