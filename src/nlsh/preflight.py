from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from nlsh.compiler import TOOL_PACKAGES


BOOTSTRAP_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_system_deps.sh"


@dataclass(slots=True)
class MissingToolsError(RuntimeError):
    missing_tools: list[str]

    @property
    def missing_packages(self) -> list[str]:
        return [TOOL_PACKAGES[tool] for tool in self.missing_tools]

    def __str__(self) -> str:
        tools = ", ".join(self.missing_tools)
        packages = " ".join(self.missing_packages)
        return (
            f"Missing required system tools: {tools}\n"
            f"Install the matching apt packages: {packages}\n"
            f"Bootstrap script: {BOOTSTRAP_SCRIPT}"
        )


def find_missing_tools(tools: list[str]) -> list[str]:
    return [tool for tool in tools if shutil.which(tool) is None]


def ensure_required_tools(tools: list[str]) -> None:
    missing = find_missing_tools(tools)
    if missing:
        raise MissingToolsError(sorted(missing))
