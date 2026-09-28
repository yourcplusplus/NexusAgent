from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest project root marker from ``start`` upward."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def default_workspace(root: Path | None = None) -> Path:
    return new_task_workspace(root)


def default_workspace_root(root: Path | None = None) -> Path:
    return (root or find_project_root()) / ".nexusagent" / "workspaces"


def new_task_workspace(root: Path | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6]
    return default_workspace_root(root) / f"workspace-{stamp}-{suffix}"
