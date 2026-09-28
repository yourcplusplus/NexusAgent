from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from nexusagent.core.state import RuntimeState
from nexusagent.tools.file_tools import display_path, read_text_lossy, resolve_workspace_path


SKIP_DIRS = {".git", ".nexusagent", ".venv", "__pycache__", ".pytest_cache"}


def _iter_files(root: Path, glob_pattern: str | None) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if glob_pattern and not fnmatch.fnmatch(path.name, glob_pattern) and not fnmatch.fnmatch(str(path), glob_pattern):
            continue
        files.append(path)
    return files


def grep(
    state: RuntimeState,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    head_limit: int | str = 50,
    ignore_case: bool = False,
) -> dict[str, Any]:
    if not pattern:
        return {"ok": False, "error": "pattern must not be empty"}
    try:
        head_limit_value = int(head_limit)
    except (TypeError, ValueError):
        return {"ok": False, "error": "head_limit must be an integer"}
    if head_limit_value <= 0:
        return {"ok": False, "error": "head_limit must be > 0"}

    root = resolve_workspace_path(state, path)
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = _iter_files(root, glob)
    else:
        return {"ok": False, "error": f"path does not exist: {display_path(state, root)}"}

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return {"ok": False, "error": f"invalid regex: {exc}"}

    matches: list[dict[str, Any]] = []
    for file in candidates:
        lines = read_text_lossy(file).splitlines()
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append({"path": display_path(state, file), "line": idx, "text": line})
                if len(matches) >= head_limit_value:
                    return {"ok": True, "pattern": pattern, "matches": matches, "truncated": True}

    return {"ok": True, "pattern": pattern, "matches": matches, "truncated": False}
