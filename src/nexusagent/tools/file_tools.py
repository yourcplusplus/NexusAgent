from __future__ import annotations

import bisect
import difflib
from pathlib import Path
from typing import Any

from nexusagent.core.state import RuntimeState

MAX_READ_LINES = 2000
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gbk")


def _error(code: str, message: str, **detail: Any) -> dict[str, Any]:
    """结构化工具错误:保留旧 `error` 字符串(兼容现有消费方),
    新增机器可读的 error_code 与可选诊断/修复提示。"""
    return {"ok": False, "error": message, "error_code": code, **detail}


def _strip_workspace_prefix(file_path: str) -> str:
    normalized = file_path.replace("\\", "/").strip()
    while normalized in {"workspace", "./workspace"} or normalized.startswith(("workspace/", "./workspace/")):
        if normalized in {"workspace", "./workspace"}:
            normalized = "."
        elif normalized.startswith("./workspace/"):
            normalized = normalized[len("./workspace/") :]
        else:
            normalized = normalized[len("workspace/") :]
    return normalized


def read_text_lossy(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        return path.read_text(encoding="utf-8", errors="replace")
    return path.read_text(encoding="utf-8")


def resolve_workspace_path(state: RuntimeState, file_path: str) -> Path:
    raw = Path(_strip_workspace_prefix(file_path)).expanduser()
    if not raw.is_absolute():
        raw = state.workspace / raw
    return state.assert_workspace_path(raw)


def display_path(state: RuntimeState, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(state.workspace.resolve()))
    except ValueError:
        return str(path)


def read_file(
    state: RuntimeState,
    file_path: str,
    offset: int | str = 0,
    limit: int | str = MAX_READ_LINES,
) -> dict[str, Any]:
    try:
        path = resolve_workspace_path(state, file_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not path.exists():
        return {"ok": False, "error": f"file does not exist: {display_path(state, path)}"}
    if not path.is_file():
        return {"ok": False, "error": f"path is not a file: {display_path(state, path)}"}
    try:
        offset_value = int(offset)
        limit_value = int(limit)
    except (TypeError, ValueError):
        return {"ok": False, "error": "offset and limit must be integers"}
    if offset_value < 0:
        return {"ok": False, "error": "offset must be >= 0"}
    if limit_value <= 0:
        return {"ok": False, "error": "limit must be > 0"}

    text = read_text_lossy(path)
    lines = text.splitlines()
    limit_value = min(limit_value, MAX_READ_LINES)
    selected = lines[offset_value : offset_value + limit_value]
    complete = offset_value == 0 and len(selected) == len(lines)
    state.record_read(path, complete=complete)
    state.record_touch(path, op="read", via="file_tools")

    numbered = "\n".join(f"{offset_value + idx + 1:>6}\t{line}" for idx, line in enumerate(selected))
    result = {
        "ok": True,
        "path": display_path(state, path),
        "total_lines": len(lines),
        "offset": offset_value,
        "limit": limit_value,
        "complete": complete,
        "content": numbered,
    }
    if not complete and offset_value + len(selected) < len(lines):
        result["next_offset"] = offset_value + len(selected)
    return result


def write_file(state: RuntimeState, file_path: str, content: str) -> dict[str, Any]:
    try:
        path = resolve_workspace_path(state, file_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    existed = path.exists()
    if existed and path.is_dir():
        return _error("NOT_A_FILE", f"path is a directory: {display_path(state, path)}")

    if existed:
        snapshot = state.snapshot_for(path)
        if snapshot is None:
            return _error(
                "NOT_READ",
                "file has not been read yet. Read it before overwriting.",
                hint=f'Call FileReadTool(file_path="{display_path(state, path)}") first, then rewrite with the full intended content.',
            )
        if path.stat().st_mtime_ns != snapshot.mtime_ns:
            return _error(
                "STALE_READ",
                "file changed after it was read. Read it again before writing.",
                hint="The file changed since the last read (a bash command may have touched it). Re-read, then write.",
            )
        original = read_text_lossy(path)
    else:
        original = ""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return _error(
            "WRITE_FAILED",
            f"{type(exc).__name__}: {exc}",
            hint="Nothing was written. Check permissions/path length, then retry.",
        )
    state.record_read(path, complete=True)
    state.record_touch(path, op="write", via="file_tools")

    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            content.splitlines(),
            fromfile=f"a/{display_path(state, path)}",
            tofile=f"b/{display_path(state, path)}",
            lineterm="",
        )
    )
    return {
        "ok": True,
        "type": "update" if existed else "create",
        "path": display_path(state, path),
        "lines": len(content.splitlines()),
        "diff": diff[:4000],
    }


def edit_file(state: RuntimeState, file_path: str, old_text: str, new_text: str) -> dict[str, Any]:
    try:
        path = resolve_workspace_path(state, file_path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not path.exists():
        return {"ok": False, "error": f"file does not exist: {display_path(state, path)}"}

    snapshot = state.snapshot_for(path)
    if snapshot is None:
        return _error(
            "NOT_READ",
            "file has not been read yet. Read it before editing.",
            hint=f'Call FileReadTool(file_path="{display_path(state, path)}") and retry with an exact snippet.',
        )
    if path.stat().st_mtime_ns != snapshot.mtime_ns:
        return _error(
            "STALE_READ",
            "file changed after it was read. Read it again before editing.",
            hint="The file changed since the last read (a bash command may have touched it). Re-read, then edit.",
        )
    if not old_text:
        return _error("EMPTY_OLD_TEXT", "old_text must not be empty")

    original = read_text_lossy(path)
    count = original.count(old_text)
    if count == 0:
        return _error(
            "NOT_FOUND",
            "old_text was not found",
            **_not_found_diagnosis(original, old_text),
            hint="Copy the snippet exactly from FileReadTool output (leading spaces and line breaks matter). Extend with neighboring lines if uniqueness requires it.",
        )
    if count > 1:
        return _error(
            "AMBIGUOUS_MATCH",
            f"old_text matched {count} times. Provide a unique snippet.",
            occurrences=_match_occurrences(original, old_text),
            hint="occurrences lists the matching line numbers; include surrounding lines so the snippet matches exactly once.",
        )

    updated = original.replace(old_text, new_text, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return _error(
            "WRITE_FAILED",
            f"{type(exc).__name__}: {exc}",
            hint="The file was NOT modified.",
        )
    state.record_read(path, complete=True)
    state.record_touch(path, op="edit", via="file_tools")

    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=f"a/{display_path(state, path)}",
            tofile=f"b/{display_path(state, path)}",
            lineterm="",
        )
    )
    return {
        "ok": True,
        "path": display_path(state, path),
        "replacements": 1,
        "diff": diff[:4000],
    }


def _line_starts(lines: list[str]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    return starts


def _match_occurrences(text: str, needle: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """定位每个匹配的行号 + 首行预览,让模型知道往哪几行补充上下文。"""
    starts = _line_starts(text.splitlines())
    occurrences: list[dict[str, Any]] = []
    search_from = 0
    while len(occurrences) < limit:
        idx = text.find(needle, search_from)
        if idx == -1:
            break
        line_no = bisect.bisect_right(starts, idx)
        preview = text[idx : idx + 120].splitlines()[0]
        occurrences.append({"line": line_no, "preview": preview})
        search_from = idx + max(len(needle), 1)
    return occurrences


def _not_found_diagnosis(text: str, old_text: str) -> dict[str, Any]:
    """NOT_FOUND 的两种最常见病因:缩进不匹配(可精确判定)、内容近似(模糊提示)。"""
    detail: dict[str, Any] = {}
    stripped_text = "\n".join(line.strip() for line in text.splitlines())
    stripped_old = "\n".join(line.strip() for line in old_text.splitlines())
    if stripped_old and stripped_old in stripped_text:
        detail["likely_cause"] = "whitespace_or_indentation_mismatch"
    else:
        first_line = old_text.strip().splitlines()[0] if old_text.strip() else ""
        candidates = [line for line in text.splitlines() if line.strip()]
        close = difflib.get_close_matches(first_line, candidates, n=3, cutoff=0.6)
        if close:
            detail["near_matches"] = close
    return detail
