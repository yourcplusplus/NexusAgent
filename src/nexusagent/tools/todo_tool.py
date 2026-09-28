from __future__ import annotations

import json
from typing import Any

from nexusagent.core.state import RuntimeState

VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}
TODO_FILE = "TODO.md"


def _normalize_items(items: Any) -> list[str]:
    if isinstance(items, str):
        stripped = items.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [line.strip("- ").strip() for line in stripped.splitlines() if line.strip()]
        return _normalize_items(decoded)
    if isinstance(items, dict):
        value = items.get("content") or items.get("description") or items.get("title") or items.get("text") or items.get("command")
        if value:
            return [str(value).strip()]
        normalized: list[str] = []
        for key, item in items.items():
            child_items = _normalize_items(item)
            normalized.extend(child_items or [str(key).strip()])
        return [item for item in normalized if item]
    if isinstance(items, list):
        normalized: list[str] = []
        for item in items:
            normalized.extend(_normalize_items(item))
        return [item for item in normalized if item]
    return [str(items).strip()] if items is not None and str(items).strip() else []


def write_todos(
    todos: list[str],
    acceptance_criteria: list[str],
    verification_commands: list[str],
) -> dict[str, Any]:
    cleaned_todos = _normalize_items(todos)
    cleaned_criteria = _normalize_items(acceptance_criteria)
    cleaned_commands = _normalize_items(verification_commands)

    return {
        "ok": bool(cleaned_todos and cleaned_criteria and cleaned_commands),
        "todos": cleaned_todos,
        "acceptance_criteria": cleaned_criteria,
        "verification_commands": cleaned_commands,
    }


def update_todo(
    todos: list[dict[str, str]],
    todo_id: str,
    status: str,
    note: str = "",
) -> dict[str, Any]:
    if status not in VALID_TODO_STATUSES:
        return {
            "ok": False,
            "error": f"status must be one of: {', '.join(sorted(VALID_TODO_STATUSES))}",
            "todos": todos,
        }

    updated: list[dict[str, str]] = []
    found = False
    for todo in todos:
        item = dict(todo)
        if item.get("id") == todo_id:
            item["status"] = status
            item["note"] = note
            found = True
        updated.append(item)

    if not found:
        return {"ok": False, "error": f"unknown todo_id: {todo_id}", "todos": todos}
    return {"ok": True, "todo_id": todo_id, "status": status, "note": note, "todos": updated}


def persist_todos(
    state: RuntimeState,
    todos: list[dict[str, Any]],
    acceptance_criteria: list[str] | None = None,
    verification_commands: list[str] | None = None,
    plan_summary: str = "",
) -> dict[str, Any]:
    path = state.assert_workspace_path(state.workspace / TODO_FILE)
    content = render_todo_markdown(todos, acceptance_criteria or [], verification_commands or [], plan_summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    state.record_read(path, complete=True)
    return {"ok": True, "path": TODO_FILE, "lines": len(content.splitlines()), "todos": todos}


def render_todo_markdown(
    todos: list[dict[str, Any]],
    acceptance_criteria: list[str],
    verification_commands: list[str],
    plan_summary: str = "",
) -> str:
    lines = ["# NexusAgent Todo", ""]
    if plan_summary:
        lines.extend(["## Plan", "", plan_summary, ""])
    lines.extend(["## Todos", ""])
    if todos:
        for todo in todos:
            status = str(todo.get("status", "pending"))
            box = {"pending": " ", "in_progress": "-", "completed": "x", "blocked": "!"}.get(status, " ")
            note = str(todo.get("note", ""))
            note_text = f" — {note}" if note else ""
            lines.append(f"- [{box}] **{todo.get('id', '')}** `{status}` {todo.get('content', '')}{note_text}")
    else:
        lines.append("- [ ] No todos yet.")
    if acceptance_criteria:
        lines.extend(["", "## Acceptance Criteria", ""])
        lines.extend(f"- {item}" for item in acceptance_criteria)
    if verification_commands:
        lines.extend(["", "## Verification Commands", ""])
        lines.extend(f"- `{command}`" for command in verification_commands)
    lines.append("")
    return "\n".join(lines)
