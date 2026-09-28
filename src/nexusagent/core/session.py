from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nexusagent.core.checkpoint import workspace_manifest


SESSION_ROOT = Path(".nexusagent") / "session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
MAX_RECENT_TURNS = 18
MAX_TURN_CONTENT = 1800
MAX_SESSION_SUMMARY = 5000
MAX_SESSION_CONTEXT = 7000


def session_dir(workspace: Path) -> Path:
    return workspace / SESSION_ROOT


def session_file(workspace: Path) -> Path:
    return session_dir(workspace) / SESSION_FILE


def session_summary_file(workspace: Path) -> Path:
    return workspace / SESSION_SUMMARY_FILE


def load_or_create_session(workspace: Path) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    path = session_file(workspace)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
    else:
        raw = {}
    session = _normalize_session(raw, workspace)
    save_session(workspace, session)
    return session


def append_user_turn(session: dict[str, Any], content: str) -> int:
    turn = int(session.get("turn_index", 0)) + 1
    session["turn_index"] = turn
    session["last_task"] = content
    _append_turn(
        session,
        {
            "turn": turn,
            "role": "user",
            "route": "",
            "content": trim_text(content, MAX_TURN_CONTENT),
            "timestamp": utc_now(),
        },
    )
    return turn


def append_assistant_turn(
    session: dict[str, Any],
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    session["last_route"] = route
    session["last_final_answer"] = trim_text(content, MAX_TURN_CONTENT)
    _append_turn(
        session,
        {
            "turn": turn,
            "role": "assistant",
            "route": route,
            "content": trim_text(content, MAX_TURN_CONTENT),
            "summary": trim_text(summary or content, 700),
            "timestamp": utc_now(),
        },
    )


def save_session(workspace: Path, session: dict[str, Any]) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    session = _normalize_session(session, workspace)
    _compact_session(session)
    session["updated_at"] = utc_now()
    path = session_file(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    session_summary_file(workspace).write_text(build_session_summary_markdown(workspace, session), encoding="utf-8")
    return session


def build_session_context(workspace: Path, session: dict[str, Any] | None = None) -> str:
    session = _normalize_session(session or load_or_create_session(workspace), workspace)
    manifest = workspace_manifest(workspace, limit=40)
    files = [
        item.get("path", "")
        for item in manifest
        if item.get("type") == "file" and not str(item.get("path", "")).startswith(".nexusagent/session/")
    ][:30]
    recent_turns = [
        {
            "turn": turn.get("turn"),
            "role": turn.get("role", ""),
            "route": turn.get("route", ""),
            "content": trim_text(str(turn.get("content", "")), 600),
            "summary": trim_text(str(turn.get("summary", "")), 300),
        }
        for turn in session.get("recent_turns", [])[-10:]
    ]
    payload = {
        "session_id": session.get("session_id", ""),
        "turn_index": session.get("turn_index", 0),
        "workspace": str(workspace),
        "summary": trim_text(str(session.get("summary", "")), 1800),
        "last_route": session.get("last_route", ""),
        "last_task": trim_text(str(session.get("last_task", "")), 600),
        "last_final_answer": trim_text(str(session.get("last_final_answer", "")), 900),
        "recent_turns": recent_turns,
        "recent_files": files,
    }
    return trim_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), MAX_SESSION_CONTEXT)


def session_started_event(workspace: Path, session: dict[str, Any], *, resumed: bool = False) -> dict[str, Any]:
    return {
        "type": "session_started",
        "session_id": session.get("session_id", ""),
        "workspace": str(workspace),
        "turn_index": session.get("turn_index", 0),
        "resumed": resumed,
        "session_file": str(session_file(workspace)),
        "summary_file": str(session_summary_file(workspace)),
    }


def session_turn_started_event(workspace: Path, session: dict[str, Any], *, turn: int, task: str) -> dict[str, Any]:
    return {
        "type": "session_turn_started",
        "session_id": session.get("session_id", ""),
        "workspace": str(workspace),
        "turn": turn,
        "task": trim_text(task, 900),
    }


def session_turn_saved_event(workspace: Path, session: dict[str, Any], *, turn: int, route: str) -> dict[str, Any]:
    return {
        "type": "session_turn_saved",
        "session_id": session.get("session_id", ""),
        "workspace": str(workspace),
        "turn": turn,
        "route": route,
        "turn_count": len(session.get("recent_turns", [])),
        "summary_file": str(session_summary_file(workspace)),
    }


def build_session_summary_markdown(workspace: Path, session: dict[str, Any]) -> str:
    lines = [
        "# NexusAgent Session Summary",
        "",
        f"- session_id: {session.get('session_id', '')}",
        f"- workspace: {workspace}",
        f"- turns: {session.get('turn_index', 0)}",
        f"- updated_at: {session.get('updated_at', '')}",
        f"- last_route: {session.get('last_route', '') or '(none)'}",
        "",
        "## Summary",
        "",
        str(session.get("summary", "") or "(no compressed summary yet)"),
        "",
        "## Recent Turns",
        "",
    ]
    for turn in session.get("recent_turns", [])[-MAX_RECENT_TURNS:]:
        role = turn.get("role", "")
        route = f" / {turn.get('route')}" if turn.get("route") else ""
        lines.append(f"- turn {turn.get('turn')}: {role}{route}: {trim_text(str(turn.get('content', '')), 260)}")
    return "\n".join(lines).rstrip() + "\n"


def trim_text(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_session(raw: dict[str, Any], workspace: Path) -> dict[str, Any]:
    now = utc_now()
    session = dict(raw) if isinstance(raw, dict) else {}
    session.setdefault("version", 1)
    session.setdefault("session_id", f"session-{uuid4().hex[:8]}")
    session.setdefault("workspace", str(workspace))
    session.setdefault("created_at", now)
    session.setdefault("updated_at", now)
    session.setdefault("turn_index", 0)
    session.setdefault("summary", "")
    session.setdefault("recent_turns", [])
    session.setdefault("last_route", "")
    session.setdefault("last_task", "")
    session.setdefault("last_final_answer", "")
    if not isinstance(session.get("recent_turns"), list):
        session["recent_turns"] = []
    session["workspace"] = str(workspace)
    session["turn_index"] = int(session.get("turn_index") or 0)
    session["summary"] = trim_text(str(session.get("summary", "")), MAX_SESSION_SUMMARY)
    return session


def _append_turn(session: dict[str, Any], turn: dict[str, Any]) -> None:
    turns = list(session.get("recent_turns", []))
    turns.append(turn)
    session["recent_turns"] = turns


def _compact_session(session: dict[str, Any]) -> None:
    turns = list(session.get("recent_turns", []))
    if len(turns) <= MAX_RECENT_TURNS:
        return
    older = turns[: -MAX_RECENT_TURNS]
    kept = turns[-MAX_RECENT_TURNS:]
    additions = []
    for turn in older:
        role = turn.get("role", "")
        route = f"/{turn.get('route')}" if turn.get("route") else ""
        summary = turn.get("summary") or turn.get("content", "")
        additions.append(f"turn {turn.get('turn')}: {role}{route}: {trim_text(str(summary), 260)}")
    existing = str(session.get("summary", ""))
    session["summary"] = trim_text("\n".join(part for part in [existing, *additions] if part), MAX_SESSION_SUMMARY)
    session["recent_turns"] = kept
