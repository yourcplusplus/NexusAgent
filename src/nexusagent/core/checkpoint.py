from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict


VALID_CHECKPOINT_MODES = {"light", "strict", "off"}
CHECKPOINT_ROOT = Path(".nexusagent") / "checkpoints"
CHECKPOINT_FILE = "checkpoint.json"
EVENTS_FILE = "events.jsonl"
STATE_FILE = "state.json"
RECOVERY_FILE = "RECOVERY.md"
GIT_DIR = "git"
MAX_RECOVERY_TEXT = 6000
MAX_MANIFEST_ITEMS = 160


def normalize_checkpoint_mode(mode: str | None) -> str:
    normalized = (mode or "light").strip().lower()
    return normalized if normalized in VALID_CHECKPOINT_MODES else "light"


def checkpoint_dir(workspace: Path) -> Path:
    return workspace / CHECKPOINT_ROOT


class CheckpointManager:
    def __init__(self, runtime: Any, task: str = "") -> None:
        self.runtime = runtime
        self.workspace = runtime.workspace
        self.mode = normalize_checkpoint_mode(getattr(runtime, "checkpoint_mode", "light"))
        self.task = task
        self.root = checkpoint_dir(self.workspace)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def save(
        self,
        state: dict[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        if event is not None and self.mode == "strict":
            self._append_event(event)

        if self.mode == "strict":
            _write_json(self.root / STATE_FILE, serialize_state(state))

        manifest = workspace_manifest(self.workspace)
        git_commit, git_error = snapshot_workspace_git(self.workspace, self.root)
        payload = self._payload(state, status=status, latest_node=latest_node, manifest=manifest, git_commit=git_commit, git_error=git_error)
        _write_json(self.root / CHECKPOINT_FILE, payload)
        (self.root / RECOVERY_FILE).write_text(build_recovery_markdown(payload), encoding="utf-8")

        return checkpoint_saved_event(payload)

    def _append_event(self, event: dict[str, Any]) -> None:
        line = {
            "timestamp": utc_now(),
            "event": json_safe(event),
        }
        with (self.root / EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def _payload(
        self,
        state: dict[str, Any],
        *,
        status: str,
        latest_node: str | None,
        manifest: list[dict[str, Any]],
        git_commit: str | None,
        git_error: str | None,
    ) -> dict[str, Any]:
        task = str(state.get("task") or self.task or "")
        summary = state_summary(state)
        return {
            "version": 1,
            "updated_at": utc_now(),
            "mode": self.mode,
            "status": status,
            "workspace": str(self.workspace),
            "checkpoint_dir": str(self.root),
            "checkpoint_file": str(self.root / CHECKPOINT_FILE),
            "recovery_file": str(self.root / RECOVERY_FILE),
            "state_file": str(self.root / STATE_FILE) if self.mode == "strict" else "",
            "events_file": str(self.root / EVENTS_FILE) if self.mode == "strict" else "",
            "task": task,
            "latest_node": latest_node or "",
            "next_node": state.get("context_next_node", ""),
            "attempts": state.get("attempts", 0),
            "max_attempts": state.get("max_attempts", 0),
            "summary": summary,
            "workspace_manifest": manifest,
            "git": {
                "dir": str(self.root / GIT_DIR),
                "commit": git_commit,
                "error": git_error,
            },
            "resume_command": resume_command(self.workspace),
        }


def checkpoint_saved_event(payload: dict[str, Any]) -> dict[str, Any]:
    git_info = payload.get("git") or {}
    return {
        "type": "checkpoint_saved",
        "mode": payload.get("mode", ""),
        "status": payload.get("status", ""),
        "workspace": payload.get("workspace", ""),
        "path": payload.get("checkpoint_dir", ""),
        "checkpoint_file": payload.get("checkpoint_file", ""),
        "recovery_file": payload.get("recovery_file", ""),
        "git_commit": git_info.get("commit"),
        "git_error": git_info.get("error"),
        "resume_command": payload.get("resume_command", ""),
    }


def checkpoint_resumed_event(
    *,
    workspace: Path,
    mode: str,
    source: str,
    fallback: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    root = checkpoint_dir(workspace)
    return {
        "type": "checkpoint_resumed",
        "mode": mode,
        "workspace": str(workspace),
        "path": str(root),
        "source": source,
        "fallback": fallback,
        "reason": reason,
    }


def load_resume_inputs(
    runtime: Any,
    *,
    task: str | None = None,
    max_attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_mode = normalize_checkpoint_mode(getattr(runtime, "checkpoint_mode", "light"))
    if requested_mode == "strict":
        try:
            state = load_strict_state(runtime, max_attempts=max_attempts)
        except Exception as exc:
            inputs = build_light_resume_inputs(runtime, task=task, max_attempts=max_attempts)
            event = checkpoint_resumed_event(
                workspace=runtime.workspace,
                mode="light",
                source=str(checkpoint_dir(runtime.workspace) / RECOVERY_FILE),
                fallback=True,
                reason=f"strict resume unavailable: {type(exc).__name__}: {exc}",
            )
            return inputs, event
        event = checkpoint_resumed_event(
            workspace=runtime.workspace,
            mode="strict",
            source=str(checkpoint_dir(runtime.workspace) / STATE_FILE),
        )
        return state, event

    inputs = build_light_resume_inputs(runtime, task=task, max_attempts=max_attempts)
    event = checkpoint_resumed_event(
        workspace=runtime.workspace,
        mode="light",
        source=str(checkpoint_dir(runtime.workspace) / RECOVERY_FILE),
    )
    return inputs, event


def load_strict_state(runtime: Any, *, max_attempts: int = 3) -> dict[str, Any]:
    state_path = checkpoint_dir(runtime.workspace) / STATE_FILE
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("state.json is not an object")
    state = deserialize_state(raw, runtime)
    state["max_attempts"] = max_attempts
    metadata = dict(state.get("metadata", {}))
    metadata.update(
        {
            "resumed": True,
            "resume_mode": "strict",
            "resume_workspace": str(runtime.workspace),
        }
    )
    state["metadata"] = metadata
    return state


def build_light_resume_inputs(runtime: Any, *, task: str | None = None, max_attempts: int = 3) -> dict[str, Any]:
    checkpoint = read_checkpoint(runtime.workspace)
    recovery = read_checkpoint_text(runtime.workspace, RECOVERY_FILE)
    todo = read_workspace_text(runtime.workspace, "TODO.md")
    notepad = read_workspace_text(runtime.workspace, "NOTEPAD.md")
    history = read_workspace_text(runtime.workspace, "HISTORY_SUMMARY.md")

    original_task = str(checkpoint.get("task") or "").strip()
    resume_task = normalize_resume_task(task.strip() if isinstance(task, str) and task.strip() else original_task)
    if not resume_task:
        resume_task = "Continue the interrupted NexusAgent task from the checkpoint."
    else:
        resume_task = f"Continue this NexusAgent task from the checkpoint: {resume_task}"

    context_parts = [
        "# Checkpoint Recovery Context",
        recovery,
        "## TODO.md",
        todo,
        "## NOTEPAD.md",
        notepad,
        "## HISTORY_SUMMARY.md",
        history,
    ]
    context_summary = trim_text("\n\n".join(part for part in context_parts if part), MAX_RECOVERY_TEXT)
    summary = checkpoint.get("summary") if isinstance(checkpoint.get("summary"), dict) else {}

    inputs: dict[str, Any] = {
        "task": resume_task,
        "runtime": runtime,
        "messages": [],
        "attempts": 0,
        "max_attempts": max_attempts,
        "context_summary": context_summary,
        "history_summary": trim_text(history, 2400),
        "metadata": {
            "resumed": True,
            "resume_mode": "light",
            "resume_workspace": str(runtime.workspace),
            "original_task": original_task,
        },
    }
    _copy_summary_fields(inputs, summary)
    return inputs


def read_checkpoint(workspace: Path) -> dict[str, Any]:
    path = checkpoint_dir(workspace) / CHECKPOINT_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def normalize_resume_task(task: str) -> str:
    prefixes = (
        "Continue this NexusAgent task from the checkpoint:",
        "Continue the interrupted NexusAgent task from the checkpoint:",
    )
    normalized = task.strip()
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                changed = True
    return normalized


def read_checkpoint_text(workspace: Path, name: str) -> str:
    path = checkpoint_dir(workspace) / name
    if not path.exists():
        return ""
    try:
        return trim_text(path.read_text(encoding="utf-8", errors="replace"), MAX_RECOVERY_TEXT)
    except OSError:
        return ""


def read_workspace_text(workspace: Path, name: str) -> str:
    path = workspace / name
    if not path.exists():
        return ""
    try:
        return trim_text(path.read_text(encoding="utf-8", errors="replace"), 2400)
    except OSError:
        return ""


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in state.items():
        if key == "runtime":
            continue
        if key == "messages" and isinstance(value, list):
            serialized[key] = [serialize_message(message) for message in value]
            continue
        serialized[key] = json_safe(value)
    return serialized


def deserialize_state(data: dict[str, Any], runtime: Any) -> dict[str, Any]:
    state = dict(data)
    messages = state.get("messages")
    if isinstance(messages, list):
        state["messages"] = deserialize_messages(messages)
    else:
        state["messages"] = []
    state["runtime"] = runtime
    return state


def serialize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, BaseMessage):
        return message_to_dict(message)
    return json_safe(message)


def deserialize_messages(messages: list[Any]) -> list[BaseMessage]:
    typed_messages = [message for message in messages if isinstance(message, dict) and "type" in message and "data" in message]
    if len(typed_messages) != len(messages):
        return []
    return list(messages_from_dict(typed_messages))


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    todos = state.get("todos", [])
    sources = state.get("sources", [])
    return {
        "plan_summary": trim_text(state.get("plan_summary", ""), 1200),
        "todos": json_safe(todos),
        "todo_count": len(todos) if isinstance(todos, list) else 0,
        "acceptance_criteria": json_safe(state.get("acceptance_criteria", [])),
        "verification_commands": json_safe(state.get("verification_commands", [])),
        "passed": state.get("passed"),
        "attempts": state.get("attempts", 0),
        "sources": json_safe(sources[:10] if isinstance(sources, list) else []),
        "source_count": len(sources) if isinstance(sources, list) else 0,
        "research_notes": trim_text(state.get("research_notes", ""), 1600),
        "code_agent_summary": trim_text(state.get("code_agent_summary", "") or state.get("last_actor_summary", ""), 1600),
        "verifier_summary": trim_text(state.get("verifier_summary", ""), 1600),
        "last_error": trim_text(state.get("last_error", ""), 1600),
        "context_summary": trim_text(state.get("context_summary", ""), 1800),
        "history_summary": trim_text(state.get("history_summary", ""), 1800),
        "final_answer": trim_text(state.get("final_answer", ""), 1800),
    }


def build_recovery_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    manifest = payload.get("workspace_manifest") if isinstance(payload.get("workspace_manifest"), list) else []
    todos = summary.get("todos") if isinstance(summary.get("todos"), list) else []
    sources = summary.get("sources") if isinstance(summary.get("sources"), list) else []
    commands = summary.get("verification_commands") if isinstance(summary.get("verification_commands"), list) else []
    criteria = summary.get("acceptance_criteria") if isinstance(summary.get("acceptance_criteria"), list) else []

    lines = [
        "# NexusAgent Recovery",
        "",
        f"- status: {payload.get('status', '')}",
        f"- mode: {payload.get('mode', '')}",
        f"- updated_at: {payload.get('updated_at', '')}",
        f"- latest_node: {payload.get('latest_node', '')}",
        f"- next_node: {payload.get('next_node', '')}",
        f"- attempts: {payload.get('attempts', 0)} / {payload.get('max_attempts', 0)}",
        f"- workspace: {payload.get('workspace', '')}",
        f"- resume: `{payload.get('resume_command', '')}`",
        "",
        "## Task",
        trim_text(payload.get("task", ""), 1200) or "(unknown)",
        "",
        "## Plan",
        summary.get("plan_summary") or "(none)",
        "",
        "## Todos",
    ]
    lines.extend(_markdown_items([f"[{todo.get('status', '')}] {todo.get('content', '')}" for todo in todos]))
    lines.extend(["", "## Acceptance Criteria"])
    lines.extend(_markdown_items(criteria))
    lines.extend(["", "## Verification Commands"])
    lines.extend(_markdown_items(commands))
    lines.extend(["", "## Sources"])
    lines.extend(_markdown_items([f"{source.get('title', '')}: {source.get('url', '')}" for source in sources]))
    lines.extend(
        [
            "",
            "## Recent Summaries",
            f"- research: {summary.get('research_notes') or '(none)'}",
            f"- codeAgent: {summary.get('code_agent_summary') or '(none)'}",
            f"- verifier: {summary.get('verifier_summary') or '(none)'}",
            f"- last_error: {summary.get('last_error') or '(none)'}",
            "",
            "## Recent Files",
        ]
    )
    lines.extend(_markdown_items([f"{item.get('path', '')} ({item.get('size', 0)} bytes)" for item in manifest[:40]]))
    return "\n".join(lines).rstrip() + "\n"


def workspace_manifest(workspace: Path, *, limit: int = MAX_MANIFEST_ITEMS) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not workspace.exists():
        return items
    for path in sorted(workspace.rglob("*")):
        if len(items) >= limit:
            break
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        if should_skip_workspace_path(rel):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            {
                "path": rel.as_posix(),
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return items


def snapshot_workspace_git(workspace: Path, root: Path) -> tuple[str | None, str | None]:
    if shutil.which("git") is None:
        return None, "git executable not found"

    workspace = workspace.resolve()
    root = root.resolve()
    git_dir = root / GIT_DIR
    git_dir.mkdir(parents=True, exist_ok=True)
    try:
        _git(workspace, git_dir, ["init", "-q"])
        _git(workspace, git_dir, ["config", "user.name", "NexusAgent Checkpoint"])
        _git(workspace, git_dir, ["config", "user.email", "nexusagent-checkpoint@example.local"])
        _ensure_git_excludes(git_dir)
        _git(workspace, git_dir, ["add", "-A", "--", "."])
        status = _git(workspace, git_dir, ["status", "--porcelain"]).stdout.strip()
        head = git_head(workspace, git_dir)
        if not status and head:
            return head, None
        args = ["commit", "-q", "-m", f"checkpoint {utc_now()}"]
        if not status:
            args.append("--allow-empty")
        _git(workspace, git_dir, args)
        return git_head(workspace, git_dir), None
    except Exception as exc:
        return git_head(workspace, git_dir), f"{type(exc).__name__}: {exc}"


def git_head(workspace: Path, git_dir: Path) -> str | None:
    try:
        result = _git(workspace, git_dir, ["rev-parse", "--short", "HEAD"])
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def resume_command(workspace: Path) -> str:
    return f"uv run nexusagent --resume {shlex.quote(str(workspace))}"


def json_safe(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def trim_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(json_safe(value), ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def should_skip_workspace_path(rel: Path) -> bool:
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == ".nexusagent" and parts[1] == "checkpoints":
        return True
    skip_names = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    return any(part in skip_names for part in parts)


def _copy_summary_fields(inputs: dict[str, Any], summary: dict[str, Any]) -> None:
    for key in (
        "plan_summary",
        "todos",
        "acceptance_criteria",
        "verification_commands",
        "sources",
        "research_notes",
        "code_agent_summary",
        "verifier_summary",
        "last_error",
    ):
        if key in summary and summary[key]:
            inputs[key] = summary[key]


def _markdown_items(items: list[Any]) -> list[str]:
    return [f"- {item}" for item in items if item] or ["- (none)"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _git(workspace: Path, git_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={workspace}", *args],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    )


def _ensure_git_excludes(git_dir: Path) -> None:
    exclude_path = git_dir / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    patterns = [
        ".nexusagent/checkpoints/",
        ".venv/",
        "venv/",
        "node_modules/",
        "__pycache__/",
        ".pytest_cache/",
    ]
    with exclude_path.open("a", encoding="utf-8") as handle:
        for pattern in patterns:
            if pattern not in existing:
                handle.write(pattern + "\n")
