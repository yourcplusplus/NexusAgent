from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nexusagent.graph.status_bar import render_event_summary


@dataclass(frozen=True)
class EventSummary:
    title: str
    body: str
    category: str = "event"
    style: str = "white"


def shorten(value: Any, limit: int = 260) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarize_event(event: dict[str, Any]) -> EventSummary:
    event_type = event.get("type")
    if event_type == "workspace":
        return EventSummary("Workspace", str(event.get("path", "")), "workspace", "blue")
    if event_type == "custom_event":
        payload = event.get("event")
        if isinstance(payload, dict):
            return summarize_custom_event(payload)
        return EventSummary("Custom Event", shorten(payload), "event", "white")
    if event_type == "graph_event":
        payload = event.get("event")
        if isinstance(payload, dict):
            return summarize_graph_event(payload)
        return EventSummary("Graph Event", shorten(payload), "event", "white")
    return EventSummary("Event", shorten(event), "event", "white")


def summarize_custom_event(event: dict[str, Any]) -> EventSummary:
    event_type = event.get("type", "event")
    if event_type == "intent_decision":
        route = str(event.get("route", "workflow"))
        return EventSummary(
            "Intent Router",
            (
                f"route: {route}\n"
                f"confidence: {event.get('confidence', 0)}\n"
                f"reason: {shorten(event.get('reason', ''), 600)}"
            ),
            "intent",
            "cyan" if route == "chat" else "magenta",
        )
    if event_type == "chat_response":
        return EventSummary(
            "NexusAgent",
            f"{shorten(event.get('response', ''), 2400)}\nmode: {event.get('mode', 'lightweight')} | reason: {event.get('reason', '')}",
            "chat",
            "cyan",
        )
    if event_type == "session_started":
        return EventSummary(
            "Session Started",
            (
                f"session: {event.get('session_id', '')}\n"
                f"workspace: {event.get('workspace', '')}\n"
                f"turns: {event.get('turn_index', 0)}\n"
                f"resumed: {event.get('resumed', False)}"
            ),
            "session",
            "cyan",
        )
    if event_type == "session_turn_started":
        return EventSummary(
            "Session Turn",
            f"turn: {event.get('turn', 0)}\n{shorten(event.get('task', ''), 900)}",
            "session",
            "magenta",
        )
    if event_type == "session_turn_saved":
        return EventSummary(
            "Session Saved",
            (
                f"turn: {event.get('turn', 0)}\n"
                f"route: {event.get('route', '')}\n"
                f"summary: {event.get('summary_file', '')}"
            ),
            "session",
            "green",
        )
    if event_type in {"plan_snapshot", "todo_update"}:
        return _summarize_plan(event, "Plan Snapshot" if event_type == "plan_snapshot" else "Todo Updated")
    if event_type == "tool_call":
        name = event.get("name", "tool")
        node = event.get("node", "agent")
        return EventSummary(f"{node} · {name}", shorten(event.get("args", {}), 500), "tool_call", "magenta")
    if event_type == "tool_result":
        name = event.get("name", "tool")
        node = event.get("node", "agent")
        result = event.get("result")
        ok = result.get("ok") if isinstance(result, dict) else None
        style = "green" if ok is not False else "red"
        return EventSummary(f"{node} · {name} result", _format_tool_result(result), "tool_result", style)
    if event_type == "handoff":
        return EventSummary(
            f"Handoff · {event.get('from', 'agent')} -> {event.get('to', 'agent')}",
            shorten(event.get("task", ""), 500),
            "handoff",
            "yellow",
        )
    if event_type == "handoff_result":
        return EventSummary(
            f"Handoff Result · {event.get('from', 'agent')}",
            shorten(event.get("summary", ""), 600),
            "handoff",
            "green",
        )
    if event_type in {"search_results", "search_summary"}:
        sources = event.get("sources") if isinstance(event.get("sources"), list) else []
        answer = event.get("answer") or event.get("summary") or ""
        return EventSummary(
            "Search Summary",
            f"{shorten(answer, 500)}\nsources: {len(sources)}",
            "search",
            "cyan",
        )
    if event_type == "memory_snapshot":
        return EventSummary("Memory Snapshot", _format_memory_snapshot(event), "memory", "cyan")
    if event_type == "context_monitor":
        return EventSummary(
            "Context Monitor",
            (
                f"tokens: {event.get('context_token_count', event.get('token_count', 0))} / "
                f"{event.get('context_token_limit', event.get('token_limit', 0))}\n"
                f"compress: {event.get('context_should_compress', event.get('should_compress', False))}\n"
                f"next: {event.get('context_next_node', event.get('next_node', ''))}"
            ),
            "context",
            "yellow" if event.get("context_should_compress", event.get("should_compress", False)) else "blue",
        )
    if event_type == "context_compression":
        compression = _latest_compression(event)
        return EventSummary(
            "Context Compression",
            (
                f"tokens: {compression.get('before_tokens')} -> {compression.get('after_tokens')}\n"
                f"removed messages: {compression.get('removed_messages')}\n"
                f"next: {compression.get('next_node')}\n"
                f"{shorten(compression.get('summary', ''), 500)}"
            ),
            "context",
            "yellow",
        )
    if event_type == "status_bar":
        return EventSummary("Environment Status", render_event_summary(event), "environment", "blue")
    if event_type == "checkpoint_saved":
        return EventSummary(
            "Checkpoint Saved",
            (
                f"mode: {event.get('mode', '')}\n"
                f"status: {event.get('status', '')}\n"
                f"path: {event.get('path', '')}\n"
                f"resume: {event.get('resume_command', '')}"
            ),
            "checkpoint",
            "yellow" if event.get("status") == "interrupted" else "blue",
        )
    if event_type == "checkpoint_resumed":
        return EventSummary(
            "Checkpoint Resumed",
            (
                f"mode: {event.get('mode', '')}\n"
                f"workspace: {event.get('workspace', '')}\n"
                f"source: {event.get('source', '')}\n"
                f"fallback: {event.get('fallback', False)}"
            ),
            "checkpoint",
            "green",
        )
    if event_type == "trace_summary":
        nodes = event.get("node_visits") if isinstance(event.get("node_visits"), dict) else {}
        node_text = ", ".join(f"{node}:{count}" for node, count in nodes.items()) or "(none)"
        return EventSummary(
            "Trace Summary",
            (
                f"trace: {event.get('trace_id', '')}\n"
                f"status: {event.get('status', '')}\n"
                f"path: {event.get('trace_dir', '')}\n"
                f"nodes: {node_text}\n"
                f"tools: {event.get('tool_calls', 0)} total / {event.get('failed_tool_calls', 0)} failed"
            ),
            "trace",
            "green" if event.get("status") == "finished" else "yellow",
        )
    return EventSummary(str(event_type), shorten(event, 1000), "event", "white")


def summarize_graph_event(payload: dict[str, Any]) -> EventSummary:
    if not payload:
        return EventSummary("Graph Event", "", "graph", "white")
    node = str(next(iter(payload)))
    update = payload.get(node)
    if not isinstance(update, dict):
        return EventSummary(node, shorten(update), "graph", "white")
    if node == "planner":
        return _summarize_plan(update, "Planner")
    if node in {"actor", "codeAgent"}:
        summary = update.get("code_agent_summary") or update.get("last_actor_summary") or update
        return EventSummary("codeAgent Summary", shorten(summary, 800), "agent", "cyan")
    if node == "verifier":
        return EventSummary("Verifier", _format_verifier(update), "verifier", "green" if update.get("passed") else "red")
    if node == "final":
        return EventSummary("Final", shorten(update.get("final_answer", update), 1200), "final", "green")
    if node == "context_monitor":
        return summarize_custom_event({"type": "context_monitor", **update})
    if node == "context_compressor":
        return summarize_custom_event({"type": "context_compression", **update})
    if node == "memory_snapshot":
        return summarize_custom_event({"type": "memory_snapshot", **update})
    return EventSummary(node, shorten(update, 800), "graph", "white")


def _summarize_plan(update: dict[str, Any], title: str) -> EventSummary:
    todos = update.get("todos") if isinstance(update.get("todos"), list) else []
    counts = _todo_counts(todos)
    lines = []
    if update.get("plan_summary"):
        lines.append(shorten(update.get("plan_summary"), 500))
    if todos:
        lines.append(f"todos: {counts}")
        for todo in todos[:6]:
            lines.append(f"- {todo.get('status', 'pending')}: {todo.get('content', todo.get('description', ''))}")
        if len(todos) > 6:
            lines.append(f"... {len(todos) - 6} more")
    commands = update.get("verification_commands") if isinstance(update.get("verification_commands"), list) else []
    if commands:
        lines.append("verify: " + "; ".join(str(command) for command in commands[:3]))
    return EventSummary(title, "\n".join(lines), "plan", "cyan")


def _format_tool_result(result: Any) -> str:
    if not isinstance(result, dict):
        return shorten(result, 700)
    keys = [
        "ok",
        "exit_code",
        "timed_out",
        "duration_ms",
        "background",
        "requires_approval",
        "approved",
        "approval_id",
        "risk_reason",
        "error",
        "path",
        "stdout_path",
        "stderr_path",
    ]
    lines = [f"{key}: {result[key]}" for key in keys if key in result]
    if result.get("stdout"):
        lines.append("stdout:\n" + shorten(result["stdout"], 500))
    if result.get("stderr"):
        lines.append("stderr:\n" + shorten(result["stderr"], 500))
    if "todos" in result:
        lines.append(f"todos: {len(result['todos'])} item(s)")
    return "\n".join(lines) or shorten(result, 700)


def _format_memory_snapshot(event: dict[str, Any]) -> str:
    layers = event.get("layers") if isinstance(event.get("layers"), dict) else {}
    return "\n".join(
        [
            f"rules: {shorten(layers.get('rules', ''), 180)}",
            f"working_memory: {shorten(layers.get('working_memory', ''), 220)}",
            f"history_summary_store: {shorten(layers.get('history_summary_store', ''), 220)}",
            (
                f"todos={event.get('todo_count', 0)} sources={event.get('source_count', 0)} "
                f"handoffs={event.get('handoff_count', 0)}"
            ),
        ]
    )


def _format_verifier(update: dict[str, Any]) -> str:
    lines = []
    if update.get("verifier_summary"):
        lines.append(shorten(update.get("verifier_summary"), 500))
    checks = update.get("verification_checks") if isinstance(update.get("verification_checks"), list) else []
    if checks:
        for check in checks[:6]:
            status = "PASS" if check.get("passed") else "FAIL"
            lines.append(f"{status}: {check.get('name', 'check')} - {shorten(check.get('detail', ''), 160)}")
    lines.append(f"passed={update.get('passed')} | attempts={update.get('attempts')}")
    return "\n".join(lines)


def _latest_compression(event: dict[str, Any]) -> dict[str, Any]:
    events = event.get("compression_events")
    if isinstance(events, list) and events and isinstance(events[-1], dict):
        return events[-1]
    return event


def _todo_counts(todos: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for todo in todos:
        status = str(todo.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{status}:{count}" for status, count in sorted(counts.items())) or "0"
