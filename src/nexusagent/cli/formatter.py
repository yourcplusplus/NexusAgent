from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nexusagent.graph.status_bar import render_event_summary

console = Console()

STATUS_SYMBOLS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "blocked": "[!]",
}

STATUS_STYLES = {
    "pending": "dim",
    "in_progress": "bold yellow",
    "completed": "bold green",
    "blocked": "bold red",
}


def safe_echo(message: Any = "", **_: Any) -> None:
    console.print(str(message))


def safe_secho(message: Any = "", **kwargs: Any) -> None:
    color = kwargs.get("fg") or kwargs.get("style")
    console.print(str(message), style=color)


def _shorten(value: Any, limit: int = 260) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type == "workspace":
        console.print(Panel(str(event["path"]), title="Workspace", border_style="blue", box=box.ROUNDED))
        return
    if event_type == "custom_event":
        print_custom_event(event["event"])
        return
    if event_type == "graph_event":
        print_graph_event(event["event"])
        return
    console.print(_shorten(event))


def print_custom_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type == "intent_decision":
        render_intent_decision(event)
        return
    if event_type == "chat_response":
        render_chat_response(event)
        return
    if event_type in {"session_started", "session_turn_started", "session_turn_saved"}:
        render_session_event(event)
        return
    if event_type == "plan_snapshot":
        render_plan(event, title=f"Plan Snapshot · {event.get('node', 'graph')}")
        return
    if event_type == "todo_update":
        render_plan(event, title="Todo Updated", border_style="yellow")
        return
    if event_type == "tool_call":
        console.print(
            Panel(
                _format_args(event.get("args", {})),
                title=f"Tool Call · {event.get('node', 'agent')} · {event.get('name')}",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )
        return
    if event_type == "tool_result":
        result = event.get("result")
        style = "green"
        if isinstance(result, dict) and result.get("ok") is False:
            style = "red"
        console.print(
            Panel(
                _format_tool_result(result),
                title=f"Tool Result · {event.get('node', 'agent')} · {event.get('name')}",
                border_style=style,
                box=box.ROUNDED,
            )
        )
        return
    if event_type == "handoff":
        render_handoff(event)
        return
    if event_type == "handoff_result":
        render_handoff_result(event)
        return
    if event_type == "search_results":
        render_sources(event.get("sources", []), title=f"searchAgent · {event.get('query', '')}", answer=event.get("answer", ""))
        return
    if event_type == "search_summary":
        render_sources(event.get("sources", []), title="searchAgent Summary", answer=event.get("summary", ""))
        return
    if event_type == "memory_snapshot":
        render_memory_snapshot(event)
        return
    if event_type == "context_monitor":
        render_context_monitor(event)
        return
    if event_type == "context_compression":
        render_context_compression(event)
        return
    if event_type == "status_bar":
        render_status_bar(event)
        return
    if event_type == "checkpoint_saved":
        render_checkpoint_saved(event)
        return
    if event_type == "checkpoint_resumed":
        render_checkpoint_resumed(event)
        return
    if event_type == "trace_summary":
        render_trace_summary(event)
        return
    console.print(Panel(_shorten(event, 1000), title="Event", box=box.ROUNDED))


def render_intent_decision(event: dict[str, Any]) -> None:
    lines = [
        f"route: {event.get('route', '')}",
        f"confidence: {event.get('confidence', 0)}",
        f"reason: {_shorten(event.get('reason', ''), 600)}",
    ]
    style = "cyan" if event.get("route") == "chat" else "magenta"
    console.print(Panel("\n".join(lines), title="Intent Router", border_style=style, box=box.ROUNDED))


def render_chat_response(event: dict[str, Any]) -> None:
    lines = [_shorten(event.get("response", ""), 1600)]
    reason = event.get("reason")
    if reason:
        lines.append(f"\nmode: {event.get('mode', 'lightweight')} | reason: {reason}")
    console.print(Panel("\n".join(lines), title="NexusAgent", border_style="cyan", box=box.ROUNDED))


def render_session_event(event: dict[str, Any]) -> None:
    event_type = event.get("type", "")
    if event_type == "session_started":
        title = "Session Started"
        lines = [
            f"session: {event.get('session_id', '')}",
            f"workspace: {event.get('workspace', '')}",
            f"turns: {event.get('turn_index', 0)}",
            f"resumed: {event.get('resumed', False)}",
        ]
    elif event_type == "session_turn_started":
        title = "Session Turn"
        lines = [
            f"turn: {event.get('turn', 0)}",
            f"task: {_shorten(event.get('task', ''), 900)}",
        ]
    else:
        title = "Session Saved"
        lines = [
            f"turn: {event.get('turn', 0)}",
            f"route: {event.get('route', '')}",
            f"summary: {event.get('summary_file', '')}",
        ]
    console.print(Panel("\n".join(lines), title=title, border_style="cyan", box=box.ROUNDED))


def print_graph_event(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        console.print(_shorten(payload))
        return

    for node, update in payload.items():
        if not isinstance(update, dict):
            console.print(Panel(_shorten(update), title=str(node), box=box.ROUNDED))
            continue
        if node == "planner":
            render_plan(update, title="Planner", border_style="cyan")
        elif node in {"actor", "codeAgent"}:
            summary = update.get("code_agent_summary") or update.get("last_actor_summary")
            if summary:
                console.print(Panel(_shorten(summary, 1200), title="codeAgent Summary", border_style="cyan"))
        elif node == "verifier":
            render_verifier(update)
        elif node == "memory_snapshot":
            render_memory_snapshot(update)
        elif node == "context_monitor":
            render_context_monitor(update)
        elif node == "context_compressor":
            render_context_compression(update)
        elif node == "final":
            render_final(update)
        else:
            console.print(Panel(_shorten(update, 1200), title=str(node), box=box.ROUNDED))


def render_plan(update: dict[str, Any], *, title: str, border_style: str = "cyan") -> None:
    plan = update.get("plan_summary", "")
    todos = update.get("todos", [])
    commands = update.get("verification_commands", [])

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Todo")
    table.add_column("Note", style="dim")
    for todo in todos:
        status = todo.get("status", "pending")
        table.add_row(
            todo.get("id", ""),
            Text(STATUS_SYMBOLS.get(status, "[?]"), style=STATUS_STYLES.get(status, "")),
            todo.get("content", ""),
            todo.get("note", ""),
        )

    command_text = "\n".join(f"  - {command}" for command in commands)
    body = Table.grid(expand=True)
    if plan:
        body.add_row(Text(plan, style="bold"))
    if todos:
        body.add_row(table)
    if commands:
        body.add_row(Text("Verifier commands\n" + command_text, style="green"))
    console.print(Panel(body, title=title, border_style=border_style, box=box.ROUNDED))


def render_verifier(update: dict[str, Any]) -> None:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    checks = update.get("verification_checks") or []
    if checks:
        for check in checks:
            ok = bool(check.get("passed"))
            status = Text("PASS" if ok else "FAIL", style="bold green" if ok else "bold red")
            table.add_row(check.get("name", "check"), status, _shorten(check.get("detail", ""), 260))
    else:
        for result in update.get("verification_results", []):
            ok = bool(result.get("ok"))
            status = Text("PASS" if ok else "FAIL", style="bold green" if ok else "bold red")
            output = result.get("stdout") or result.get("stderr") or ""
            table.add_row(result.get("command", ""), status, _shorten(output, 260))
    footer = f"passed={update.get('passed')} | attempts={update.get('attempts')}"
    panel_grid = Table.grid(expand=True)
    summary = update.get("verifier_summary")
    if summary:
        panel_grid.add_row(Text(_shorten(summary, 600), style="bold"))
    panel_grid.add_row(table)
    panel_grid.add_row(Text(footer, style="yellow"))
    console.print(Panel(panel_grid, title="Verifier", border_style="green" if update.get("passed") else "red"))


def render_final(update: dict[str, Any]) -> None:
    answer = update.get("final_answer", "")
    style = "green" if "PASSED" in answer else "red"
    console.print(Panel(_shorten(answer, 2000), title="Final", border_style=style, box=box.ROUNDED))


def render_context_monitor(update: dict[str, Any]) -> None:
    should = bool(update.get("context_should_compress", update.get("should_compress")))
    token_count = update.get("context_token_count", update.get("token_count", 0))
    token_limit = update.get("context_token_limit", update.get("token_limit", 0))
    next_node = update.get("context_next_node", update.get("next_node", ""))
    message_count = update.get("message_count")
    lines = [
        f"tokens: {token_count} / {token_limit}",
        f"compress: {should}",
        f"next: {next_node}",
    ]
    if message_count is not None:
        lines.append(f"messages: {message_count}")
    console.print(
        Panel(
            "\n".join(lines),
            title="Context Monitor",
            border_style="yellow" if should else "blue",
            box=box.ROUNDED,
        )
    )


def render_context_compression(update: dict[str, Any]) -> None:
    events = update.get("compression_events")
    if events:
        event = events[-1]
    else:
        event = update
    lines = [
        f"tokens: {event.get('before_tokens')} -> {event.get('after_tokens')}",
        f"removed messages: {event.get('removed_messages')}",
        f"next: {event.get('next_node')}",
    ]
    summary = event.get("summary")
    if summary:
        lines.append("summary:\n" + _shorten(summary, 900))
    console.print(Panel("\n".join(lines), title="Context Compression", border_style="yellow", box=box.ROUNDED))


def render_status_bar(update: dict[str, Any]) -> None:
    console.print(
        Panel(
            render_event_summary(update),
            title="Environment Status",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def render_memory_snapshot(update: dict[str, Any]) -> None:
    layers = update.get("layers", {})
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Layer", no_wrap=True)
    table.add_column("Summary")
    for name in ("rules", "working_memory", "history_summary_store"):
        table.add_row(name, _shorten(layers.get(name, ""), 360))

    footer = (
        f"node={update.get('node', '')} | "
        f"rules={update.get('rules_count', 0)} | "
        f"todos={update.get('todo_count', 0)} | "
        f"sources={update.get('source_count', 0)} | "
        f"handoffs={update.get('handoff_count', 0)} | "
        f"notepad={update.get('notepad_exists')} | "
        f"history={update.get('history_exists')} {update.get('history_path', '')}"
    )
    body = Table.grid(expand=True)
    body.add_row(table)
    body.add_row(Text(footer, style="yellow"))
    console.print(Panel(body, title="Memory Snapshot", border_style="cyan", box=box.ROUNDED))


def render_checkpoint_saved(event: dict[str, Any]) -> None:
    lines = [
        f"mode: {event.get('mode', '')}",
        f"status: {event.get('status', '')}",
        f"path: {event.get('path', '')}",
        f"checkpoint: {event.get('checkpoint_file', '')}",
        f"recovery: {event.get('recovery_file', '')}",
        f"git: {event.get('git_commit') or '(not available)'}",
        f"resume: {event.get('resume_command', '')}",
    ]
    if event.get("git_error"):
        lines.append(f"git_error: {_shorten(event.get('git_error'), 180)}")
    style = "yellow" if event.get("status") == "interrupted" else "blue"
    console.print(Panel("\n".join(lines), title="Checkpoint Saved", border_style=style, box=box.ROUNDED))


def render_checkpoint_resumed(event: dict[str, Any]) -> None:
    lines = [
        f"mode: {event.get('mode', '')}",
        f"workspace: {event.get('workspace', '')}",
        f"source: {event.get('source', '')}",
        f"fallback: {event.get('fallback', False)}",
    ]
    if event.get("reason"):
        lines.append(f"reason: {_shorten(event.get('reason'), 300)}")
    console.print(Panel("\n".join(lines), title="Checkpoint Resumed", border_style="green", box=box.ROUNDED))


def render_trace_summary(event: dict[str, Any]) -> None:
    node_visits = event.get("node_visits") if isinstance(event.get("node_visits"), dict) else {}
    node_text = ", ".join(f"{node}:{count}" for node, count in node_visits.items()) or "(none)"
    lines = [
        f"trace: {event.get('trace_id', '')}",
        f"status: {event.get('status', '')}",
        f"duration_ms: {event.get('duration_ms', 0)}",
        f"path: {event.get('trace_dir', '')}",
        f"nodes: {node_text}",
        f"tools: {event.get('tool_calls', 0)} total / {event.get('failed_tool_calls', 0)} failed",
        f"approvals: {event.get('approval_count', 0)}",
        f"checkpoints: {event.get('checkpoint_count', 0)}",
        f"final: {event.get('final_status', '') or '(unknown)'}",
    ]
    errors = event.get("errors") if isinstance(event.get("errors"), list) else []
    if errors:
        lines.append("errors: " + _shorten(errors, 260))
    style = "green" if event.get("status") == "finished" else "yellow"
    console.print(Panel("\n".join(lines), title="Trace Summary", border_style=style, box=box.ROUNDED))


def _format_args(args: Any) -> str:
    return _shorten(args, 900)


def _format_tool_result(result: Any) -> str:
    if not isinstance(result, dict):
        return _shorten(result, 900)
    keys = [
        "ok",
        "type",
        "path",
        "exit_code",
        "timed_out",
        "duration_ms",
        "background",
        "pid",
        "requires_approval",
        "approved",
        "approval_id",
        "risk_reason",
        "error",
    ]
    lines = [f"{key}: {result[key]}" for key in keys if key in result]
    if "stdout" in result and result["stdout"]:
        lines.append("stdout:\n" + _shorten(result["stdout"], 500))
    if "stderr" in result and result["stderr"]:
        lines.append("stderr:\n" + _shorten(result["stderr"], 500))
    for path_key in ("stdout_path", "stderr_path"):
        if result.get(path_key):
            lines.append(f"{path_key}: {result[path_key]}")
    if "todos" in result:
        lines.append(f"todos: {len(result['todos'])} item(s)")
    if "heading" in result:
        lines.append(f"heading: {result['heading']}")
    if "content" in result and result["content"]:
        lines.append("content:\n" + _shorten(result["content"], 500))
    if "answer" in result and result["answer"]:
        lines.append("answer:\n" + _shorten(result["answer"], 500))
    if "results" in result:
        lines.append(f"sources: {len(result['results'])} item(s)")
    if "summary" in result and result["summary"]:
        lines.append("summary:\n" + _shorten(result["summary"], 500))
    if "sources" in result:
        lines.append(f"sources: {len(result['sources'])} item(s)")
    if not lines:
        lines.append(_shorten(result, 900))
    return "\n".join(lines)


def render_handoff(event: dict[str, Any]) -> None:
    title = f"Handoff · {event.get('from', 'agent')} -> {event.get('to', 'agent')}"
    console.print(Panel(_shorten(event.get("instruction", ""), 900), title=title, border_style="blue", box=box.ROUNDED))


def render_handoff_result(event: dict[str, Any]) -> None:
    title = f"Return · {event.get('from', 'agent')} -> {event.get('to', 'agent')}"
    console.print(Panel(_shorten(event.get("result", ""), 1200), title=title, border_style="cyan", box=box.ROUNDED))


def render_sources(sources: list[dict[str, Any]], *, title: str, answer: str = "") -> None:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Title")
    table.add_column("URL")
    table.add_column("Snippet")
    for source in sources[:6]:
        table.add_row(
            _shorten(source.get("title", ""), 48),
            _shorten(source.get("url", ""), 58),
            _shorten(source.get("content", ""), 120),
        )
    body = Table.grid(expand=True)
    if answer:
        body.add_row(Text(_shorten(answer, 900), style="bold"))
    if sources:
        body.add_row(table)
    console.print(Panel(body, title=title, border_style="blue", box=box.ROUNDED))
