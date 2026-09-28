from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Literal

from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, Input, Static

from nexusagent.cli.event_summary import EventSummary, shorten, summarize_event
from nexusagent.cli.tui.approval import ApprovalGate, ApprovalModal
from nexusagent.cli.tui.logo import render_logo
from nexusagent.core.approval import ApprovalDecision, ApprovalRequest
from nexusagent.core.agent import stream_session_events
from nexusagent.core.paths import default_workspace


StreamFactory = Callable[..., Iterable[dict[str, Any]]]


class AgentEventMessage(Message):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


class RunFinishedMessage(Message):
    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


class ApprovalRequestedMessage(Message):
    def __init__(self, gate: ApprovalGate) -> None:
        super().__init__()
        self.gate = gate


class NexusAgentTuiApp(App[None]):
    CSS = """
    Screen {
        background: #101113;
        color: #d7d1c9;
    }

    #root {
        height: 1fr;
    }

    #top {
        height: 8;
        border-bottom: solid #2f3437;
        padding: 0 2;
        background: #151719;
    }

    #logo {
        width: 32;
        height: 7;
        content-align: center middle;
    }

    #title-block {
        width: 1fr;
        height: 7;
        padding-left: 2;
        content-align: left middle;
    }

    #title {
        text-style: bold;
        color: #f3ede3;
    }

    #status {
        color: #9aa4a6;
    }

    #subtitle {
        color: #7fd6c2;
    }

    #body {
        height: 1fr;
    }

    #events {
        width: 1fr;
        height: 100%;
        border-right: solid #2f3437;
        padding: 1 1;
        background: #101113;
    }

    #sidebar {
        width: 36;
        min-width: 30;
        height: 100%;
        padding: 1 1;
        background: #151719;
    }

    #side-title {
        text-style: bold;
        color: #f4bf75;
        margin-bottom: 1;
    }

    #side-state {
        color: #d7d1c9;
    }

    #input-row {
        height: 3;
        border: round #4a8f86;
        padding: 0 1;
        background: #151719;
    }

    #prompt {
        width: 3;
        height: 1;
        content-align: center middle;
        color: #7fd6c2;
        text-style: bold;
    }

    #task-input {
        width: 1fr;
        height: 1;
        border: none;
        background: #151719;
        color: #f3ede3;
    }

    #hint {
        color: #8a9294;
        width: 32;
        height: 1;
        padding-left: 1;
        content-align: right middle;
    }

    .event-card {
        height: auto;
        min-height: 1;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: solid #3f474b;
    }

    .event-summary {
        height: auto;
        min-height: 1;
    }

    .event-running {
        border-left: solid #f4bf75;
    }

    .event-success {
        border-left: solid #7fd68a;
    }

    .event-error {
        border-left: solid #ef6f6c;
    }

    .event-info {
        border-left: solid #7fd6c2;
    }

    .event-user {
        border-left: solid #f4bf75;
        background: #222426;
    }

    .detail {
        height: auto;
        max-height: 12;
        color: #b7b0a8;
        padding: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "cancel_or_quit", "Cancel/Quit"),
        ("ctrl+l", "clear_events", "Clear"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        initial_task: str | None = None,
        workspace: Path | None = None,
        max_attempts: int = 3,
        approval_mode: Literal["inline", "auto", "deny"] = "inline",
        checkpoint_mode: Literal["light", "strict", "off"] = "light",
        trace_mode: Literal["on", "off"] = "on",
        resume: Path | None = None,
        stream_factory: StreamFactory = stream_session_events,
    ) -> None:
        super().__init__()
        self.initial_task = initial_task
        self.workspace = resume or workspace or default_workspace()
        self.session_workspace = self.workspace
        self.max_attempts = max_attempts
        self.approval_mode = approval_mode
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self.resume = resume
        self.stream_factory = stream_factory
        self.running = False
        self.run_count = 0
        self.approval_count = 0
        self.failed_tool_count = 0
        self.tool_count = 0
        self.latest_workspace = str(self.session_workspace)
        self.latest_checkpoint = ""
        self.latest_trace = ""
        self.session_id = ""
        self.session_turn = 0
        self.last_route = ""
        self.sidebar_text = ""
        self.todos: list[dict[str, Any]] = []
        self._state_lock = Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            with Horizontal(id="top"):
                yield Static(render_logo(max_width=28, max_rows=7), id="logo")
                with Vertical(id="title-block"):
                    yield Static("NexusAgent", id="title")
                    yield Static("ready", id="status")
                    yield Static("coding session with context + harness", id="subtitle")
            with Horizontal(id="body"):
                yield VerticalScroll(id="events")
                with Vertical(id="sidebar"):
                    yield Static("Session", id="side-title")
                    yield Static("", id="side-state")
            with Horizontal(id="input-row"):
                yield Static("❯", id="prompt")
                yield Input(placeholder="Chat or ask for coding work, then press Enter", id="task-input")
                yield Static("Enter send · /new session · Ctrl+L clear", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self._write_welcome()
        self._refresh_sidebar()
        self.query_one("#task-input", Input).focus()
        if self.initial_task:
            self.call_after_refresh(self.start_task, self.initial_task, self.resume)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "task-input":
            return
        task = event.value.strip()
        if not task or self.running:
            return
        event.input.value = ""
        if task == "/new":
            self.start_new_session()
            return
        self.start_task(task, None)

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        self._handle_event(message.event)

    def on_run_finished_message(self, message: RunFinishedMessage) -> None:
        self.running = False
        self.resume = None
        self.query_one("#task-input", Input).disabled = False
        self.query_one("#task-input", Input).focus()
        self.query_one("#status", Static).update(f"{message.status}; ready for next task")
        self._refresh_sidebar()

    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        workspace = self.latest_workspace or str(self.workspace or "")
        self.push_screen(ApprovalModal(message.gate.request, workspace), self._resolve_approval(message.gate))

    def action_cancel_or_quit(self) -> None:
        if self.running:
            self.notify("A run is active. Press Ctrl+Q to quit and let checkpoint handle recovery.", severity="warning")
            return
        self.exit()

    def action_clear_events(self) -> None:
        self.query_one("#events", VerticalScroll).remove_children()
        self._write_welcome()

    def start_task(self, task: str, resume: Path | None = None) -> None:
        if self.running:
            self.notify("NexusAgent is already running a task.", severity="warning")
            return
        self.running = True
        self.run_count += 1
        self.todos = []
        self.failed_tool_count = 0
        self.tool_count = 0
        self.query_one("#task-input", Input).disabled = True
        self.query_one("#status", Static).update("running")
        self._refresh_sidebar()
        self._write_run_start(task, resume)
        self.run_worker(lambda: self._run_stream(task, resume), thread=True, exclusive=False, name=f"nexusagent-run-{self.run_count}")

    def _run_stream(self, task: str, resume: Path | None) -> None:
        status = "finished"
        try:
            approval_handler = self._approval_handler if self.approval_mode == "inline" else None
            for event in self.stream_factory(
                task,
                session_workspace=self.session_workspace,
                max_attempts=self.max_attempts,
                approval_mode=self.approval_mode,
                approval_handler=approval_handler,
                checkpoint_mode=self.checkpoint_mode,
                resume_workspace=resume,
                trace_mode=self.trace_mode,
            ):
                self.call_from_thread(self.post_message, AgentEventMessage(event))
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "failed"
            error_event = {"type": "custom_event", "event": {"type": "tui_error", "error": f"{type(exc).__name__}: {exc}"}}
            self.call_from_thread(self.post_message, AgentEventMessage(error_event))
        finally:
            self.call_from_thread(self.post_message, RunFinishedMessage(status))

    def _approval_handler(self, request: ApprovalRequest) -> ApprovalDecision:
        gate = ApprovalGate(request)
        self.call_from_thread(self.post_message, ApprovalRequestedMessage(gate))
        return gate.wait()

    def _resolve_approval(self, gate: ApprovalGate) -> Callable[[bool | None], None]:
        def resolve(result: bool | None) -> None:
            approved = bool(result)
            gate.resolve(approved)
            self._refresh_sidebar()

        return resolve

    def _handle_event(self, event: dict[str, Any]) -> None:
        self._update_state_from_event(event)
        if self._should_hide_event(event):
            self._refresh_sidebar()
            return
        summary = summarize_event(event)
        self._write_summary(summary)
        self._refresh_sidebar()

    def _update_state_from_event(self, event: dict[str, Any]) -> None:
        with self._state_lock:
            if event.get("type") == "workspace":
                self.latest_workspace = str(event.get("path", ""))
                self.session_workspace = Path(self.latest_workspace)
                return
            payload = event.get("event")
            if event.get("type") == "graph_event" and isinstance(payload, dict):
                for update in payload.values():
                    if isinstance(update, dict):
                        self._update_from_payload(update)
            elif event.get("type") == "custom_event" and isinstance(payload, dict):
                self._update_from_payload(payload)

    def _update_from_payload(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("todos"), list):
            self.todos = payload["todos"]
        if payload.get("type") == "tool_call":
            self.tool_count += 1
        if payload.get("type") == "tool_result":
            result = payload.get("result")
            if isinstance(result, dict):
                if result.get("ok") is False:
                    self.failed_tool_count += 1
                if result.get("requires_approval"):
                    self.approval_count += 1
        if payload.get("type") == "checkpoint_saved":
            self.latest_checkpoint = str(payload.get("path", ""))
        if payload.get("type") == "trace_summary":
            self.latest_trace = str(payload.get("trace_dir", ""))
        if payload.get("type") == "session_started":
            self.session_id = str(payload.get("session_id", ""))
            self.session_turn = int(payload.get("turn_index", 0) or 0)
            self.latest_workspace = str(payload.get("workspace", self.latest_workspace))
        if payload.get("type") == "session_turn_started":
            self.session_turn = int(payload.get("turn", self.session_turn) or self.session_turn)
        if payload.get("type") == "session_turn_saved":
            self.session_turn = int(payload.get("turn", self.session_turn) or self.session_turn)
            self.last_route = str(payload.get("route", self.last_route))

    def _write_welcome(self) -> None:
        self._mount_event_card(
            "NexusAgent",
            "Ask for a quick answer or coding work. Use /new to open a fresh workspace.",
            category="info",
            collapsed=True,
            detail="Persistent TUI sessions keep one workspace across turns. Workflow turns still use approval, checkpoint, trace, and layered memory.",
        )

    def _write_run_start(self, task: str, resume: Path | None) -> None:
        mode = f"resume: {resume}" if resume is not None else f"workspace: {self.session_workspace}"
        self._mount_event_card(
            f"You · turn {self.run_count}",
            shorten(task, 500),
            category="user",
            collapsed=False,
            detail=mode,
        )

    def _write_summary(self, summary: EventSummary) -> None:
        self._mount_event_card(
            summary.title,
            self._compact_body(summary),
            category=self._event_category(summary),
            collapsed=self._should_collapse(summary),
            detail=summary.body,
        )

    def _refresh_sidebar(self) -> None:
        status = "running" if self.running else "ready"
        workspace = shorten(self.latest_workspace or str(self.session_workspace), 80)
        checkpoint = shorten(self.latest_checkpoint or "(waiting)", 80)
        trace = shorten(self.latest_trace or "(waiting)", 80)
        tools = f"{self.tool_count} total / {self.failed_tool_count} failed"
        approvals = str(self.approval_count)
        todos = self._todo_sidebar_text()
        self.sidebar_text = "\n".join(
            [
                f"status {status}",
                f"turns {self.run_count}",
                f"session {self.session_id}",
                f"route {self.last_route or '(none)'}",
                f"workspace {workspace}",
                f"checkpoint {checkpoint}",
                f"trace {trace}",
                f"tools {tools}",
                f"approvals {approvals}",
                f"todos {todos}",
            ]
        )
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("status", status)
        table.add_row("turns", str(self.run_count))
        table.add_row("session", shorten(self.session_id or "(starting)", 24))
        table.add_row("route", self.last_route or "(none)")
        table.add_row("workspace", workspace)
        table.add_row("checkpoint", checkpoint)
        table.add_row("trace", trace)
        table.add_row("tools", tools)
        table.add_row("approvals", approvals)
        table.add_row("todos", todos)
        self.query_one("#side-state", Static).update(table)

    def _todo_sidebar_text(self) -> str:
        if not self.todos:
            return "(none yet)"
        counts: dict[str, int] = {}
        for todo in self.todos:
            status = str(todo.get("status", "pending"))
            counts[status] = counts.get(status, 0) + 1
        current = next((todo for todo in self.todos if todo.get("status") == "in_progress"), None)
        count_text = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        if current:
            return f"{count_text}\n{shorten(current.get('content', current.get('description', '')), 120)}"
        return count_text

    def _mount_event_card(
        self,
        title: str,
        body: str,
        *,
        category: str = "info",
        collapsed: bool = True,
        detail: str | None = None,
    ) -> None:
        events = self.query_one("#events", VerticalScroll)
        title_text = f"{self._category_marker(category)} {title}"
        summary = Static(Text(body or " ", style=self._category_style(category)), classes="event-summary")
        detail_text = detail if detail is not None else body
        if collapsed:
            card = Collapsible(
                summary,
                Static(self._detail_renderable(detail_text), classes="detail"),
                title=title_text,
                collapsed=True,
                classes=f"event-card {self._category_class(category)}",
            )
        else:
            card = Vertical(
                Static(Text(title_text, style=f"bold {self._category_style(category)}")),
                summary,
                classes=f"event-card {self._category_class(category)}",
            )
            card.styles.height = "auto"
        events.mount(card)
        events.scroll_end(animate=False)

    def _compact_body(self, summary: EventSummary) -> str:
        title = summary.title
        body = summary.body or ""
        if summary.category == "session":
            return self._first_matching_line(body, ("route:", "turn:", "workspace:", "session:")) or shorten(body, 140)
        if summary.category == "intent":
            route = self._line_value(body, "route")
            reason = self._line_value(body, "reason")
            return f"route {route or 'workflow'}" + (f" · {shorten(reason, 90)}" if reason else "")
        if summary.category == "chat":
            return shorten(body.split("\nmode:")[0], 2400)
        if summary.category == "plan":
            todos = self._line_value(body, "todos")
            first = body.splitlines()[0] if body.splitlines() else title
            return shorten(first + (f" · todos {todos}" if todos else ""), 180)
        if summary.category == "tool_call":
            return shorten(body, 160)
        if summary.category == "tool_result":
            ok = self._line_value(body, "ok")
            path = self._line_value(body, "path") or self._line_value(body, "stdout_path")
            pieces = [f"ok={ok}" if ok else "tool result"]
            if path:
                pieces.append(path)
            return shorten(" · ".join(pieces), 180)
        if summary.category == "handoff":
            return shorten(body, 180)
        if summary.category == "memory":
            return "memory snapshot updated"
        if summary.category == "context":
            return self._first_matching_line(body, ("tokens:", "compress:", "next:")) or shorten(body, 160)
        if summary.category == "environment":
            return shorten(self._first_matching_line(body, ("[Git]", "not a git repo", "context:")) or body, 180)
        if summary.category == "checkpoint":
            status = self._line_value(body, "status") or self._line_value(body, "mode")
            return f"checkpoint {status}" if status else "checkpoint updated"
        if summary.category == "trace":
            status = self._line_value(body, "status")
            tools = self._line_value(body, "tools")
            return " · ".join(part for part in [f"status {status}" if status else "", f"tools {tools}" if tools else ""] if part)
        if summary.category == "final":
            return shorten(body.splitlines()[0] if body.splitlines() else body, 220)
        if summary.category == "verifier":
            return shorten(body.splitlines()[0] if body.splitlines() else body, 180)
        return shorten(body, 180)

    def _should_collapse(self, summary: EventSummary) -> bool:
        return summary.category not in {"chat", "final", "verifier"}

    def _event_category(self, summary: EventSummary) -> str:
        if summary.category in {"final", "trace"}:
            return "success"
        if summary.category in {"verifier", "tool_result"} and "FAIL" in summary.body:
            return "error"
        if summary.category in {"plan", "tool_call", "handoff", "context", "checkpoint"}:
            return "running"
        return "info"

    def _should_hide_event(self, event: dict[str, Any]) -> bool:
        if event.get("type") == "workspace":
            return True
        payload = event.get("event")
        if event.get("type") == "graph_event" and isinstance(payload, dict):
            hidden_nodes = {"intent_router", "chat_responder"}
            return all(node in hidden_nodes for node in payload)
        if event.get("type") == "custom_event" and isinstance(payload, dict):
            return payload.get("type") in {"session_started", "session_turn_started", "memory_snapshot"}
        return False

    def _detail_renderable(self, detail: str) -> Any:
        text = detail or "(no details)"
        if len(text) > 1600:
            text = text[:1597] + "..."
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return Text(text)
        return Pretty(parsed, max_depth=4)

    def _category_marker(self, category: str) -> str:
        return {
            "running": "•",
            "success": "✓",
            "error": "!",
            "info": "·",
            "user": ">",
        }.get(category, "·")

    def _category_style(self, category: str) -> str:
        return {
            "running": "#f4bf75",
            "success": "#7fd68a",
            "error": "#ef6f6c",
            "info": "#7fd6c2",
            "user": "#f3ede3",
        }.get(category, "#d7d1c9")

    def _category_class(self, category: str) -> str:
        return {
            "running": "event-running",
            "success": "event-success",
            "error": "event-error",
            "info": "event-info",
            "user": "event-user",
        }.get(category, "event-info")

    def _line_value(self, body: str, key: str) -> str:
        prefix = f"{key}:"
        for line in body.splitlines():
            if line.strip().startswith(prefix):
                return line.split(":", 1)[1].strip()
        return ""

    def _first_matching_line(self, body: str, prefixes: tuple[str, ...]) -> str:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefixes):
                return stripped
        return ""

    def start_new_session(self) -> None:
        if self.running:
            self.notify("NexusAgent is already running a task.", severity="warning")
            return
        self.workspace = default_workspace()
        self.session_workspace = self.workspace
        self.resume = None
        self.latest_workspace = str(self.session_workspace)
        self.latest_checkpoint = ""
        self.latest_trace = ""
        self.session_id = ""
        self.session_turn = 0
        self.last_route = ""
        self.todos = []
        self.failed_tool_count = 0
        self.tool_count = 0
        self.approval_count = 0
        self._refresh_sidebar()
        self._mount_event_card(
            "New Session",
            str(self.session_workspace),
            category="info",
            collapsed=False,
        )
