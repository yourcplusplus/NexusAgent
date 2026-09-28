from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text
from typer.testing import CliRunner

from nexusagent.cli.app import app
from nexusagent.cli.event_summary import summarize_event
from nexusagent.cli.tui import NexusAgentTuiApp
from nexusagent.cli.tui.approval import ApprovalGate
from nexusagent.cli.tui.logo import render_logo
from nexusagent.core.approval import ApprovalRequest


def test_tui_help_is_available() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["tui", "--help"])

    assert result.exit_code == 0
    assert "Textual terminal interface" in result.output


def test_tui_options_are_accepted(monkeypatch) -> None:
    runner = CliRunner()
    captured = {}

    class FakeApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr("nexusagent.cli.tui.NexusAgentTuiApp", FakeApp)

    result = runner.invoke(
        app,
        ["tui", "--trace-mode", "off", "--checkpoint-mode", "strict", "--approval-mode", "deny", "--workspace", "demo-workspace"],
    )

    assert result.exit_code == 0
    assert captured["trace_mode"] == "off"
    assert captured["checkpoint_mode"] == "strict"
    assert captured["approval_mode"] == "deny"
    assert captured["workspace"] == Path("demo-workspace")


def test_natural_task_entry_still_works(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    calls = []

    def fake_stream(*args, **kwargs):
        calls.append((args, kwargs))
        yield {"type": "workspace", "path": str(tmp_path)}

    monkeypatch.setattr("nexusagent.cli.app.stream_agent_events", fake_stream)

    result = runner.invoke(app, ["demo task"])

    assert result.exit_code == 0
    assert calls[0][0][0] == "demo task"


def test_logo_renderer_returns_non_empty_text() -> None:
    logo = render_logo(max_width=20, max_rows=8)

    assert isinstance(logo, Text)
    assert str(logo).strip()
    assert len(str(logo).splitlines()) <= 8


def test_logo_renderer_falls_back_for_missing_asset(tmp_path) -> None:
    logo = render_logo(tmp_path / "missing.png", max_width=20, max_rows=8)

    assert str(logo).strip()


def test_tui_renders_fake_stream_events(tmp_path) -> None:
    def fake_stream(*args, **kwargs):
        yield {
            "type": "custom_event",
            "event": {"type": "session_started", "session_id": "session-demo", "workspace": str(tmp_path / "workspace-a")},
        }
        yield {"type": "workspace", "path": str(tmp_path / "workspace-a")}
        yield {
            "type": "custom_event",
            "event": {
                "type": "todo_update",
                "plan_summary": "demo plan",
                "todos": [{"id": "todo-1", "content": "write file", "status": "in_progress"}],
            },
        }
        yield {
            "type": "graph_event",
            "event": {"final": {"final_answer": "PASSED: wrote the file"}},
        }
        yield {
            "type": "custom_event",
            "event": {
                "type": "trace_summary",
                "trace_id": "trace-demo",
                "status": "finished",
                "trace_dir": str(tmp_path / "trace-demo"),
                "node_visits": {"final": 1},
                "tool_calls": 1,
                "failed_tool_calls": 0,
            },
        }

    async def run() -> None:
        app = NexusAgentTuiApp(initial_task="demo task", stream_factory=fake_stream)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause(0.3)
            assert "workspace-a" in app.sidebar_text
            assert "trace-demo" in app.sidebar_text
            assert "session-demo" in app.sidebar_text
            assert app.run_count == 1
            assert not app.running

    asyncio.run(run())


def test_tui_renders_lightweight_chat_response() -> None:
    def fake_stream(*args, **kwargs):
        yield {
            "type": "custom_event",
            "event": {
                "type": "chat_response",
                "mode": "lightweight",
                "reason": "greeting",
                "response": "你好，我在。",
            },
        }

    async def run() -> None:
        app = NexusAgentTuiApp(initial_task="你好", stream_factory=fake_stream)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.2)
            assert app.run_count == 1
            assert app.latest_workspace
            assert not app.running

    asyncio.run(run())


def test_tui_chat_response_keeps_long_body_visible() -> None:
    body = (
        "Cloudflare 是一家云服务平台。\n"
        "- CDN 与性能优化。\n"
        "- 安全能力：DDoS 防护、WAF、Bot 管理。\n"
        "- DNS 服务：托管权威 DNS、DNSSEC、智能解析和全局 Anycast 网络。"
    )

    async def run() -> None:
        app = NexusAgentTuiApp(stream_factory=lambda *args, **kwargs: [])
        async with app.run_test(size=(100, 30)) as pilot:
            app._handle_event(
                {
                    "type": "custom_event",
                    "event": {"type": "chat_response", "mode": "lightweight", "reason": "q&a", "response": body},
                }
            )
            await pilot.pause(0.1)
            summary = summarize_event(
                {"type": "custom_event", "event": {"type": "chat_response", "mode": "lightweight", "reason": "q&a", "response": body}}
            )
            assert "DNSSEC" in app._compact_body(summary)

    asyncio.run(run())

    summary = summarize_event({"type": "custom_event", "event": {"type": "chat_response", "response": body}})
    assert "DNSSEC" in summary.body


def test_tui_hides_low_level_entry_graph_updates() -> None:
    async def run() -> None:
        app = NexusAgentTuiApp(stream_factory=lambda *args, **kwargs: [])
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            before = len(app.query_one("#events").children)
            app._handle_event({"type": "graph_event", "event": {"intent_router": {"intent_route": "chat"}}})
            app._handle_event({"type": "graph_event", "event": {"chat_responder": {"chat_response": "hi"}}})
            app._handle_event({"type": "custom_event", "event": {"type": "session_turn_started", "turn": 1, "task": "hello"}})
            await pilot.pause(0.1)
            assert len(app.query_one("#events").children) == before

    asyncio.run(run())


def test_tui_input_bar_stays_visible() -> None:
    async def run() -> None:
        app = NexusAgentTuiApp(stream_factory=lambda *args, **kwargs: [])
        async with app.run_test(size=(100, 28)) as pilot:
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.pause(0.1)
            input_widget = app.query_one("#task-input")
            footer = app.query_one("Footer")
            assert input_widget.region.y < footer.region.y
            assert input_widget.value == "hello"

    asyncio.run(run())


def test_tui_user_message_card_keeps_compact_height() -> None:
    async def run() -> None:
        app = NexusAgentTuiApp(stream_factory=lambda *args, **kwargs: [])
        async with app.run_test(size=(100, 24)) as pilot:
            app._write_run_start("你好", None)
            await pilot.pause(0.1)
            user_card = app.query(".event-user").last()
            assert user_card.region.height <= 4

    asyncio.run(run())


def test_tui_runs_multiple_tasks_in_same_session_workspace(tmp_path) -> None:
    calls = []

    def fake_stream(*args, **kwargs):
        calls.append((args, kwargs))
        yield {"type": "workspace", "path": str(kwargs["session_workspace"])}

    async def run() -> None:
        app = NexusAgentTuiApp(workspace=tmp_path / "session-workspace", stream_factory=fake_stream)
        async with app.run_test(size=(100, 30)) as pilot:
            app.start_task("first")
            await pilot.pause(0.2)
            app.start_task("second")
            await pilot.pause(0.2)

    asyncio.run(run())

    assert len(calls) == 2
    assert calls[0][1]["session_workspace"] == tmp_path / "session-workspace"
    assert calls[1][1]["session_workspace"] == tmp_path / "session-workspace"
    assert calls[0][0][0] == "first"
    assert calls[1][0][0] == "second"


def test_tui_new_session_command_switches_workspace(tmp_path) -> None:
    async def run() -> None:
        app = NexusAgentTuiApp(workspace=tmp_path / "first", stream_factory=lambda *args, **kwargs: [])
        async with app.run_test(size=(100, 30)) as pilot:
            old_workspace = app.session_workspace
            app.start_new_session()
            await pilot.pause(0.1)
            assert app.session_workspace != old_workspace
            assert app.latest_workspace == str(app.session_workspace)

    asyncio.run(run())


def test_approval_gate_returns_decision() -> None:
    gate = ApprovalGate(ApprovalRequest(id="approval-demo", command="uv add fastapi", risk_reason="dependency change"))

    gate.resolve(True)

    assert gate.wait().approved is True
