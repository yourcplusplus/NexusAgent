from __future__ import annotations

from pathlib import Path

from nexusagent.core.agent import create_runtime
from nexusagent.core.paths import new_task_workspace


def test_new_task_workspace_is_unique(tmp_path: Path) -> None:
    first = new_task_workspace(tmp_path)
    second = new_task_workspace(tmp_path)

    assert first != second
    assert first.parent == tmp_path / ".nexusagent" / "workspaces"
    assert first.name.startswith("workspace-")


def test_create_runtime_uses_fresh_default_workspace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nexusagent.core.paths.find_project_root", lambda start=None: tmp_path)

    first = create_runtime()
    second = create_runtime()

    assert first.workspace != second.workspace
    assert first.workspace.exists()
    assert second.workspace.exists()


def test_create_runtime_respects_explicit_workspace(tmp_path: Path) -> None:
    explicit = tmp_path / "my-workspace"

    runtime = create_runtime(explicit)

    assert runtime.workspace == explicit
    assert explicit.exists()


def test_create_runtime_sets_approval_configuration(tmp_path: Path) -> None:
    handler = lambda request: True

    runtime = create_runtime(tmp_path / "workspace", approval_mode="deny", approval_handler=handler)

    assert runtime.approval_mode == "deny"
    assert runtime.approval_handler is handler


def test_create_runtime_reads_bash_harness_env(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / "agent.env"
    monkeypatch.setenv("NEXUS_BASH_DEFAULT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("NEXUS_BASH_MAX_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("NEXUS_BASH_MAX_OUTPUT_CHARS", "1234")
    monkeypatch.setenv("NEXUS_BASH_ENV_FILE", str(env_file))

    runtime = create_runtime(tmp_path / "workspace")

    assert runtime.bash_default_timeout_seconds == 45
    assert runtime.bash_max_timeout_seconds == 300
    assert runtime.bash_max_output_chars == 1234
    assert runtime.bash_env_file == env_file


def test_create_runtime_sets_checkpoint_configuration(tmp_path: Path) -> None:
    resume = tmp_path / "workspace"

    runtime = create_runtime(resume, checkpoint_mode="strict", resume_from=resume)

    assert runtime.checkpoint_mode == "strict"
    assert runtime.resume_from == resume


def test_create_runtime_reads_checkpoint_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEXUS_CHECKPOINT_MODE", "deny-this-invalid-mode")

    runtime = create_runtime(tmp_path / "workspace", checkpoint_mode=None)

    assert runtime.checkpoint_mode == "light"


def test_create_runtime_sets_trace_configuration(tmp_path: Path) -> None:
    runtime = create_runtime(tmp_path / "workspace", trace_mode="off")

    assert runtime.trace_mode == "off"


def test_create_runtime_reads_trace_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEXUS_TRACE_MODE", "off")

    runtime = create_runtime(tmp_path / "workspace", trace_mode=None)

    assert runtime.trace_mode == "off"


def test_stream_agent_events_routes_model_chat_without_workspace(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeEntryWorkflow:
        def stream(self, inputs, stream_mode):
            yield (
                "custom",
                {"type": "intent_decision", "route": "chat", "reason": "model classified as greeting", "confidence": 0.91},
            )
            yield (
                "custom",
                {"type": "chat_response", "mode": "lightweight", "reason": "model classified as greeting", "response": "你好，我在。"},
            )

    def fail_complex_workflow():
        raise AssertionError("complex workflow should not be built for chat route")

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: FakeEntryWorkflow())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", fail_complex_workflow)

    events = list(stream_agent_events("你好", workspace=tmp_path, approval_mode="deny"))

    assert [event["event"]["type"] for event in events if event.get("type") == "custom_event"] == ["intent_decision", "chat_response"]
    assert not (tmp_path / ".nexusagent").exists()


def test_stream_agent_events_routes_model_workflow_to_complex_graph(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeEntryWorkflow:
        def stream(self, inputs, stream_mode):
            yield (
                "custom",
                {"type": "intent_decision", "route": "workflow", "reason": "deliverable requested", "confidence": 0.94},
            )

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("updates", {"final": {"final_answer": "PASSED"}})

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: FakeEntryWorkflow())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(stream_agent_events("帮我创建一个 HTML 页面", workspace=tmp_path, checkpoint_mode="off", trace_mode="off", approval_mode="deny"))

    assert events[0]["event"]["type"] == "intent_decision"
    assert any(event.get("type") == "workspace" for event in events)


def test_stream_agent_events_saves_checkpoint_on_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("updates", {"planner": {"plan_summary": "plan", "messages": []}})
            raise KeyboardInterrupt

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: _WorkflowEntry())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(
        stream_agent_events(
            "demo task",
            workspace=tmp_path,
            checkpoint_mode="light",
            approval_mode="deny",
        )
    )

    assert any(event.get("type") == "custom_event" and event["event"].get("type") == "checkpoint_saved" for event in events)
    assert (tmp_path / ".nexusagent" / "checkpoints" / "RECOVERY.md").exists()
    assert "plan" in (tmp_path / ".nexusagent" / "checkpoints" / "RECOVERY.md").read_text(encoding="utf-8")


def test_stream_agent_events_writes_trace_summary_on_finish(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("custom", {"type": "tool_call", "node": "codeAgent", "name": "BashTool", "args": {"command": "true"}})
            yield ("custom", {"type": "tool_result", "node": "codeAgent", "name": "BashTool", "result": {"ok": True}})
            yield ("updates", {"final": {"final_answer": "LangGraph MultiAgent workflow finished: PASSED"}})

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: _WorkflowEntry())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(
        stream_agent_events(
            "demo task",
            workspace=tmp_path,
            checkpoint_mode="off",
            trace_mode="on",
            approval_mode="deny",
        )
    )

    trace_events = [event["event"] for event in events if event.get("type") == "custom_event" and event["event"].get("type") == "trace_summary"]
    assert trace_events
    assert trace_events[-1]["status"] == "finished"
    assert trace_events[-1]["tool_calls"] == 1
    assert (tmp_path / ".nexusagent" / "traces").exists()


def test_stream_agent_events_checkpoints_only_at_safety_points(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("custom", {"type": "tool_call", "node": "codeAgent", "name": "BashTool", "args": {"command": "true"}})
            yield ("custom", {"type": "tool_result", "node": "codeAgent", "name": "BashTool", "result": {"ok": True}})
            yield ("updates", {"planner": {"plan_summary": "plan", "messages": []}})
            yield (
                "custom",
                {
                    "type": "tool_result",
                    "node": "codeAgent",
                    "name": "BashTool",
                    "result": {"ok": False, "requires_approval": True, "approved": False},
                },
            )
            yield ("updates", {"final": {"final_answer": "LangGraph MultiAgent workflow finished: PASSED"}})

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: _WorkflowEntry())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(
        stream_agent_events(
            "demo task",
            workspace=tmp_path,
            checkpoint_mode="light",
            trace_mode="on",
            approval_mode="deny",
        )
    )

    trace_events = [event["event"] for event in events if event.get("type") == "custom_event" and event["event"].get("type") == "trace_summary"]
    assert trace_events
    # started + planner graph update + failed/approval tool result + final graph update + finished
    assert trace_events[-1]["checkpoint_count"] == 5


def test_stream_agent_events_writes_trace_summary_on_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("updates", {"planner": {"plan_summary": "plan", "messages": []}})
            raise KeyboardInterrupt

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: _WorkflowEntry())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(
        stream_agent_events(
            "demo task",
            workspace=tmp_path,
            checkpoint_mode="off",
            trace_mode="on",
            approval_mode="deny",
        )
    )

    trace_events = [event["event"] for event in events if event.get("type") == "custom_event" and event["event"].get("type") == "trace_summary"]
    assert trace_events
    assert trace_events[-1]["status"] == "interrupted"
    assert trace_events[-1]["node_visits"] == {"planner": 1}


def test_stream_agent_events_trace_off_creates_no_trace_dir(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("updates", {"final": {"final_answer": "PASSED"}})

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: _WorkflowEntry())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(
        stream_agent_events(
            "demo task",
            workspace=tmp_path,
            checkpoint_mode="off",
            trace_mode="off",
            approval_mode="deny",
        )
    )

    assert not any(event.get("type") == "custom_event" and event["event"].get("type") == "trace_summary" for event in events)
    assert not (tmp_path / ".nexusagent" / "traces").exists()


def test_stream_agent_events_trace_records_resume(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events
    from nexusagent.core.checkpoint import CheckpointManager
    from nexusagent.core.state import RuntimeState

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("updates", {"final": {"final_answer": "PASSED"}})

    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    CheckpointManager(runtime, task="original task").save(
        {"task": "original task", "runtime": runtime, "messages": [], "max_attempts": 3},
        status="interrupted",
        latest_node="planner",
    )
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    list(
        stream_agent_events(
            workspace=tmp_path,
            resume_workspace=tmp_path,
            checkpoint_mode="off",
            trace_mode="on",
            approval_mode="deny",
        )
    )

    events_files = list((tmp_path / ".nexusagent" / "traces").glob("trace-*/events.jsonl"))
    assert events_files
    content = events_files[0].read_text(encoding="utf-8")
    assert "checkpoint_resumed" in content


def test_stream_agent_events_resume_skips_entry_router(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_agent_events
    from nexusagent.core.checkpoint import CheckpointManager
    from nexusagent.core.state import RuntimeState

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            yield ("updates", {"final": {"final_answer": "PASSED"}})

    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    CheckpointManager(runtime, task="original task").save(
        {"task": "original task", "runtime": runtime, "messages": [], "max_attempts": 3},
        status="interrupted",
        latest_node="planner",
    )

    def fail_entry_workflow():
        raise AssertionError("resume should skip entry router")

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", fail_entry_workflow)
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(stream_agent_events(workspace=tmp_path, resume_workspace=tmp_path, checkpoint_mode="off", trace_mode="off", approval_mode="deny"))

    assert any(event.get("type") == "custom_event" and event["event"].get("type") == "checkpoint_resumed" for event in events)


def test_stream_session_events_chat_writes_session_without_harness(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_session_events

    class FakeEntryWorkflow:
        def stream(self, inputs, stream_mode):
            assert inputs["session_context"]
            yield (
                "custom",
                {"type": "intent_decision", "route": "chat", "reason": "greeting", "confidence": 0.9},
            )
            yield ("updates", {"chat_responder": {"chat_response": "你好，我在。", "final_answer": "你好，我在。"}})

    def fail_complex_workflow():
        raise AssertionError("complex workflow should not run for chat route")

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: FakeEntryWorkflow())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", fail_complex_workflow)

    events = list(stream_session_events("你好", session_workspace=tmp_path, checkpoint_mode="light", trace_mode="on", approval_mode="deny"))

    custom_types = [event["event"]["type"] for event in events if event.get("type") == "custom_event"]
    assert "session_started" in custom_types
    assert "session_turn_saved" in custom_types
    assert not (tmp_path / ".nexusagent" / "checkpoints").exists()
    assert not (tmp_path / ".nexusagent" / "traces").exists()
    assert (tmp_path / ".nexusagent" / "session" / "session.json").exists()
    assert (tmp_path / "SESSION_SUMMARY.md").exists()


def test_stream_session_events_workflow_reuses_workspace_and_session_context(monkeypatch, tmp_path: Path) -> None:
    from nexusagent.core.agent import stream_session_events

    captured = {}

    class FakeEntryWorkflow:
        def stream(self, inputs, stream_mode):
            yield (
                "custom",
                {"type": "intent_decision", "route": "workflow", "reason": "needs files", "confidence": 0.9},
            )

    class FakeWorkflow:
        def stream(self, inputs, stream_mode):
            captured["workspace"] = inputs["runtime"].workspace
            captured["session_context"] = inputs.get("session_context", "")
            captured["session_turn"] = inputs.get("session_turn")
            yield ("updates", {"final": {"final_answer": "PASSED: done"}})

    monkeypatch.setattr("nexusagent.core.agent.build_entry_workflow", lambda: FakeEntryWorkflow())
    monkeypatch.setattr("nexusagent.core.agent.build_complex_workflow", lambda: FakeWorkflow())

    events = list(
        stream_session_events(
            "帮我创建 app.py",
            session_workspace=tmp_path,
            checkpoint_mode="off",
            trace_mode="off",
            approval_mode="deny",
        )
    )

    assert captured["workspace"] == tmp_path
    assert captured["session_turn"] == 1
    assert "帮我创建 app.py" in captured["session_context"]
    assert any(event.get("type") == "workspace" and event.get("path") == str(tmp_path) for event in events)
    assert (tmp_path / ".nexusagent" / "session" / "session.json").exists()


class _WorkflowEntry:
    def stream(self, inputs, stream_mode):
        yield (
            "custom",
            {"type": "intent_decision", "route": "workflow", "reason": "test workflow route", "confidence": 0.9},
        )
