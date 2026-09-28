from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage

from nexusagent.core.checkpoint import (
    CHECKPOINT_ROOT,
    CheckpointManager,
    build_light_resume_inputs,
    load_resume_inputs,
    normalize_resume_task,
    serialize_state,
    deserialize_state,
    workspace_manifest,
)
from nexusagent.core.state import RuntimeState


def sample_state(runtime: RuntimeState) -> dict:
    return {
        "task": "build demo",
        "runtime": runtime,
        "messages": [AIMessage(content="hello")],
        "plan_summary": "demo plan",
        "todos": [{"id": "todo-1", "content": "write app", "status": "in_progress", "note": ""}],
        "acceptance_criteria": ["app exists"],
        "verification_commands": ["python --version"],
        "sources": [{"title": "Docs", "url": "https://example.com"}],
        "research_notes": "notes",
        "attempts": 1,
        "max_attempts": 3,
        "context_next_node": "verifier",
    }


def test_light_checkpoint_writes_recovery_and_checkpoint(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    (tmp_path / "TODO.md").write_text("todo content", encoding="utf-8")

    event = CheckpointManager(runtime, task="build demo").save(sample_state(runtime), status="running", latest_node="planner")

    root = tmp_path / CHECKPOINT_ROOT
    assert event is not None
    assert event["type"] == "checkpoint_saved"
    assert (root / "checkpoint.json").exists()
    assert (root / "RECOVERY.md").exists()
    recovery = (root / "RECOVERY.md").read_text(encoding="utf-8")
    assert "demo plan" in recovery
    assert "write app" in recovery
    assert not (root / "state.json").exists()


def test_strict_checkpoint_writes_state_and_events(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="strict")
    manager = CheckpointManager(runtime, task="build demo")

    manager.save(sample_state(runtime), status="running", latest_node="planner", event={"mode": "custom", "payload": {"type": "x"}})

    root = tmp_path / CHECKPOINT_ROOT
    assert (root / "state.json").exists()
    assert (root / "events.jsonl").exists()
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert "runtime" not in state
    assert state["messages"][0]["type"] == "ai"


def test_state_serialization_restores_messages_and_runtime(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="strict")
    payload = serialize_state(sample_state(runtime))

    restored = deserialize_state(payload, runtime)

    assert restored["runtime"] is runtime
    assert restored["messages"][0].content == "hello"
    assert restored["plan_summary"] == "demo plan"


def test_workspace_manifest_excludes_checkpoint_directory(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hi')", encoding="utf-8")
    checkpoint_file = tmp_path / CHECKPOINT_ROOT / "checkpoint.json"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_text("{}", encoding="utf-8")

    manifest = workspace_manifest(tmp_path)

    assert [item["path"] for item in manifest] == ["app.py"]


def test_light_checkpoint_git_snapshot_handles_relative_workspace(tmp_path: Path, monkeypatch) -> None:
    if shutil.which("git") is None:
        return
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('hi')", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runtime = RuntimeState(workspace=Path("workspace"), checkpoint_mode="light")

    event = CheckpointManager(runtime, task="relative workspace").save(sample_state(runtime), status="running", latest_node="planner")

    assert event is not None
    assert event["git_error"] in {None, ""}
    assert event["git_commit"]


def test_light_resume_context_includes_workspace_memory(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    (tmp_path / "TODO.md").write_text("TODO from disk", encoding="utf-8")
    (tmp_path / "NOTEPAD.md").write_text("Important note", encoding="utf-8")
    (tmp_path / "HISTORY_SUMMARY.md").write_text("History summary", encoding="utf-8")
    CheckpointManager(runtime, task="original task").save(sample_state(runtime), status="interrupted", latest_node="verifier")

    inputs = build_light_resume_inputs(runtime, max_attempts=5)

    assert "Continue this NexusAgent task" in inputs["task"]
    assert "TODO from disk" in inputs["context_summary"]
    assert "Important note" in inputs["context_summary"]
    assert "History summary" in inputs["context_summary"]
    assert inputs["plan_summary"] == "demo plan"
    assert inputs["max_attempts"] == 5


def test_light_resume_normalizes_repeated_resume_prefix(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="light")
    repeated = (
        "Continue this NexusAgent task from the checkpoint: "
        "Continue the interrupted NexusAgent task from the checkpoint: "
        "original task"
    )
    CheckpointManager(runtime, task=repeated).save({**sample_state(runtime), "task": repeated}, status="interrupted", latest_node="planner")

    inputs = build_light_resume_inputs(runtime)

    assert inputs["task"] == "Continue this NexusAgent task from the checkpoint: original task"


def test_normalize_resume_task_strips_nested_prefixes() -> None:
    assert normalize_resume_task(
        "Continue this NexusAgent task from the checkpoint: Continue this NexusAgent task from the checkpoint: demo"
    ) == "demo"


def test_strict_resume_falls_back_to_light_when_state_missing(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path, checkpoint_mode="strict")
    (tmp_path / "TODO.md").write_text("resume todo", encoding="utf-8")
    CheckpointManager(RuntimeState(workspace=tmp_path, checkpoint_mode="light"), task="original").save(
        sample_state(runtime),
        status="interrupted",
        latest_node="planner",
    )

    inputs, event = load_resume_inputs(runtime, max_attempts=2)

    assert event["type"] == "checkpoint_resumed"
    assert event["fallback"] is True
    assert event["mode"] == "light"
    assert "resume todo" in inputs["context_summary"]
