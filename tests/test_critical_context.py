from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from nexusagent.agents.code_agent import _code_agent_input
from nexusagent.agents.search_agent import run_search_agent
from nexusagent.core.state import ARTIFACT_OP, MAX_ARTIFACTS, MAX_TOUCHED_FILES, RuntimeState
from nexusagent.graph.memory import (
    CRITICAL_CONTEXT_FOOTER,
    CRITICAL_CONTEXT_HEADER,
    _artifact_size_text,
    _format_file_row,
    _short_time,
    build_layered_memory,
    format_layered_memory_for_prompt,
    render_critical_context,
    rollup_evicted_messages,
)
from nexusagent.graph.nodes import _planner_input, _verifier_input

ARTIFACT_PATH = ".nexusagent/tool-outputs/BashTool-1.json"


def make_graph_state(tmp_path: Path) -> dict:
    runtime = RuntimeState(workspace=tmp_path)
    runtime.record_touch("src/app.py", op="write", via="file_tools")
    runtime.record_touch(
        ARTIFACT_PATH,
        op=ARTIFACT_OP,
        via="BashTool",
        meta={
            "source_lines": 3000,
            "source_bytes": 185999,
            "json_lines": 7,
            "json_bytes": 189093,
            "source": "pytest -x -q",
        },
    )
    return {
        "runtime": runtime,
        "task": "build the page",
        "plan_summary": "plan",
        "todos": [{"id": "todo-1", "content": "write page", "status": "in_progress", "note": ""}],
        "acceptance_criteria": ["page exists"],
        # 快照写成字面量,不调 refresh_env_status:这里测的是渲染契约,
        # 真实探针(子进程、TTL)由 test_status_bar.py 覆盖,不在此处付进程开销
        "env_status": {
            "git": {"is_repo": True, "repo": "/tmp/proj", "branch": "main", "tracked": False},
            "delta": {"baseline": False, "added": 1, "modified": 0, "deleted": 0, "samples": [{"op": "+", "path": "report.html"}]},
        },
    }


# ---------- 1. 六个板块与空状态退化 ----------


def test_render_critical_context_emits_all_six_sections(tmp_path: Path) -> None:
    text = render_critical_context(make_graph_state(tmp_path))

    assert text.startswith(CRITICAL_CONTEXT_HEADER)
    assert text.endswith(CRITICAL_CONTEXT_FOOTER)
    assert "[Goal] build the page" in text
    assert "  - page exists" in text
    assert "  - todo-1 [in_progress] write page" in text
    assert "src/app.py (write," in text
    assert "[Git] repo=/tmp/proj branch=main" in text
    assert "workspace not tracked by this repo" in text
    assert "delta +1 ~0 -0 +report.html" in text
    assert "from pytest -x -q" in text


def test_render_critical_context_degrades_to_none_on_empty_state(tmp_path: Path) -> None:
    text = render_critical_context({"runtime": RuntimeState(workspace=tmp_path)})

    assert "[Goal] (none)" in text
    assert "[Constraints] (none)" in text
    assert "[TODOs] (none)" in text
    assert "[Files] (none)" in text
    assert "[Artifacts] (none)" in text
    # 还没采集过快照时才回落 pending(生产路径上节点入口保证快照存在)
    assert "[Git] (git: pending)" in text


def test_render_critical_context_survives_missing_runtime() -> None:
    text = render_critical_context({})

    assert text.startswith(CRITICAL_CONTEXT_HEADER)
    assert "[Files] (none)" in text
    assert "[Artifacts] (none)" in text


def test_render_critical_context_lists_files_most_recent_first(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    runtime.record_touch("first.py", op="read", via="file_tools")
    runtime.record_touch("second.py", op="write", via="file_tools")
    runtime.record_touch("first.py", op="edit", via="file_tools")

    entries = [
        line.removeprefix("  - ")
        for line in render_critical_context({"runtime": runtime}).splitlines()
        if line.startswith("  - ")
    ]

    assert entries[0].startswith("first.py (edit")
    assert entries[1].startswith("second.py (write")


# ---------- 2. record_touch 分流 ----------


def test_record_touch_routes_artifacts_away_from_touched_files(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)

    runtime.record_touch("src/app.py", op="write", via="file_tools")
    runtime.record_touch(ARTIFACT_PATH, op=ARTIFACT_OP, via="BashTool")

    assert list(runtime.touched_files) == ["src/app.py"]
    assert list(runtime.artifacts) == [ARTIFACT_PATH]


def test_artifact_entries_keep_metadata(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)

    runtime.record_touch(ARTIFACT_PATH, op=ARTIFACT_OP, via="BashTool", meta={"source_lines": 12})

    assert runtime.artifacts[ARTIFACT_PATH]["meta"] == {"source_lines": 12}
    assert runtime.artifacts[ARTIFACT_PATH]["via"] == "BashTool"


def test_plain_touch_has_no_meta_key(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)

    runtime.record_touch("src/app.py", op="read")

    assert "meta" not in runtime.touched_files["src/app.py"]


# ---------- 3. LRU 边界 ----------


def test_touched_files_evicts_oldest_beyond_cap(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    for index in range(MAX_TOUCHED_FILES + 1):
        runtime.record_touch(f"src/f{index}.py", op="read")

    assert len(runtime.touched_files) == MAX_TOUCHED_FILES
    assert "src/f0.py" not in runtime.touched_files
    assert f"src/f{MAX_TOUCHED_FILES}.py" in runtime.touched_files


def test_artifacts_evict_oldest_beyond_cap(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    for index in range(MAX_ARTIFACTS + 1):
        runtime.record_touch(f".nexusagent/tool-outputs/a{index}.json", op=ARTIFACT_OP)

    assert len(runtime.artifacts) == MAX_ARTIFACTS
    assert ".nexusagent/tool-outputs/a0.json" not in runtime.artifacts
    assert f".nexusagent/tool-outputs/a{MAX_ARTIFACTS}.json" in runtime.artifacts


def test_many_reads_do_not_evict_artifacts(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    runtime.record_touch(ARTIFACT_PATH, op=ARTIFACT_OP)
    for index in range(MAX_TOUCHED_FILES * 3):
        runtime.record_touch(f"src/f{index}.py", op="read")

    assert list(runtime.artifacts) == [ARTIFACT_PATH]


# ---------- 4. source_* / json_* 整组退让 ----------


def test_artifact_size_prefers_the_source_group() -> None:
    meta = {"source_lines": 3000, "source_bytes": 185999, "json_lines": 7, "json_bytes": 189093}

    assert _artifact_size_text(meta) == "3,000 lines / 185,999 bytes"


def test_artifact_size_falls_back_as_a_whole_group() -> None:
    """只有半组 source_* 时必须整组退回,否则会出现「3000 行 / 189KB」跨组混搭。"""
    meta = {"source_lines": 3000, "json_lines": 7, "json_bytes": 189093}

    assert _artifact_size_text(meta) == "7 lines / 189,093 bytes"


def test_artifact_size_without_metadata_is_empty() -> None:
    assert _artifact_size_text({}) == ""


# ---------- 5. 四个注入点:块在最前且仅一次 ----------


def test_block_leads_planner_verifier_and_code_agent_prompts(tmp_path: Path) -> None:
    state = make_graph_state(tmp_path)
    memory = build_layered_memory(state, node="planner")

    prompts = {
        "planner": _planner_input(state, memory),
        "verifier": _verifier_input(state, memory),
        "codeAgent": _code_agent_input(state, "implement it", memory),
    }

    for name, text in prompts.items():
        assert text.startswith(CRITICAL_CONTEXT_HEADER), name
        assert text.count(CRITICAL_CONTEXT_HEADER) == 1, name
        assert text.count(CRITICAL_CONTEXT_FOOTER) == 1, name
        assert text.index(CRITICAL_CONTEXT_FOOTER) < text.index("Task:"), name


def test_block_leads_search_agent_prompt(tmp_path: Path) -> None:
    captured: list = []

    class FakeModel:
        def bind_tools(self, tools):  # type: ignore[no-untyped-def]
            return self

        def invoke(self, messages):  # type: ignore[no-untyped-def]
            captured.extend(messages)
            return AIMessage(content="research done", tool_calls=[])

    with patch("nexusagent.agents.search_agent.create_model", lambda: FakeModel()):
        run_search_agent(make_graph_state(tmp_path), "find sources", writer=lambda _: None)

    human = captured[1].content
    assert human.startswith(CRITICAL_CONTEXT_HEADER)
    assert human.count(CRITICAL_CONTEXT_HEADER) == 1
    assert human.index(CRITICAL_CONTEXT_FOOTER) < human.index("Task:")


# ---------- 6. formatter 剔除该层,压缩路径不搬运该层 ----------


def test_formatter_strips_critical_context_layer(tmp_path: Path) -> None:
    memory = build_layered_memory(make_graph_state(tmp_path), node="planner")
    assert "critical_context" in memory

    payload = json.loads(format_layered_memory_for_prompt(memory))

    assert "critical_context" not in payload
    assert set(payload) == {"rules", "working_memory", "history_summary_store"}


def test_rollup_never_receives_the_whitelist(tmp_path: Path) -> None:
    """压缩只搬运被逐出的消息与叙事,白名单不经这条路径,因此不可能被压缩丢失。"""
    captured: list = []

    class FakeModel:
        def invoke(self, messages):  # type: ignore[no-untyped-def]
            captured.extend(messages)
            return AIMessage(content='{"summary": "entry"}')

    outcome = ToolMessage(
        content=json.dumps({"ok": True, "artifact_path": ".nexusagent/tool-outputs/a.json"}),
        name="BashTool",
        tool_call_id="tc1",
    )
    outcome.id = "planner-a1-0002"

    with patch("nexusagent.graph.memory.create_model", lambda: FakeModel()):
        entry, used_fallback = rollup_evicted_messages([outcome], current_narrative="prior narrative")

    assert used_fallback is False
    assert entry == "entry"
    payload_text = captured[1].content
    assert CRITICAL_CONTEXT_HEADER not in payload_text
    assert set(json.loads(payload_text)) == {"current_narrative", "evicted_messages"}


# ---------- 7/8. 时间戳渲染 ----------


def test_file_row_shortens_iso_timestamp() -> None:
    row = _format_file_row(
        {"path": "src/app.py", "op": "write", "at": "2026-09-26T13:41:03.228688+00:00", "via": "file_tools"}
    )

    assert row == "src/app.py (write, 09-26 13:41, file_tools)"


def test_short_time_returns_invalid_input_unchanged() -> None:
    assert _short_time("not-a-timestamp") == "not-a-timestamp"
    assert _short_time("") == ""
    assert _short_time(None) is None  # type: ignore[arg-type]


def test_file_row_tolerates_bad_timestamp() -> None:
    row = _format_file_row({"path": "a.py", "op": "read", "at": "garbage", "via": ""})

    assert row == "a.py (read, garbage)"
