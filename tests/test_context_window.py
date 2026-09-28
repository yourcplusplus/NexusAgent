from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import add_messages

import nexusagent.graph.nodes as nodes
from nexusagent.core.state import RuntimeState
from nexusagent.graph.context_window import (
    DEFAULT_KEEP_GROUPS,
    close_window,
    group_messages,
    plan_eviction,
)
from nexusagent.graph.memory import (
    CRITICAL_CONTEXT_HEADER,
    fallback_narrative_entry,
    merge_narrative,
    render_critical_context,
    rollup_evicted_messages,
)
from nexusagent.graph.nodes import agent_status_bar_route, context_compressor_node


def think(attempt: int, *, call_id: str | None = None, chars: int = 400) -> AIMessage:
    """一次 planner 执行的首条消息;带 tool_calls 时与其结果天然成对。"""
    calls = [{"name": "BashTool", "args": {"command": "pytest"}, "id": call_id}] if call_id else []
    message = AIMessage(content="t" * chars, tool_calls=calls)
    message.id = f"planner-a{attempt}-0001"
    return message


def result_of(attempt: int, *, call_id: str, payload: dict | None = None) -> ToolMessage:
    message = ToolMessage(
        content=json.dumps(payload if payload is not None else {"ok": True, "exit_code": 0}),
        tool_name="BashTool",
        tool_call_id=call_id,
    )
    message.id = f"planner-a{attempt}-0002"
    return message


def verdict(attempt: int, *, chars: int = 400) -> AIMessage:
    message = AIMessage(content="v" * chars)
    message.id = f"verifier-a{attempt}-0001"
    return message


def transcript(attempts: int = 3) -> list:
    messages: list = []
    for attempt in range(1, attempts + 1):
        messages.append(think(attempt, call_id=f"tc{attempt}"))
        messages.append(result_of(attempt, call_id=f"tc{attempt}"))
        messages.append(verdict(attempt))
    return messages


# ---------- 1. 分组 ----------


def test_group_messages_by_id_prefix_keeps_transcript_order() -> None:
    messages = transcript(2)

    groups = group_messages(messages)

    assert [group.key for group in groups] == ["planner-a1", "verifier-a1", "planner-a2", "verifier-a2"]
    assert [len(group.messages) for group in groups] == [2, 1, 2, 1]


def test_leading_human_message_is_pinned_but_later_ones_are_not() -> None:
    first = HumanMessage(content="original task")
    first.id = "task-0001"
    second = HumanMessage(content="later turn")
    second.id = "task-0002"

    groups = group_messages([first, think(1), second, think(2)])

    assert [(group.key, group.pinned) for group in groups] == [
        ("task", True),
        ("planner-a1", False),
        ("task", False),
        ("planner-a2", False),
    ]


def test_unparseable_id_becomes_a_solo_group() -> None:
    orphan = AIMessage(content="legacy")
    orphan.id = "not-a-group-id"

    groups = group_messages([orphan, think(1)])

    assert [(group.key, group.pinned) for group in groups] == [("solo", False), ("planner-a1", False)]


# ---------- 2. 窗口 ----------


def test_window_evicts_oldest_groups_beyond_keep_groups() -> None:
    messages = transcript(3)  # 6 组,每组 400 字符

    plan = plan_eviction(messages, keep_groups=DEFAULT_KEEP_GROUPS)

    assert plan.skipped_reason == ""
    assert plan.evict_ids == ["planner-a1-0001", "planner-a1-0002", "verifier-a1-0001"]
    assert plan.messages_evicted == 3
    assert plan.messages_kept == 6
    assert plan.notes == ["group_count_limit_reached"]


def test_window_token_ceiling_trims_more_groups_than_count_limit() -> None:
    messages = transcript(3)
    # verifier 组 = "v"*400 -> 100 tokens;ceiling = 150 只装得下最新一组
    # (planner 组含 tool_calls 参数,约 112 tokens,100+112 > 150)
    plan = plan_eviction(messages, keep_groups=99, token_limit=150, window_ratio=1.0)

    assert plan.messages_kept == 1
    assert plan.messages_evicted == 8
    assert plan.notes == ["window_token_ceiling_reached"]


def test_newest_group_is_kept_even_when_it_exceeds_the_ceiling() -> None:
    messages = transcript(3)

    plan = plan_eviction(messages, keep_groups=99, token_limit=1, window_ratio=1.0)

    assert plan.messages_kept == 1
    assert plan.messages_evicted == 8
    assert plan.notes == ["window_token_ceiling_reached", "newest_group_exceeds_window"]


# ---------- 3. close_window 两条规则 ----------


def test_close_window_pulls_back_a_result_whose_call_was_kept() -> None:
    """规则 1:调用在窗口内、结果被落在窗口外 -> 把结果拉回来(宁可少逐出)。"""
    call = think(2, call_id="tc2")
    outcome = result_of(2, call_id="tc2")
    other = verdict(2)

    keep, evict = close_window([call], [call, outcome, other])

    assert outcome in keep
    assert outcome not in evict
    assert other in evict


def test_close_window_drops_a_result_whose_call_is_gone() -> None:
    """规则 2:结果在窗口内、调用已被逐出 -> 踢出这个孤儿结果。"""
    call = think(1, call_id="tc1")
    outcome = result_of(1, call_id="tc1")

    keep, evict = close_window([outcome], [call, outcome])

    assert keep == []
    assert outcome in evict


def test_close_window_leaves_a_well_formed_window_untouched() -> None:
    messages = transcript(2)

    keep, evict = close_window(messages[3:], messages)

    assert keep == messages[3:]
    assert evict == messages[:3]


# ---------- 4. id=None 防御 ----------


def test_plan_eviction_skips_instead_of_raising_on_unidentified_messages() -> None:
    legacy = AIMessage(content="old")
    legacy.id = None
    task = HumanMessage(content="task")
    task.id = None
    legacy_new = AIMessage(content="new")
    legacy_new.id = None
    grouped = think(1)

    # 组构成: solo / task(钉住) / solo / planner-a1 -> 最旧的 solo 必须被逐出
    plan = plan_eviction([legacy, task, legacy_new, grouped], keep_groups=1)

    assert plan.skipped_reason == "unidentified_messages"
    assert plan.evict_ids == []
    assert any(note.startswith("unidentified_messages=") for note in plan.notes)


# ---------- 5. 跳过路径 ----------


def test_plan_eviction_reports_no_messages_for_empty_transcript() -> None:
    assert plan_eviction([]).skipped_reason == "no_messages"


def test_plan_eviction_reports_nothing_to_evict_inside_the_window() -> None:
    plan = plan_eviction([think(1)], keep_groups=DEFAULT_KEEP_GROUPS)

    assert plan.skipped_reason == "nothing_to_evict"
    assert plan.evict_ids == []


def test_plan_eviction_reports_empty_window_when_closure_empties_it() -> None:
    orphan = ToolMessage(content="dangling", tool_name="BashTool", tool_call_id="gone")
    orphan.id = "planner-a1-0001"

    plan = plan_eviction([orphan], keep_groups=DEFAULT_KEEP_GROUPS)

    assert plan.skipped_reason == "empty_window"
    assert plan.evict_ids == []


def test_compressor_returns_empty_update_and_event_when_skipped(tmp_path: Path) -> None:
    state = {
        "runtime": RuntimeState(workspace=tmp_path),
        "task": "demo",
        "messages": [think(1)],
        "context_next_node": "planner",
    }

    with patch.object(nodes, "rollup_evicted_messages", side_effect=AssertionError("skipped must not roll up")):
        result = context_compressor_node(state)

    assert result == {}
    assert not (tmp_path / "HISTORY_SUMMARY.md").exists()


# ---------- 6. 增量叙事 ----------


def test_merge_narrative_concatenates_while_under_the_limit() -> None:
    assert merge_narrative("first", "second", max_chars=100) == "first\nsecond"


def test_merge_narrative_folds_oldest_entries_and_keeps_newest_intact() -> None:
    # 30 条 × 209 字符 ≈ 6270,远超 400;折叠后必须收进上限且最新条目原样
    existing = "\n".join(f"entry-{index:02d} " + "z" * 200 for index in range(30))

    merged = merge_narrative(existing, "newest entry", max_chars=400)

    lines = merged.splitlines()
    assert len(merged) <= 400
    assert lines[0].startswith("[... ")
    assert lines[-1] == "newest entry"
    assert any("entry-" in line for line in lines[1:])


def test_merge_narrative_truncates_a_single_oversized_entry() -> None:
    merged = merge_narrative("", "n" * 500, max_chars=100)

    assert len(merged) == 100
    assert merged.endswith("...")


def test_merge_narrative_keeps_newest_entry_complete_across_rounds() -> None:
    narrative = ""
    for round_index in range(6):
        narrative = merge_narrative(narrative, f"round-{round_index} " + "q" * 300, max_chars=1000)
        assert narrative.splitlines()[-1].startswith(f"round-{round_index}")


# ---------- 7. 确定性 fallback ----------


def test_fallback_narrative_entry_keeps_artifact_pointer_and_exit_code() -> None:
    call = think(1, call_id="tc1")
    outcome = result_of(
        1,
        call_id="tc1",
        payload={
            "ok": False,
            "exit_code": 1,
            "stderr": "boom",
            "stdout": "x" * 4000,
            "artifact_path": ".nexusagent/tool-outputs/BashTool-9.json",
        },
    )

    entry = fallback_narrative_entry([call, outcome])

    assert "artifact=.nexusagent/tool-outputs/BashTool-9.json" in entry
    assert "exit=1" in entry
    assert "err=boom" in entry


def test_rollup_falls_back_when_the_model_is_unavailable() -> None:
    class Boom:
        def invoke(self, messages):  # type: ignore[no-untyped-def]
            raise RuntimeError("no api key")

    outcome = result_of(1, call_id="tc1", payload={"ok": True, "artifact_path": "a.json"})

    with patch("nexusagent.graph.memory.create_model", lambda: Boom()):
        entry, used_fallback = rollup_evicted_messages([outcome], current_narrative="prior")

    assert used_fallback is True
    assert "artifact=a.json" in entry


def test_rollup_uses_the_model_entry_when_available() -> None:
    class Ok:
        def invoke(self, messages):  # type: ignore[no-untyped-def]
            return AIMessage(content='```json\n{"summary": "ran pytest, exit 1"}\n```')

    with patch("nexusagent.graph.memory.create_model", lambda: Ok()):
        entry, used_fallback = rollup_evicted_messages([think(1)], current_narrative="prior")

    assert used_fallback is False
    assert entry == "ran pytest, exit 1"


def test_rollup_of_nothing_is_a_no_op() -> None:
    assert rollup_evicted_messages([]) == ("", False)


# ---------- 8. 端到端:压缩后白名单仍可重建 ----------


def test_compression_evicts_by_group_and_keeps_whitelist_out_of_narrative(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    runtime.record_touch("src/app.py", op="write", via="file_tools")
    messages = transcript(3)
    state = {
        "runtime": runtime,
        "task": "demo task",
        "messages": messages,
        "todos": [{"id": "todo-1", "content": "verify", "status": "pending", "note": ""}],
        "acceptance_criteria": ["done"],
        "context_next_node": "verifier",
        "context_token_count": 5000,
    }

    with patch.object(nodes, "rollup_evicted_messages", lambda evicted, **kwargs: ("rolled-up entry", False)):
        result = context_compressor_node(state)

    # 逐条逐出,绝不全量清空
    assert all(isinstance(message, RemoveMessage) for message in result["messages"])
    assert [message.id for message in result["messages"]] == [
        "planner-a1-0001",
        "planner-a1-0002",
        "verifier-a1-0001",
    ]
    # 经 reducer 合并后,窗口内只剩较新的两组
    merged = add_messages(messages, result["messages"])
    assert [message.id for message in merged] == [
        "planner-a2-0001",
        "planner-a2-0002",
        "verifier-a2-0001",
        "planner-a3-0001",
        "planner-a3-0002",
        "verifier-a3-0001",
    ]

    # 叙事承载压缩结果,且不搬运白名单
    assert result["history_summary"] == "rolled-up entry"
    assert result["context_summary"] == "rolled-up entry"
    assert CRITICAL_CONTEXT_HEADER not in result["history_summary"]
    persisted = (tmp_path / "HISTORY_SUMMARY.md").read_text(encoding="utf-8")
    assert "rolled-up entry" in persisted
    assert CRITICAL_CONTEXT_HEADER not in persisted

    # 白名单仍能由实时 state 独立重建,且含压缩前记录的文件足迹
    block = render_critical_context({**state, **result})
    assert block.startswith(CRITICAL_CONTEXT_HEADER)
    assert "src/app.py (write," in block

    # 事件字段
    event = result["compression_events"][0]
    assert event["evicted_messages"] == 3
    assert event["kept_messages"] == 6
    assert event["groups_evicted"] == 2
    assert event["skipped_reason"] == ""
    assert event["before_tokens"] == 5000
    assert event["after_tokens"] < event["before_tokens"]
    assert event["removed_messages"] == 3
    assert isinstance(event["before_tokens_exact"], int)
    # 路由职责已并入状态栏节点:压缩后由它按 context_next_node 决定去向
    assert agent_status_bar_route({**state, **result}) == "verifier"


def test_compressor_skips_and_leaves_state_untouched_when_window_is_minimal(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    state = {
        "runtime": runtime,
        "task": "demo",
        "messages": [think(1)],
        "context_next_node": "verifier",
        "context_token_count": 100,
    }

    with patch.object(nodes, "rollup_evicted_messages", lambda evicted, **kwargs: ("entry", False)):
        first = context_compressor_node(state)
        second = context_compressor_node({**state, **first})

    assert first == {}
    assert second == {}
