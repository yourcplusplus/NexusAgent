from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import add_messages

from nexusagent.core.state import RuntimeState
from nexusagent.graph.memory import (
    CRITICAL_CONTEXT_HEADER,
    build_layered_memory,
    persist_history_summary,
    read_history_summary,
)
from nexusagent.graph.nodes import (
    _call_code_agent_tool,
    _call_search_agent_tool,
    agent_status_bar_route,
    chat_responder_node,
    context_compressor_node,
    context_monitor_node,
    context_monitor_route,
    count_tokens_exact,
    estimate_context_tokens,
    final_node,
    get_context_token_limit,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from nexusagent.graph.workflow import build_workflow


def test_model_verifier_passes_from_json(monkeypatch, tmp_path: Path) -> None:
    class FakeBoundModel:
        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "passed": True,
                        "reason": "HTML file satisfies the request.",
                        "checks": [{"name": "html", "passed": True, "detail": "ok"}],
                        "recommended_next_instruction": "",
                    }
                )
            )

    class FakeModel:
        def bind_tools(self, tools):
            return FakeBoundModel()

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())
    state = {
        "runtime": RuntimeState(workspace=tmp_path),
        "task": "demo",
        "todos": [{"id": "todo-1", "content": "verify", "status": "in_progress", "note": ""}],
        "attempts": 0,
        "max_attempts": 3,
    }

    result = verifier_node(state)

    assert result["passed"] is True
    assert result["attempts"] == 1
    assert result["todos"][0]["status"] == "completed"
    assert result["verification_checks"][0]["name"] == "html"
    assert verifier_route({**state, **result}) == "final"


def test_model_verifier_invalid_json_fails_and_routes_back(monkeypatch, tmp_path: Path) -> None:
    class FakeBoundModel:
        def invoke(self, messages):
            return AIMessage(content="not json")

    class FakeModel:
        def bind_tools(self, tools):
            return FakeBoundModel()

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())
    state = {
        "runtime": RuntimeState(workspace=tmp_path),
        "task": "demo",
        "attempts": 0,
        "max_attempts": 3,
    }

    result = verifier_node(state)

    assert result["passed"] is False
    assert "valid JSON" in result["last_error"]
    assert verifier_route({**state, **result}) == "planner"


def test_verifier_routes_to_final_at_max_attempts() -> None:
    assert verifier_route({"passed": False, "attempts": 3, "max_attempts": 3}) == "final"


def test_final_node_reports_multiagent_status() -> None:
    result = final_node(
        {
            "passed": True,
            "plan_summary": "demo plan",
            "todos": [{"content": "write page", "status": "completed"}],
            "verification_checks": [{"name": "html", "passed": True, "detail": "ok"}],
            "sources": [{"title": "source", "url": "https://example.com"}],
            "code_agent_summary": "done",
            "verifier_summary": "looks good",
        }
    )

    assert "PASSED" in result["final_answer"]
    assert "demo plan" in result["final_answer"]
    assert "https://example.com" in result["final_answer"]


def test_workflow_compiles_without_fixed_actor_node() -> None:
    workflow = build_workflow()

    assert workflow is not None


def test_intent_router_routes_chat_with_model_json(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content='{"route":"chat","reason":"greeting","confidence":0.92}')

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())

    result = intent_router_node({"task": "你好"})

    assert result["intent_route"] == "chat"
    assert result["intent_reason"] == "greeting"
    assert result["intent_confidence"] == 0.92
    assert intent_route_fn(result) == "chat_responder"


def test_intent_router_routes_workflow_with_model_json(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content='{"route":"workflow","reason":"needs files","confidence":0.88}')

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())

    result = intent_router_node({"task": "帮我创建一个 HTML 页面"})

    assert result["intent_route"] == "workflow"
    assert intent_route_fn(result) == "planner"


def test_intent_router_invalid_json_defaults_to_workflow(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content="not json")

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())

    result = intent_router_node({"task": "你好"})

    assert result["intent_route"] == "workflow"
    assert intent_route_fn(result) == "planner"


def test_chat_responder_node_returns_chat_response(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content="你好，我在。")

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())

    result = chat_responder_node({"task": "你好", "intent_reason": "greeting"})

    assert result["chat_response"] == "你好，我在。"
    assert result["final_answer"] == "你好，我在。"


def test_context_token_limit_defaults_and_env(monkeypatch) -> None:
    # getter 住在 graph/config.py(Phase 4 抽出以避开 status_bar/code_agent 的循环导入),
    # patch 目标随实现一起搬
    monkeypatch.setattr("nexusagent.graph.config.load_dotenv", lambda: None)
    monkeypatch.delenv("NEXUS_CONTEXT_TOKEN_LIMIT", raising=False)
    assert get_context_token_limit() == 400000

    monkeypatch.setenv("NEXUS_CONTEXT_TOKEN_LIMIT", "1234")
    assert get_context_token_limit() == 1234


def test_count_tokens_exact_uses_model_counter(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        def get_num_tokens_from_messages(self, messages):
            return 42 + len(messages)

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())
    result = count_tokens_exact(
        {
            "runtime": RuntimeState(workspace=tmp_path),
            "task": "demo",
            "messages": [HumanMessage(content="hello")],
        }
    )

    assert result == 44


def test_estimate_context_tokens_is_incremental_without_model_or_memory_rebuild(monkeypatch, tmp_path: Path) -> None:
    """阈值判断走增量估算:不调 tokenizer、也不重建记忆载荷(否则 monitor 会读盘)。"""
    payload = {"working_memory": {"task": "demo"}}

    def explode(*args, **kwargs):
        raise AssertionError("estimate_context_tokens 不应重建记忆载荷")

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", explode)
    monkeypatch.setattr("nexusagent.graph.nodes.build_layered_memory", explode)
    result = estimate_context_tokens(
        {
            "runtime": RuntimeState(workspace=tmp_path),
            "task": "demo",
            "messages": [HumanMessage(content="x" * 400)],
            "memory_snapshot": payload,
        }
    )

    # 窗口 400//4 = 100,加上载荷的字符量估算
    assert result == 100 + len(json.dumps(payload, ensure_ascii=False)) // 4


def test_context_monitor_does_not_compress_below_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEXUS_CONTEXT_TOKEN_LIMIT", "100")
    monkeypatch.setattr("nexusagent.graph.nodes.estimate_context_tokens", lambda state: 10)
    result = context_monitor_node(
        {
            "runtime": RuntimeState(workspace=tmp_path),
            "task": "demo",
            "context_next_node": "verifier",
        }
    )

    assert result["context_should_compress"] is False
    assert result["context_next_node"] == "verifier"
    # 目标节点改由状态栏节点承接(它按 context_next_node 决定去向)
    assert context_monitor_route(result) == "agent_status_bar"
    assert agent_status_bar_route(result) == "verifier"


def test_context_monitor_compresses_at_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEXUS_CONTEXT_TOKEN_LIMIT", "100")
    monkeypatch.setattr("nexusagent.graph.nodes.estimate_context_tokens", lambda state: 100)
    result = context_monitor_node(
        {
            "runtime": RuntimeState(workspace=tmp_path),
            "task": "demo",
            "context_next_node": "planner",
        }
    )

    assert result["context_should_compress"] is True
    assert context_monitor_route(result) == "context_compressor"


def test_context_compressor_evicts_oldest_groups_and_preserves_state(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}

    def fake_estimate(state):
        calls["count"] += 1
        return 1000 if calls["count"] == 1 else 50

    monkeypatch.setattr("nexusagent.graph.nodes.estimate_context_tokens", fake_estimate)
    monkeypatch.setattr(
        "nexusagent.graph.nodes.rollup_evicted_messages",
        lambda evicted, **kwargs: ("rolled-up summary", False),
    )

    def attempt_messages(attempt: int) -> list:
        call = AIMessage(
            content="thinking " * 20,
            tool_calls=[{"name": "BashTool", "args": {}, "id": f"tc{attempt}"}],
        )
        call.id = f"planner-a{attempt}-0001"
        outcome = ToolMessage(content=json.dumps({"ok": True}), name="BashTool", tool_call_id=f"tc{attempt}")
        outcome.id = f"planner-a{attempt}-0002"
        verdict_message = AIMessage(content="verdict")
        verdict_message.id = f"verifier-a{attempt}-0001"
        return [call, outcome, verdict_message]

    messages = attempt_messages(1) + attempt_messages(2) + attempt_messages(3)
    state = {
        "runtime": RuntimeState(workspace=tmp_path),
        "task": "demo",
        "messages": messages,
        "plan_summary": "plan",
        "todos": [{"id": "todo-1", "content": "verify", "status": "pending", "note": ""}],
        "acceptance_criteria": ["done"],
        "verification_commands": ["python --version"],
        "research_notes": "research " * 100,
        "context_next_node": "verifier",
    }

    result = context_compressor_node(state)

    # 逐条逐出最旧的执行组,而非全量清空
    assert all(isinstance(message, RemoveMessage) for message in result["messages"])
    assert [message.id for message in result["messages"]] == [
        "planner-a1-0001",
        "planner-a1-0002",
        "verifier-a1-0001",
    ]
    # 叙事进 state 与磁盘
    assert "rolled-up summary" in result["context_summary"]
    assert "rolled-up summary" in result["history_summary"]
    assert "rolled-up summary" in (tmp_path / "HISTORY_SUMMARY.md").read_text(encoding="utf-8")
    # 事件报告
    event = result["compression_events"][0]
    assert event["before_tokens"] == 1000
    assert event["after_tokens"] == 50
    assert event["removed_messages"] == 3
    assert event["evicted_messages"] == 3
    assert event["skipped_reason"] == ""
    # 结构化状态不受压缩影响
    assert state["todos"][0]["content"] == "verify"
    # 合并后窗口内只剩较新的两组
    merged = add_messages(messages, result["messages"])
    assert [message.id for message in merged] == [
        "planner-a2-0001",
        "planner-a2-0002",
        "verifier-a2-0001",
        "planner-a3-0001",
        "planner-a3-0002",
        "verifier-a3-0001",
    ]
    assert agent_status_bar_route({**state, **result}) == "verifier"


def test_planner_falls_back_to_default_plan(monkeypatch, tmp_path: Path) -> None:
    class FakeBoundModel:
        def invoke(self, messages):
            return AIMessage(content="plan ready")

    class FakeModel:
        def bind_tools(self, tools):
            return FakeBoundModel()

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())
    result = planner_node(
        {
            "task": "写一个 Python 的 inventory 包，配 pytest 测试并全部跑通",
            "runtime": RuntimeState(workspace=tmp_path),
            "attempts": 0,
            "max_attempts": 3,
        }
    )

    assert result["verification_commands"] == []
    assert result["todos"][0]["id"] == "todo-1"
    assert result["todos"][0]["status"] == "pending"
    assert (tmp_path / "TODO.md").exists()
    assert "Clarify the deliverable" in (tmp_path / "TODO.md").read_text(encoding="utf-8")


def test_call_search_agent_tool_updates_state(monkeypatch, tmp_path: Path) -> None:
    def fake_search_agent(state, instruction, *, writer=None, max_loops=4):
        return {
            "summary": "The walrus operator assigns inside expressions.",
            "sources": [{"title": "PEP 572", "url": "https://example.com/pep-572"}],
            "queries": ["Python walrus operator"],
        }

    monkeypatch.setattr("nexusagent.graph.nodes.run_search_agent", fake_search_agent)
    state = {"task": "写一篇 Python walrus operator 的介绍", "runtime": RuntimeState(workspace=tmp_path)}

    result = _call_search_agent_tool(state, lambda event: None, "research the walrus operator")

    assert result["ok"] is True
    assert "walrus operator" in state["research_notes"]
    assert state["sources"][0]["url"] == "https://example.com/pep-572"
    assert state["agent_handoffs"][0]["to_agent"] == "searchAgent"


def test_call_code_agent_tool_updates_state(monkeypatch, tmp_path: Path) -> None:
    def fake_code_agent(state, instruction, *, writer=None, max_loops=10):
        return {
            "summary": "Created inventory.py",
            "todos": [{"id": "todo-1", "content": "write", "status": "completed", "note": ""}],
        }

    monkeypatch.setattr("nexusagent.graph.nodes.run_code_agent", fake_code_agent)
    state = {
        "task": "写一个 inventory 包",
        "runtime": RuntimeState(workspace=tmp_path),
        "todos": [{"id": "todo-1", "content": "write", "status": "pending", "note": ""}],
    }

    result = _call_code_agent_tool(state, lambda event: None, "write page")

    assert result["ok"] is True
    assert state["code_agent_summary"] == "Created inventory.py"
    assert state["todos"][0]["status"] == "completed"
    assert state["agent_handoffs"][0]["to_agent"] == "codeAgent"


def test_layered_memory_splits_rules_working_and_history(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    (tmp_path / "NOTEPAD.md").write_text("# NexusAgent Notepad\n\nImportant durable note.\n", encoding="utf-8")
    persist_history_summary(runtime, "Previous compressed history.")

    memory = build_layered_memory(
        {
            "runtime": runtime,
            "task": "demo",
            "plan_summary": "demo plan",
            "todos": [{"id": "todo-1", "content": "write", "status": "pending", "note": ""}],
            "acceptance_criteria": ["file exists"],
            "verification_commands": ["python --version"],
            "research_notes": "research",
            "sources": [{"title": "source", "url": "https://example.com"}],
        },
        node="planner",
    )

    assert set(memory) == {"rules", "critical_context", "working_memory", "history_summary_store"}
    assert memory["critical_context"].startswith(CRITICAL_CONTEXT_HEADER)
    assert memory["rules"]["scope"] == "workspace"
    assert memory["working_memory"]["task"] == "demo"
    assert memory["working_memory"]["todos"][0]["content"] == "write"
    assert memory["working_memory"]["sources"][0]["url"] == "https://example.com"
    assert memory["history_summary_store"]["notepad_exists"] is True
    assert "Important durable note" in memory["history_summary_store"]["notepad"]
    assert "Previous compressed history" in memory["history_summary_store"]["history_summary"]


def test_layered_memory_trims_long_history_and_handoffs(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path)
    (tmp_path / "NOTEPAD.md").write_text("note " * 1000, encoding="utf-8")
    handoffs = [
        {"from_agent": "planner", "to_agent": "codeAgent", "instruction": "i" * 1000, "result": "r" * 1000}
        for _ in range(8)
    ]

    memory = build_layered_memory(
        {
            "runtime": runtime,
            "task": "demo",
            "research_notes": "research " * 1000,
            "agent_handoffs": handoffs,
        },
        node="planner",
    )

    assert len(memory["working_memory"]["research_notes"]) <= 1600
    assert len(memory["working_memory"]["agent_handoffs"]) == 6
    assert len(memory["working_memory"]["agent_handoffs"][0]["instruction"]) <= 500
    assert len(memory["history_summary_store"]["notepad"]) <= 1800


def test_history_summary_read_missing_file(tmp_path: Path) -> None:
    result = read_history_summary(RuntimeState(workspace=tmp_path))

    assert result["ok"] is True
    assert result["exists"] is False
    assert result["content"] == ""


# ---------- 工具名别名(评测实证:模型幻觉名 74 例全部是 "Bash") ----------


def test_resolve_tool_name_maps_bash_alias() -> None:
    from nexusagent.tools import resolve_tool_name

    assert resolve_tool_name("Bash") == "BashTool"
    assert resolve_tool_name("BashTool") == "BashTool"
    assert resolve_tool_name("NoSuchTool") == "NoSuchTool"
    assert resolve_tool_name("") == ""


def test_code_agent_executes_bash_alias_with_note(tmp_path: Path) -> None:
    from nexusagent.agents.code_agent import execute_code_agent_tool

    runtime = RuntimeState(workspace=tmp_path)
    message, _todos = execute_code_agent_tool(
        runtime,
        [],
        {"name": "Bash", "args": {"command": "echo alias-ok"}, "id": "tc1"},
    )

    payload = json.loads(message.content)
    assert payload["ok"] is True
    assert "alias-ok" in payload["stdout"]
    assert "resolved to 'BashTool'" in payload["note"]


def test_read_only_executor_resolves_bash_alias(tmp_path: Path) -> None:
    from nexusagent.graph.nodes import _execute_read_only_tool

    state = {"runtime": RuntimeState(workspace=tmp_path)}
    message = _execute_read_only_tool(state, {"name": "Bash", "args": {"command": "echo verifier-ok"}, "id": "tc2"})

    payload = json.loads(message.content)
    assert payload["ok"] is True
    assert "verifier-ok" in payload["stdout"]


def test_planner_executor_does_not_fabricate_tools_via_alias(tmp_path: Path) -> None:
    """planner 工具集里没有 BashTool,别名只做真名归一,不该让不存在的工具凭空可用。"""
    from nexusagent.graph.nodes import _execute_planner_tool

    state = {"runtime": RuntimeState(workspace=tmp_path), "task": "demo"}
    message = _execute_planner_tool(state, lambda event: None, {"name": "Bash", "args": {"command": "echo hi"}})

    payload = json.loads(message.content)
    assert payload["ok"] is False
    assert "unknown tool: Bash" in payload["error"]


def test_planner_executor_runs_its_own_registered_tool(tmp_path: Path) -> None:
    from nexusagent.graph.nodes import _execute_planner_tool

    state = {"runtime": RuntimeState(workspace=tmp_path), "task": "demo"}
    message = _execute_planner_tool(
        state,
        lambda event: None,
        {
            "name": "TodoWriteTool",
            "args": {
                "todos": ["write page"],
                "acceptance_criteria": ["page exists"],
                "verification_commands": ["python -c \"print(1)\""],
            },
            "id": "tc4",
        },
    )

    assert json.loads(message.content)["ok"] is True
    assert (tmp_path / "TODO.md").exists()


def test_unknown_tool_name_still_errors(tmp_path: Path) -> None:
    from nexusagent.agents.code_agent import execute_code_agent_tool

    runtime = RuntimeState(workspace=tmp_path)
    message, _todos = execute_code_agent_tool(
        runtime,
        [],
        {"name": "DefinitelyNotATool", "args": {}, "id": "tc3"},
    )

    payload = json.loads(message.content)
    assert payload["ok"] is False
    assert "unknown tool: DefinitelyNotATool" in payload["error"]
