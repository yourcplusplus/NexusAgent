from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer

from nexusagent.agents.code_agent import run_code_agent
from nexusagent.agents.search_agent import run_search_agent
from nexusagent.graph.config import (
    get_context_keep_groups,
    get_context_token_limit,
    get_context_window_ratio,
)
from nexusagent.graph.context_window import (
    DEFAULT_CHARS_PER_TOKEN,
    EvictionPlan,
    estimate_payload_tokens,
    estimate_window_tokens,
    plan_eviction,
)
from nexusagent.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    merge_narrative,
    persist_history_summary,
    rollup_evicted_messages,
)
from nexusagent.graph.state import NexusGraphState, TodoItem, VerificationCheck
from nexusagent.graph.status_bar import (
    append_status_update,
    ensure_env_status,
    env_seq,
    render_env_status,
    with_fresh_env,
)
from nexusagent.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from nexusagent.providers.openai_provider import create_model
from nexusagent.tools import build_read_only_tools, note_alias, resolve_tool_name
from nexusagent.tools.output_sink import spill_tool_output
from nexusagent.tools.todo_tool import persist_todos, write_todos


CONTEXT_PRESSURE_WARN_RATIO = 0.7
AGENT_STATUS_BAR_NODE = "agent_status_bar"
WHITELIST_BOARDS = ("[Goal]", "[Constraints]", "[TODOs]", "[Files]", "[Git]", "[Artifacts]")
DEFAULT_TODOS = [
    "Clarify the deliverable and acceptance criteria.",
    "Delegate specialist work needed for the task.",
    "Verify the generated result.",
]

INTENT_ROUTER_PROMPT = """You are the intent router for NexusAgent.

Classify the user's latest input into exactly one route:

- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A, or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands, installing packages, searching the web, checking the current project, verifying a result, or producing a concrete deliverable.

When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.

Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}

If uncertain, choose workflow.
"""

CHAT_RESPONDER_PROMPT = """You are NexusAgent's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""


def intent_router_node(state: NexusGraphState) -> dict[str, Any]:
    writer = _get_writer()
    route = "workflow"
    reason = "router fallback: default to workflow"
    confidence = 0.0
    try:
        response = create_model().invoke(
            [
                SystemMessage(content=INTENT_ROUTER_PROMPT),
                HumanMessage(content=_router_input(state)),
            ]
        )
        parsed = _extract_json(str(response.content)) or {}
        candidate = str(parsed.get("route", "")).strip().lower()
        parsed_confidence = _coerce_confidence(parsed.get("confidence"))
        if candidate in {"chat", "workflow"} and parsed_confidence >= 0.55:
            route = candidate
            confidence = parsed_confidence
            reason = str(parsed.get("reason") or "")
        else:
            reason = str(parsed.get("reason") or "router returned low-confidence or invalid route")
            confidence = parsed_confidence
    except Exception as exc:
        reason = f"router error: {type(exc).__name__}: {exc}"

    event = {
        "type": "intent_decision",
        "route": route,
        "reason": reason,
        "confidence": confidence,
    }
    writer(event)
    return {
        "intent_route": route,
        "intent_reason": reason,
        "intent_confidence": confidence,
    }


def intent_route_fn(state: NexusGraphState) -> str:
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"


def chat_responder_node(state: NexusGraphState) -> dict[str, Any]:
    writer = _get_writer()
    try:
        response = create_model().invoke(
            [
                SystemMessage(content=CHAT_RESPONDER_PROMPT),
                HumanMessage(content=_chat_input(state)),
            ]
        )
        text = str(getattr(response, "content", "") or "").strip()
    except Exception as exc:
        text = f"这是轻量聊天分支，但模型回复暂不可用：{type(exc).__name__}: {exc}"
    if not text:
        text = "我在。你可以继续提问，或者直接描述一个需要我完成的任务。"
    event = {
        "type": "chat_response",
        "mode": "lightweight",
        "reason": state.get("intent_reason", ""),
        "response": text,
    }
    writer(event)
    return {"chat_response": text, "final_answer": text}


def planner_node(state: NexusGraphState) -> dict[str, Any]:
    writer = _get_writer()
    working_state: NexusGraphState = {**state}
    if not working_state.get("todos"):
        _apply_plan(working_state, _default_plan())
        persist_todos(
            working_state["runtime"],
            working_state.get("todos", []),
            working_state.get("acceptance_criteria", []),
            working_state.get("verification_commands", []),
            working_state.get("plan_summary", ""),
        )

    # 首个 planner 先于 context_monitor 运行,拓扑上还没有经过 agent_status_bar,
    # 所以这里自己保证快照存在(TTL 内为空操作,不会重复起子进程)。
    working_state = with_fresh_env(working_state)
    memory = build_layered_memory(working_state, node="planner")
    writer(memory_event(memory, node="planner"))
    model = create_model()
    planner = model.bind_tools(_build_planner_tools(working_state, writer))
    messages: list[Any] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working_state, memory)),
    ]
    produced_messages: list[Any] = []
    last_status = env_seq(working_state)

    writer(
        {
            "type": "plan_snapshot",
            "node": "planner",
            "plan_summary": working_state.get("plan_summary", ""),
            "todos": working_state.get("todos", []),
            "verification_commands": working_state.get("verification_commands", []),
            "attempts": working_state.get("attempts", 0),
        }
    )

    for _ in range(8):
        response = planner.invoke(messages)
        produced_messages.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            tool_message = _execute_planner_tool(working_state, writer, call)
            produced_messages.append(tool_message)
            messages.append(tool_message)
        # 工具跑完后按 TTL 检查:过期才追加一条 [STATUS ...] 进本地 messages
        last_status = append_status_update(
            messages,
            runtime=working_state["runtime"],
            state=working_state,
            last_seq=last_status,
        )
    else:
        produced_messages.append(AIMessage(content="planner stopped after the maximum supervisor tool loop count."))

    metadata = dict(working_state.get("metadata", {}))
    metadata["planner_raw"] = _last_ai_content(produced_messages)
    _stamp_message_ids(produced_messages, node="planner", attempt=working_state.get("attempts", 0) + 1)
    final_memory = build_layered_memory(working_state, node="planner")
    return {
        "plan_summary": working_state.get("plan_summary", ""),
        "todos": working_state.get("todos", []),
        "acceptance_criteria": working_state.get("acceptance_criteria", []),
        "verification_commands": working_state.get("verification_commands", []),
        "research_notes": working_state.get("research_notes", ""),
        "sources": working_state.get("sources", []),
        "agent_handoffs": working_state.get("agent_handoffs", []),
        "code_agent_summary": working_state.get("code_agent_summary", ""),
        "last_actor_summary": working_state.get("code_agent_summary", ""),
        "messages": produced_messages,
        "memory_snapshot": final_memory,
        "history_summary": final_memory.get("history_summary_store", {}).get("history_summary", ""),
        "metadata": metadata,
        "env_status": working_state.get("env_status", {}),
        "context_next_node": "verifier",
    }


def verifier_node(state: NexusGraphState) -> dict[str, Any]:
    writer = _get_writer()
    working_state = with_fresh_env(state)
    memory = build_layered_memory(working_state, node="verifier")
    writer(memory_event(memory, node="verifier"))
    writer(
        {
            "type": "plan_snapshot",
            "node": "verifier",
            "plan_summary": state.get("plan_summary", ""),
            "todos": state.get("todos", []),
            "verification_commands": state.get("verification_commands", []),
        }
    )

    model = create_model()
    verifier = model.bind_tools(build_read_only_tools(state["runtime"]))
    messages: list[Any] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=_verifier_input(working_state, memory)),
    ]
    produced_messages: list[Any] = []
    tool_events: list[dict[str, Any]] = []
    last_status = env_seq(working_state)

    for _ in range(8):
        response = verifier.invoke(messages)
        produced_messages.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            writer({"type": "tool_call", "node": "verifier", "name": call.get("name"), "args": call.get("args", {})})
            tool_message = _execute_read_only_tool(state, call)
            event = _tool_result_event(tool_message, node="verifier")
            tool_events.append(event)
            writer(event)
            produced_messages.append(tool_message)
            messages.append(tool_message)
        last_status = append_status_update(
            messages,
            runtime=working_state["runtime"],
            state=working_state,
            last_seq=last_status,
        )
    else:
        produced_messages.append(
            AIMessage(
                content=json.dumps(
                    {
                        "passed": False,
                        "reason": "Verifier stopped after the maximum tool loop count.",
                        "checks": [],
                        "recommended_next_instruction": "Inspect the workspace and complete the unfinished task.",
                    },
                    ensure_ascii=False,
                )
            )
        )

    parsed = _extract_json(_last_ai_content(produced_messages)) or {
        "passed": False,
        "reason": "Verifier did not return valid JSON.",
        "checks": [
            {
                "name": "verifier_json",
                "passed": False,
                "detail": _last_ai_content(produced_messages)[:800],
            }
        ],
        "recommended_next_instruction": "Return valid verifier JSON after inspecting the result.",
    }
    checks = _normalize_checks(parsed.get("checks"))
    passed = bool(parsed.get("passed"))
    reason = str(parsed.get("reason") or "")
    recommended = str(parsed.get("recommended_next_instruction") or "")
    attempts = state.get("attempts", 0) + 1
    todos = [dict(todo) for todo in state.get("todos", [])]
    if passed:
        todos = [
            {
                **todo,
                "status": "completed" if todo.get("status") != "blocked" else todo.get("status", "blocked"),
                "note": todo.get("note") or "verified",
            }
            for todo in todos
        ]
        writer(
            {
                "type": "todo_update",
                "node": "verifier",
                "plan_summary": state.get("plan_summary", ""),
                "todos": todos,
                "verification_commands": state.get("verification_commands", []),
            }
        )
    last_error = "" if passed else _format_verifier_error(reason, recommended, tool_events)
    _stamp_message_ids(produced_messages, node="verifier", attempt=attempts)

    return {
        "messages": produced_messages,
        "verification_results": _tool_events_to_verification_results(tool_events),
        "verification_checks": checks,
        "verifier_summary": reason,
        "passed": passed,
        "attempts": attempts,
        "last_error": last_error,
        "todos": todos,
        "memory_snapshot": memory,
        "history_summary": memory.get("history_summary_store", {}).get("history_summary", ""),
        "env_status": working_state.get("env_status", {}),
        "context_next_node": verifier_route({**state, "passed": passed, "attempts": attempts}),
    }


def context_monitor_node(state: NexusGraphState) -> dict[str, Any]:
    writer = _get_writer()
    token_limit = get_context_token_limit()
    token_count = estimate_context_tokens(state)
    should_compress = token_count >= token_limit
    pressure = token_count / token_limit if token_limit > 0 else 0.0
    next_node = state.get("context_next_node") or "verifier"
    # 回落到预警线以下即重新武装:压缩后压力自然下降,压缩器无需显式重置
    notified = bool(state.get("context_pressure_notified")) and pressure >= CONTEXT_PRESSURE_WARN_RATIO
    writer(
        {
            "type": "context_monitor",
            "token_count": token_count,
            "token_limit": token_limit,
            "pressure": round(pressure, 3),
            "should_compress": should_compress,
            "next_node": next_node,
            "message_count": len(state.get("messages", [])),
        }
    )
    if pressure >= CONTEXT_PRESSURE_WARN_RATIO and not notified:
        notified = True
        writer(
            {
                "type": "context_pressure",
                "token_count": token_count,
                "token_limit": token_limit,
                "pressure": round(pressure, 3),
                "warn_ratio": CONTEXT_PRESSURE_WARN_RATIO,
                "message_count": len(state.get("messages", [])),
            }
        )
    return {
        "context_token_count": token_count,
        "context_token_limit": token_limit,
        "context_should_compress": should_compress,
        "context_next_node": next_node,
        "context_pressure_notified": notified,
    }


def context_monitor_route(state: NexusGraphState) -> str:
    if state.get("context_should_compress"):
        return "context_compressor"
    return AGENT_STATUS_BAR_NODE


def agent_status_bar_node(state: NexusGraphState) -> dict[str, Any]:
    """环境状态栏:进入 LLM 节点前刷新快照,并据 context_next_node 决定去向。

    不强制重采:TTL 内复用缓存,于是「每次节点转移都刷新一次」与「TTL 内 git 子进程
    ≤ 1 次」同时成立。快照经 ``env_status`` 进 state,白名单 ``[Git]`` 板块与 prompt
    的 Environment status 小节都从它渲染。

    路由职责承接自旧 ``context_compressor_route``:压缩与不压缩两条路在这里汇合,
    目标由 ``context_next_node`` 决定。
    """
    writer = _get_writer()
    env = ensure_env_status(state["runtime"], state)
    writer({"type": "status_bar", **env})
    return {"env_status": env}


def agent_status_bar_route(state: NexusGraphState) -> str:
    return state.get("context_next_node") or "verifier"


def context_compressor_node(state: NexusGraphState) -> dict[str, Any]:
    """滑动窗口压缩:逐条逐出最旧的执行组,把被逐出的片段滚成增量叙事。

    不做全量清空,也不把摘要消息塞回转录:叙事经 history_summary 进入各节点的
    prompt,白名单块由 render_critical_context 从实时 state 独立重建,两者都不
    经过这条压缩路径。
    """
    writer = _get_writer()
    memory = build_layered_memory(state, node="context_compressor")
    writer(memory_event(memory, node="context_compressor"))

    messages = list(state.get("messages", []))
    before_tokens = int(state.get("context_token_count") or estimate_context_tokens(state))
    plan = plan_eviction(
        messages,
        keep_groups=get_context_keep_groups(),
        token_limit=get_context_token_limit(),
        window_ratio=get_context_window_ratio(),
    )

    if plan.skipped_reason:
        writer({"type": "context_compression", **_compression_event(state, plan, before_tokens, before_tokens)})
        return {}

    evicted_ids = set(plan.evict_ids)
    evicted = [message for message in messages if str(getattr(message, "id", "")) in evicted_ids]
    kept = [message for message in messages if str(getattr(message, "id", "")) not in evicted_ids]

    narrative_before = state.get("history_summary") or state.get("context_summary") or ""
    entry, used_fallback = rollup_evicted_messages(evicted, current_narrative=narrative_before)
    narrative = merge_narrative(narrative_before, entry)
    persist_history_summary(state["runtime"], narrative)

    post_memory = build_layered_memory(
        {**state, "history_summary": narrative, "context_summary": narrative},
        node="context_compressor",
    )
    post_state: NexusGraphState = {
        **state,
        "messages": kept,
        "history_summary": narrative,
        "context_summary": narrative,
        "memory_snapshot": post_memory,
    }
    after_tokens = estimate_context_tokens(post_state)

    event = _compression_event(state, plan, before_tokens, after_tokens)
    event["before_tokens_exact"] = count_tokens_exact(state)
    event["after_tokens_exact"] = count_tokens_exact(post_state)
    event["fallback"] = used_fallback
    event["narrative_chars"] = len(narrative)
    # 逐出【之后】重建的白名单自检:让「压缩不丢关键信息」成为事件流里可核验的事实,
    # 而不是只能靠"渲染本来就是代码重建的"这条原理去推断(计划里的 whitelist_digest)。
    event["whitelist_digest"] = _whitelist_digest(post_memory)
    event["summary"] = _short_text(entry or "(no narrative entry)", 1200)
    writer({"type": "context_compression", **event})
    return {
        "messages": [RemoveMessage(id=message_id) for message_id in plan.evict_ids],
        "context_summary": narrative,
        "history_summary": narrative,
        "memory_snapshot": post_memory,
        "compression_events": list(state.get("compression_events", [])) + [event],
        "context_token_count": after_tokens,
        "context_should_compress": False,
    }


def _compression_event(
    state: NexusGraphState,
    plan: EvictionPlan,
    before_tokens: int,
    after_tokens: int,
) -> dict[str, Any]:
    """压缩事件的标准字段;跳过压缩时前后 token 相同、逐出统计为零。"""
    event: dict[str, Any] = {
        "before_tokens": int(before_tokens),
        "after_tokens": int(after_tokens),
        "before_tokens_exact": None,
        "after_tokens_exact": None,
        "evicted_messages": plan.messages_evicted,
        "kept_messages": plan.messages_kept,
        "groups_evicted": plan.groups_evicted,
        "skipped_reason": plan.skipped_reason,
        "notes": plan.notes,
        "fallback": False,
        "narrative_chars": 0,
        "next_node": state.get("context_next_node", "verifier"),
        # 保留旧字段名,兼容既有渲染器与最终报告:
        "removed_messages": plan.messages_evicted,
        "summary": "",
    }
    if plan.skipped_reason:
        event["summary"] = f"(compression skipped: {plan.skipped_reason})"
    return event


def verifier_route(state: NexusGraphState) -> str:
    if state.get("passed"):
        return "final"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "final"
    return "planner"


def final_node(state: NexusGraphState) -> dict[str, Any]:
    status = "PASSED" if state.get("passed") else "FAILED"
    checks = "\n".join(
        f"- {check.get('name', 'check')}: {'PASS' if check.get('passed') else 'FAIL'} - {check.get('detail', '')}"
        for check in state.get("verification_checks", [])
    )
    todos = "\n".join(f"- [{todo.get('status', '')}] {todo.get('content', '')}" for todo in state.get("todos", []))
    sources = "\n".join(f"- {source.get('title', '')}: {source.get('url', '')}" for source in state.get("sources", []))
    compression_events = state.get("compression_events", [])
    compression_text = "(none)"
    if compression_events:
        latest = compression_events[-1]
        compression_text = (
            f"{len(compression_events)} compression(s); "
            f"latest {latest.get('before_tokens')} -> {latest.get('after_tokens')} tokens; "
            f"removed {latest.get('removed_messages')} message(s)"
        )
    final_answer = (
        f"LangGraph MultiAgent workflow finished: {status}\n\n"
        f"Plan: {state.get('plan_summary', '')}\n\n"
        f"Todos:\n{todos}\n\n"
        f"Research sources:\n{sources or '(none)'}\n\n"
        f"Verifier:\n{state.get('verifier_summary', '')}\n\n"
        f"Checks:\n{checks or '(none)'}\n\n"
        f"Context compression:\n{compression_text}\n\n"
        f"CodeAgent summary:\n{state.get('code_agent_summary') or state.get('last_actor_summary', '')}"
    )
    return {"final_answer": final_answer}


def estimate_context_tokens(state: NexusGraphState) -> int:
    """快速估算"窗口 + 记忆载荷"规模,供 monitor 做阈值判断。

    刻意不用 tokenizer、也不重建分层记忆:monitor 每个图循环都会调用一次,而旧
    实现每次都要 tokenize 全量消息、并重读 NOTEPAD + HISTORY_SUMMARY 两块磁盘
    文件。这里只做字符量除法,对阈值判断而言精度足够。口径与状态栏共用。
    """
    window = estimate_window_tokens(list(state.get("messages", [])))
    return max(1, window + estimate_payload_tokens(state.get("memory_snapshot") or {}))


def count_tokens_exact(state: NexusGraphState) -> int:
    """tokenizer 精确计数:只给压缩事件的 before/after 报告用,不进 monitor 热路径。

    压缩是稀有事件,这里的成本可以接受;更重要的是它同时用于压缩前后,
    两个数字同源可比。
    """
    messages = list(state.get("messages", []))
    payload = build_layered_memory(state, node="context_compressor")
    payload_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str))
    try:
        model = create_model()
        return int(model.get_num_tokens_from_messages(messages + [payload_message]))
    except Exception:
        text = "\n".join(_message_text(message) for message in messages)
        return max(1, (len(text) + len(payload_message.content)) // DEFAULT_CHARS_PER_TOKEN)


def _whitelist_digest(memory: dict[str, Any]) -> dict[str, Any]:
    """压缩后白名单块的自检摘要:六个板块是否齐全 + 块字符数。

    白名单由 render_critical_context 从实时 state 重建,原理上不经过压缩路径;
    但「原理上不会丢」需要一个可核验的证据,故把逐出之后重新渲染的结果做成事件
    字段——评测与事故复盘据此判断,不必依赖"压缩之后恰好还有节点跑过"这种运气。
    """
    block = str(memory.get("critical_context", ""))
    return {
        "present": [board for board in WHITELIST_BOARDS if board in block],
        "missing": [board for board in WHITELIST_BOARDS if board not in block],
        "chars": len(block),
    }


def _build_planner_tools(state: NexusGraphState, writer) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="TodoWriteTool",
            func=lambda todos, acceptance_criteria, verification_commands, plan_summary="": _todo_write_tool(
                state, writer, todos, acceptance_criteria, verification_commands, plan_summary
            ),
            description=(
                "Publish or revise plan state. Args: todos, acceptance_criteria, "
                "verification_commands, optional plan_summary."
            ),
        ),
        StructuredTool.from_function(
            name="CallSearchAgentTool",
            func=lambda instruction: _call_search_agent_tool(state, writer, instruction),
            description="Delegate research work to searchAgent. Args: instruction.",
        ),
        StructuredTool.from_function(
            name="CallCodeAgentTool",
            func=lambda instruction: _call_code_agent_tool(state, writer, instruction),
            description="Delegate implementation work to codeAgent. Args: instruction.",
        ),
    ]


def _todo_write_tool(
    state: NexusGraphState,
    writer,
    todos: Any,
    acceptance_criteria: Any,
    verification_commands: Any,
    plan_summary: str = "",
) -> dict[str, Any]:
    result = write_todos(todos, acceptance_criteria, verification_commands)
    if result.get("ok"):
        state["plan_summary"] = plan_summary or state.get("plan_summary") or "MultiAgent plan"
        state["todos"] = _todo_items(result["todos"], existing=state.get("todos", []))
        state["acceptance_criteria"] = result["acceptance_criteria"]
        state["verification_commands"] = result["verification_commands"]
        persist_todos(
            state["runtime"],
            state["todos"],
            state["acceptance_criteria"],
            state["verification_commands"],
            state.get("plan_summary", ""),
        )
        writer(
            {
                "type": "plan_snapshot",
                "node": "planner",
                "plan_summary": state.get("plan_summary", ""),
                "todos": state.get("todos", []),
                "verification_commands": state.get("verification_commands", []),
                "acceptance_criteria": state.get("acceptance_criteria", []),
            }
        )
    return {
        **result,
        "plan_summary": state.get("plan_summary", ""),
        "todo_items": state.get("todos", []),
    }


def _call_search_agent_tool(state: NexusGraphState, writer, instruction: str) -> dict[str, Any]:
    writer({"type": "handoff", "from": "planner", "to": "searchAgent", "instruction": instruction})
    result = run_search_agent(state, instruction, writer=writer)
    existing_sources = list(state.get("sources", []))
    state["research_notes"] = _join_notes(state.get("research_notes", ""), result.get("summary", ""))
    state["sources"] = _dedupe_sources(existing_sources + list(result.get("sources", [])))
    handoff = {
        "from_agent": "planner",
        "to_agent": "searchAgent",
        "instruction": instruction,
        "result": result.get("summary", ""),
    }
    state["agent_handoffs"] = list(state.get("agent_handoffs", [])) + [handoff]
    writer({"type": "handoff_result", "from": "searchAgent", "to": "planner", "result": result.get("summary", "")})
    return {
        "ok": True,
        "summary": result.get("summary", ""),
        "sources": state.get("sources", []),
        "queries": result.get("queries", []),
    }


def _call_code_agent_tool(state: NexusGraphState, writer, instruction: str) -> dict[str, Any]:
    writer({"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": instruction})
    result = run_code_agent(state, instruction, writer=writer)
    state["todos"] = result.get("todos", state.get("todos", []))
    state["code_agent_summary"] = result.get("summary", "")
    state["last_actor_summary"] = result.get("summary", "")
    handoff = {
        "from_agent": "planner",
        "to_agent": "codeAgent",
        "instruction": instruction,
        "result": result.get("summary", ""),
    }
    state["agent_handoffs"] = list(state.get("agent_handoffs", [])) + [handoff]
    writer({"type": "handoff_result", "from": "codeAgent", "to": "planner", "result": result.get("summary", "")})
    return {"ok": True, "summary": result.get("summary", ""), "todos": state.get("todos", [])}


def _execute_planner_tool(state: NexusGraphState, writer, call: dict[str, Any]) -> ToolMessage:
    name = call.get("name", "")
    args = call.get("args") or {}
    writer({"type": "tool_call", "node": "planner", "name": name, "args": args})
    tools = {tool.name: tool for tool in _build_planner_tools(state, writer)}
    tool = tools.get(resolve_tool_name(name))
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    note_alias(result, name)
    spilled = spill_tool_output(state["runtime"], name, result)
    tool_message = ToolMessage(
        content=json.dumps(spilled, ensure_ascii=False),
        name=name,
        tool_call_id=call.get("id") or f"{name}-call",
    )
    writer(_tool_result_event(tool_message, node="planner"))
    return tool_message


def _execute_read_only_tool(state: NexusGraphState, call: dict[str, Any]) -> ToolMessage:
    name = call.get("name", "")
    args = call.get("args") or {}
    tools = {tool.name: tool for tool in build_read_only_tools(state["runtime"])}
    tool = tools.get(resolve_tool_name(name))
    if tool is None:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    note_alias(result, name)
    spilled = spill_tool_output(state["runtime"], name, result)
    return ToolMessage(
        content=json.dumps(spilled, ensure_ascii=False),
        name=name,
        tool_call_id=call.get("id") or f"{name}-call",
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _planner_input(state: NexusGraphState, memory: dict[str, Any]) -> str:
    parts = [
        memory.get("critical_context", ""),
        f"Task: {state['task']}",
        f"Attempt: {state.get('attempts', 0) + 1}",
    ]
    if state.get("session_context"):
        parts.append("Session context for this multi-turn coding session:\n" + str(state.get("session_context", "")))
    parts.append("Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory))
    parts.append(render_env_status(state.get("env_status")))
    return "\n\n".join(part for part in parts if part)


def _verifier_input(state: NexusGraphState, memory: dict[str, Any]) -> str:
    parts = [
        memory.get("critical_context", ""),
        f"Task: {state['task']}",
    ]
    if state.get("session_context"):
        parts.append("Session context for this multi-turn coding session:\n" + str(state.get("session_context", "")))
    parts.append("Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory))
    parts.append("Inspect the workspace with tools and return only verifier JSON.")
    parts.append(render_env_status(state.get("env_status")))
    return "\n\n".join(part for part in parts if part)


def _router_input(state: NexusGraphState) -> str:
    parts = [f"User input:\n{state.get('task', '')}"]
    if state.get("session_context"):
        parts.append("Session context:\n" + str(state.get("session_context", "")))
    return "\n\n".join(parts)


def _chat_input(state: NexusGraphState) -> str:
    parts = [f"User input:\n{state.get('task', '')}"]
    if state.get("session_context"):
        parts.append("Session context:\n" + str(state.get("session_context", "")))
    return "\n\n".join(parts)


def _default_plan() -> dict[str, Any]:
    return {
        "plan_summary": "Coordinate specialist agents to complete and verify the requested deliverable.",
        "todos": DEFAULT_TODOS,
        "acceptance_criteria": ["The requested deliverable exists.", "The verifier model confirms completion."],
        "verification_commands": [],
    }


def _apply_plan(state: NexusGraphState, plan: dict[str, Any]) -> None:
    state["plan_summary"] = str(plan.get("plan_summary", ""))
    state["todos"] = _todo_items([str(item) for item in plan.get("todos", [])], existing=state.get("todos", []))
    state["acceptance_criteria"] = [str(item) for item in plan.get("acceptance_criteria", [])]
    state["verification_commands"] = _verification_commands_for_task(plan)


def _verification_commands_for_task(parsed: dict[str, Any]) -> list[str]:
    return [str(item) for item in parsed.get("verification_commands") or []]


def _todo_items(todos: list[str], *, existing: list[dict[str, Any]] | None = None) -> list[TodoItem]:
    existing_by_content = {todo.get("content", ""): todo for todo in existing or []}
    items: list[TodoItem] = []
    for idx, todo in enumerate(todos, start=1):
        previous = existing_by_content.get(todo, {})
        items.append(
            {
                "id": str(previous.get("id") or f"todo-{idx}"),
                "content": todo,
                "status": str(previous.get("status") or "pending"),
                "note": str(previous.get("note") or ""),
            }
        )
    return items


def _extract_json(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _tool_result_event(tool_message: ToolMessage, *, node: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        parsed = tool_message.content
    return {"type": "tool_result", "node": node, "name": tool_message.name, "result": parsed}


def _tool_events_to_verification_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for event in events:
        result = event.get("result", {})
        if not isinstance(result, dict):
            continue
        results.append(
            {
                "command": result.get("command") or event.get("name", ""),
                "ok": bool(result.get("ok")),
                "exit_code": result.get("exit_code"),
                "stdout": str(result.get("stdout", "")),
                "stderr": str(result.get("stderr") or result.get("error", "")),
            }
        )
    return results


def _normalize_checks(raw: Any) -> list[VerificationCheck]:
    if not isinstance(raw, list):
        return []
    checks: list[VerificationCheck] = []
    for item in raw:
        if isinstance(item, dict):
            checks.append(
                {
                    "name": str(item.get("name") or "check"),
                    "passed": bool(item.get("passed")),
                    "detail": str(item.get("detail") or ""),
                }
            )
    return checks


def _format_verifier_error(reason: str, recommended: str, tool_events: list[dict[str, Any]]) -> str:
    event_text = json.dumps(tool_events[-3:], ensure_ascii=False, default=str)[:1600]
    return (
        f"Verifier failed: {reason}\n"
        f"Recommended next instruction: {recommended}\n"
        f"Recent verifier tool events:\n{event_text}"
    )


def _join_notes(existing: str, new: str) -> str:
    if not existing:
        return new
    if not new:
        return existing
    return existing + "\n\n" + new


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for source in sources:
        url = str(source.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(source)
    return deduped


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _stamp_message_ids(messages: list[Any], *, node: str, attempt: int) -> None:
    """给本节点产出的消息盖上确定性 ID,供滑动窗口按前缀分组。

    reducer 只在 id 为 None 时分配随机 UUID,因此这里显式盖章的 ID 会被保留。
    无条件覆盖:模型响应可能自带服务端 id(如 chatcmpl-xxx),而它不参与配对
    ——工具调用的配对走 ToolMessage.tool_call_id,与本字段无关。
    """
    for index, message in enumerate(messages, start=1):
        message.id = f"{node}-a{attempt}-{index:04d}"


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""


def _todos_text(todos: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {todo.get('id', '')} [{todo.get('status', '')}] {todo.get('content', '')} {todo.get('note', '')}".strip()
        for todo in todos
    )


def _list_text(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _get_writer():
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
