from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from nexusagent.core.state import RuntimeState
from nexusagent.graph.memory import build_layered_memory, format_layered_memory_for_prompt, memory_event
from nexusagent.graph.state import NexusGraphState
from nexusagent.graph.status_bar import append_status_update, env_seq, render_env_status, with_fresh_env
from nexusagent.prompts.stage3 import CODE_AGENT_PROMPT
from nexusagent.providers.openai_provider import create_model
from nexusagent.tools import build_tools, note_alias, resolve_tool_name
from nexusagent.tools.output_sink import spill_tool_output
from nexusagent.tools.todo_tool import persist_todos, update_todo


Writer = Callable[[dict[str, Any]], None]


def run_code_agent(
    state: NexusGraphState,
    instruction: str,
    *,
    writer: Writer | None = None,
    max_loops: int = 10,
) -> dict[str, Any]:
    runtime = state["runtime"]
    todos = [dict(todo) for todo in state.get("todos", [])]
    writer = writer or (lambda _: None)
    # 本 agent 由图节点内部调用而非图节点,所以自己保证环境快照存在且新鲜
    env_state = with_fresh_env({**state})
    memory = build_layered_memory({**env_state, "todos": todos}, node="codeAgent")
    writer(memory_event(memory, node="codeAgent"))
    model = create_model()
    code_agent = model.bind_tools(build_tools(runtime) + [_build_todo_update_tool(todos)])

    writer(
        {
            "type": "plan_snapshot",
            "node": "codeAgent",
            "plan_summary": state.get("plan_summary", ""),
            "todos": todos,
            "verification_commands": state.get("verification_commands", []),
        }
    )

    messages = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=_code_agent_input(env_state, instruction, memory)),
    ]
    produced_messages: list[Any] = []
    tool_events: list[dict[str, Any]] = []
    last_status = env_seq(env_state)

    for _ in range(max_loops):
        response = code_agent.invoke(messages)
        produced_messages.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            writer(
                {
                    "type": "tool_call",
                    "node": "codeAgent",
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                }
            )
            tool_result, todos = execute_code_agent_tool(runtime, todos, call)
            event = tool_result_event(tool_result, node="codeAgent")
            tool_events.append(event)
            writer(event)
            if call.get("name") == "TodoUpdateTool":
                persist_todos(
                    runtime,
                    todos,
                    state.get("acceptance_criteria", []),
                    state.get("verification_commands", []),
                    state.get("plan_summary", ""),
                )
                writer(
                    {
                        "type": "todo_update",
                        "node": "codeAgent",
                        "plan_summary": state.get("plan_summary", ""),
                        "todos": todos,
                        "verification_commands": state.get("verification_commands", []),
                    }
                )
            produced_messages.append(tool_result)
            messages.append(tool_result)
        # 工具跑完后按 TTL 检查:过期才追加一条 [STATUS ...] 进本地 messages
        last_status = append_status_update(
            messages,
            runtime=runtime,
            state=env_state,
            last_seq=last_status,
        )
    else:
        produced_messages.append(
            AIMessage(content="codeAgent stopped after the maximum tool loop count; verifier will inspect current files.")
        )

    summary = _last_ai_content(produced_messages)
    return {
        "ok": True,
        "summary": summary,
        "todos": todos or state.get("todos", []),
        "messages": produced_messages,
        "tool_events": tool_events,
    }


def execute_code_agent_tool(runtime: RuntimeState, todos: list[dict[str, str]], call: dict[str, Any]):
    name = call.get("name", "")
    args = call.get("args") or {}
    if name == "TodoUpdateTool":
        result = update_todo(todos, args.get("todo_id", ""), args.get("status", ""), args.get("note", ""))
        if result.get("ok"):
            todos = result["todos"]
    else:
        tools = {tool.name: tool for tool in build_tools(runtime)}
        tool = tools.get(resolve_tool_name(name))
        if tool is None:
            result = {"ok": False, "error": f"unknown tool: {name}"}
        else:
            try:
                result = tool.invoke(args)
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        note_alias(result, name)
    spilled = spill_tool_output(runtime, name, result)
    tool_call_id = call.get("id") or f"{name}-call"
    return ToolMessage(content=json.dumps(spilled, ensure_ascii=False), name=name, tool_call_id=tool_call_id), todos


def tool_result_event(tool_message: ToolMessage, *, node: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(tool_message.content))
    except json.JSONDecodeError:
        parsed = tool_message.content
    return {"type": "tool_result", "node": node, "name": tool_message.name, "result": parsed}


def _build_todo_update_tool(todos: list[dict[str, str]]) -> StructuredTool:
    return StructuredTool.from_function(
        name="TodoUpdateTool",
        func=lambda todo_id, status, note="": update_todo(todos, todo_id, status, note),
        description="Update one existing todo status. Args: todo_id, status, optional note.",
    )


def _code_agent_input(state: NexusGraphState, instruction: str, memory: dict[str, Any]) -> str:
    parts = [
        memory.get("critical_context", ""),
        f"Task: {state['task']}",
        f"Planner instruction:\n{instruction}",
    ]
    if state.get("session_context"):
        parts.append("Session context for this multi-turn coding session:\n" + str(state.get("session_context", "")))
    parts.append("Layered memory snapshot:\n" + format_layered_memory_for_prompt(memory))
    parts.append(render_env_status(state.get("env_status")))
    return "\n\n".join(part for part in parts if part)


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""
