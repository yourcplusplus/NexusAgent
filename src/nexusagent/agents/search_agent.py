from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from nexusagent.core.state import RuntimeState
from nexusagent.graph.memory import render_critical_context
from nexusagent.graph.state import NexusGraphState
from nexusagent.prompts.stage3 import SEARCH_AGENT_PROMPT
from nexusagent.providers.openai_provider import create_model
from nexusagent.tools.output_sink import spill_tool_output
from nexusagent.tools.registry import resolve_tool_name
from nexusagent.tools.web_search_tool import build_web_search_tool


Writer = Callable[[dict[str, Any]], None]


def run_search_agent(
    state: NexusGraphState,
    instruction: str,
    *,
    writer: Writer | None = None,
    max_loops: int = 4,
) -> dict[str, Any]:
    writer = writer or (lambda _: None)
    model = create_model()
    search_agent = model.bind_tools([build_web_search_tool()])
    messages = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(
            content=(
                f"{render_critical_context(state)}\n\n"
                f"Task: {state['task']}\n\n"
                f"Planner instruction:\n{instruction}\n\n"
                f"Existing research notes:\n{state.get('research_notes', '')}\n\n"
                "Search as needed and finish with a concise research summary plus source URLs."
            )
        ),
    ]

    produced_messages: list[Any] = []
    queries: list[str] = []
    sources: list[dict[str, Any]] = []
    answers: list[str] = []
    tool_events: list[dict[str, Any]] = []

    for _ in range(max_loops):
        response = search_agent.invoke(messages)
        produced_messages.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            args = call.get("args") or {}
            query = str(args.get("query", ""))
            if query:
                queries.append(query)
            writer({"type": "tool_call", "node": "searchAgent", "name": call.get("name"), "args": args})
            tool_result, search_result = _execute_search_tool(state["runtime"], call)
            event = _tool_result_event(tool_result)
            tool_events.append(event)
            writer(event)
            # 记账用原始结果:ToolMessage 内容超长时已被替换为落盘摘要
            parsed = search_result
            if isinstance(parsed, dict):
                if parsed.get("answer"):
                    answers.append(str(parsed["answer"]))
                for item in parsed.get("results", []) or []:
                    if isinstance(item, dict):
                        sources.append(item)
                writer(
                    {
                        "type": "search_results",
                        "query": parsed.get("query", query),
                        "answer": parsed.get("answer", ""),
                        "sources": parsed.get("results", []),
                    }
                )
            produced_messages.append(tool_result)
            messages.append(tool_result)

    summary = _last_ai_content(produced_messages) or "\n".join(answers)
    result = {
        "ok": True,
        "summary": summary,
        "queries": queries,
        "sources": _dedupe_sources(sources),
        "messages": produced_messages,
        "tool_events": tool_events,
    }
    writer(
        {
            "type": "search_summary",
            "summary": result["summary"],
            "queries": result["queries"],
            "sources": result["sources"],
        }
    )
    return result


def _execute_search_tool(runtime: RuntimeState, call: dict[str, Any]) -> tuple[ToolMessage, dict[str, Any]]:
    tool = build_web_search_tool()
    name = call.get("name", "")
    args = call.get("args") or {}
    if resolve_tool_name(name) != tool.name:
        result = {"ok": False, "error": f"unknown tool: {name}"}
    else:
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    spilled = spill_tool_output(runtime, name, result)
    return (
        ToolMessage(
            content=json.dumps(spilled, ensure_ascii=False),
            name=name,
            tool_call_id=call.get("id") or f"{name}-call",
        ),
        result,
    )


def _tool_result_event(tool_message: ToolMessage) -> dict[str, Any]:
    parsed = _parse_tool_content(tool_message.content)
    return {"type": "tool_result", "node": "searchAgent", "name": tool_message.name, "result": parsed}


def _parse_tool_content(content: Any) -> Any:
    try:
        return json.loads(str(content))
    except json.JSONDecodeError:
        return content


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


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""
