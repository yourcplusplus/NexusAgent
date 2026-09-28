from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from nexusagent.core.state import MAX_ARTIFACTS, MAX_TOUCHED_FILES, RuntimeState
from nexusagent.graph.status_bar import render_git_board
from nexusagent.prompts.stage4 import CONTEXT_ROLLUP_PROMPT
from nexusagent.providers.openai_provider import create_model
from nexusagent.tools.file_tools import read_text_lossy
from nexusagent.tools.notepad_tool import NOTEPAD_FILE, read_notepad

HISTORY_SUMMARY_FILE = "HISTORY_SUMMARY.md"

CRITICAL_CONTEXT_HEADER = "=== CRITICAL CONTEXT (auto-pinned, never dropped) ==="
CRITICAL_CONTEXT_FOOTER = "=== END CRITICAL CONTEXT ==="

RULES_LAYER = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}

MAX_TEXT_CHARS = {
    "research_notes": 1600,
    "agent_handoff_instruction": 500,
    "agent_handoff_result": 700,
    "code_agent_summary": 1000,
    "verifier_summary": 1000,
    "last_error": 1400,
    "context_summary": 1600,
    "session_context": 1800,
    "notepad": 1800,
    "history_summary": 2200,
}

NARRATIVE_MAX_CHARS = MAX_TEXT_CHARS["history_summary"]  # 叙事段与渲染上限保持一致
FALLBACK_MAX_CHARS = 1500
FOLD_SNIPPET_CHARS = 100
FOLD_BUCKET_MAX_SNIPPETS = 12
SKELETON_TEXT_CHARS = 200
SKELETON_ERROR_CHARS = 120
ROLLUP_MESSAGE_CHARS = 2000


def build_layered_memory(state: dict[str, Any], *, node: str = "graph") -> dict[str, Any]:
    runtime = state["runtime"]
    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)
    sources = [
        {
            "title": source.get("title", ""),
            "url": source.get("url", ""),
        }
        for source in state.get("sources", [])
    ]
    working_memory = {
        "node": node,
        "task": state.get("task", ""),
        "session_id": state.get("session_id", ""),
        "session_turn": state.get("session_turn", 0),
        "session_context": _short_text(state.get("session_context", ""), MAX_TEXT_CHARS["session_context"]),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": _short_text(state.get("research_notes", ""), MAX_TEXT_CHARS["research_notes"]),
        "sources": sources,
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),
        "code_agent_summary": _short_text(state.get("code_agent_summary", ""), MAX_TEXT_CHARS["code_agent_summary"]),
        "verifier_summary": _short_text(state.get("verifier_summary", ""), MAX_TEXT_CHARS["verifier_summary"]),
        "verification_checks": state.get("verification_checks", []),
        "last_error": _short_text(state.get("last_error", ""), MAX_TEXT_CHARS["last_error"]),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
        "context_next_node": state.get("context_next_node", ""),
    }
    history_summary = state.get("history_summary") or history.get("content", "")
    history_summary_store = {
        "history_path": HISTORY_SUMMARY_FILE,
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history_summary, MAX_TEXT_CHARS["history_summary"]),
        "notepad_path": NOTEPAD_FILE,
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), MAX_TEXT_CHARS["notepad"]),
        "context_summary": _short_text(state.get("context_summary", ""), MAX_TEXT_CHARS["context_summary"]),
        "compression_events": state.get("compression_events", [])[-3:],
    }
    return {
        "rules": dict(RULES_LAYER),
        "critical_context": render_critical_context(state),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def format_layered_memory_for_prompt(memory: dict[str, Any]) -> str:
    """序列化分层记忆供节点 prompt 使用。

    critical_context 由调用方前置注入(最高注意力位置),故此处剔除,避免同一
    白名单块在一个 prompt 里出现两次。压缩器与 token 估算器直接内嵌 memory
    dict,不走本函数,因此仍能看到该层。
    """
    payload = {key: value for key, value in memory.items() if key != "critical_context"}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def render_critical_context(state: dict[str, Any]) -> str:
    """渲染关键信息白名单块:压缩时原样保留,不经 LLM 转述。

    Goal / Constraints / TODOs 每次从 graph state 重建(始终最新);Files /
    Artifacts 读 RuntimeState 的足迹登记,按最近触碰倒序;Git 段读
    ``state.env_status`` 的快照(agent_status_bar 在每次进入 LLM 节点前刷新),
    无快照时回落为 pending。

    本块绝不截断,体积靠数量上限控制。
    """
    return "\n".join(
        [
            CRITICAL_CONTEXT_HEADER,
            f"[Goal] {state.get('task', '') or '(none)'}",
            _critical_constraints(state),
            _critical_todos(state),
            _critical_files(state),
            render_git_board(state.get("env_status")),
            _critical_artifacts(state),
            CRITICAL_CONTEXT_FOOTER,
        ]
    )


def _critical_constraints(state: dict[str, Any]) -> str:
    criteria = [str(item) for item in state.get("acceptance_criteria", []) or []]
    if not criteria:
        return "[Constraints] (none)"
    return "\n".join(["[Constraints]", *(f"  - {item}" for item in criteria)])


def _critical_todos(state: dict[str, Any]) -> str:
    todos = list(state.get("todos", []) or [])
    if not todos:
        return "[TODOs] (none)"
    rows = []
    for todo in todos:
        note = str(todo.get("note", "") or "")
        note_text = f" -- {note}" if note else ""
        rows.append(f"  - {todo.get('id', '')} [{todo.get('status', '')}] {todo.get('content', '')}{note_text}")
    return "\n".join(["[TODOs]", *rows])


def _critical_files(state: dict[str, Any]) -> str:
    entries = _recent_entries(state, "touched_files", MAX_TOUCHED_FILES)
    if not entries:
        return "[Files] (none)"
    return "\n".join(["[Files]", *(f"  - {_format_file_row(entry)}" for entry in entries)])


def _critical_artifacts(state: dict[str, Any]) -> str:
    entries = _recent_entries(state, "artifacts", MAX_ARTIFACTS)
    if not entries:
        return "[Artifacts] (none)"
    return "\n".join(["[Artifacts]", *(f"  - {_format_artifact_row(entry)}" for entry in entries)])


def _recent_entries(state: dict[str, Any], attr: str, limit: int) -> list[dict[str, Any]]:
    """最近触碰倒序:足迹字典的插入序即触碰序,重复触碰会被移到队尾。"""
    store = getattr(state.get("runtime"), attr, None)
    if not isinstance(store, dict):
        return []
    return list(reversed(list(store.values())))[:limit]


def _format_file_row(entry: dict[str, Any]) -> str:
    trail = ", ".join(
        part for part in (str(entry.get("op", "")), _short_time(str(entry.get("at", ""))), str(entry.get("via", ""))) if part
    )
    path = str(entry.get("path", ""))
    return f"{path} ({trail})" if trail else path


def _format_artifact_row(entry: dict[str, Any]) -> str:
    meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
    details = []
    size = _artifact_size_text(meta)
    if size:
        details.append(size)
    source = str(meta.get("source") or entry.get("via") or "")
    if source:
        details.append(f"from {source}")
    path = str(entry.get("path", ""))
    return f"{path} ({', '.join(details)})" if details else path


def _artifact_size_text(meta: dict[str, Any]) -> str:
    """体积整组优先用原始主文本(source_*),缺失才整组退回落盘文件(json_*)。

    行数与字节数必须来自同一组,否则模型会把「3000 行 / 189KB」误读成同一份
    内容的两个维度,而它们本就描述不同对象。
    """
    lines = meta.get("source_lines")
    byte_count = meta.get("source_bytes")
    if not isinstance(lines, int) or not isinstance(byte_count, int):
        lines = meta.get("json_lines")
        byte_count = meta.get("json_bytes")
    segments = []
    if isinstance(lines, int):
        segments.append(f"{lines:,} lines")
    if isinstance(byte_count, int):
        segments.append(f"{byte_count:,} bytes")
    return " / ".join(segments)


def _short_time(value: str) -> str:
    """完整 UTC ISO 时间戳 -> 紧凑的 MM-DD HH:MM(存储保真、渲染瘦身)。"""
    try:
        return datetime.fromisoformat(value).strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return value


def rollup_evicted_messages(evicted: list[Any], *, current_narrative: str = "") -> tuple[str, bool]:
    """把被逐出的消息压成一段增量叙事,返回 (条目, 是否走了确定性 fallback)。

    只喂【被逐出的片段】,不喂整个转录,也不喂白名单块——白名单由
    render_critical_context 在每次构建 prompt 时从实时 state 重建,根本不经过
    压缩路径,所以不存在"压缩时丢失"的可能。增量输入同时修掉了旧实现
    "压缩调用本身可能爆上下文"的问题。
    """
    if not evicted:
        return "", False
    payload = {
        "current_narrative": current_narrative,
        "evicted_messages": [_message_snapshot(message) for message in evicted],
    }
    try:
        response = create_model().invoke(
            [
                SystemMessage(content=CONTEXT_ROLLUP_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ]
        )
        summary = _parse_rollup_summary(str(getattr(response, "content", "") or ""))
        if summary:
            return summary, False
    except Exception:
        pass
    return fallback_narrative_entry(evicted), True


def fallback_narrative_entry(evicted: list[Any], *, max_chars: int = FALLBACK_MAX_CHARS) -> str:
    """LLM 不可用时的确定性骨架:逐条结构化提取,核心是保住产物指针。

    产物指针在盘上永远有效,只要指针还在叙事里,被逐出的长输出就能按路径回读
    ——这是 Phase 1 落盘成果不在压缩中丢失的关键。超出预算时丢最旧的记录,
    与叙事折叠同向降级。
    """
    rows = [row for row in (_skeleton_row(message) for message in evicted) if row]
    if not rows:
        return ""
    dropped = 0
    while len(rows) > 1 and len(_render_rows(rows, dropped)) > max_chars:
        rows.pop(0)
        dropped += 1
    return _render_rows(rows, dropped)[:max_chars]


def merge_narrative(current: str, addition: str, *, max_chars: int = NARRATIVE_MAX_CHARS) -> str:
    """旧叙事 + 新条目;超上限时把最旧条目【逐条折叠成一行】,而非整段重述。

    逐条降级让信息密度随时间梯度下降,而不是整体一起糊掉——这是防"摘要的摘要"
    衰减的关键:旧条目被压成单行快照,新条目保持完整。
    """
    entries = _narrative_entries(current) + _narrative_entries(addition)
    if not entries:
        return ""
    if len("\n".join(entries)) <= max_chars:
        return "\n".join(entries)
    folded: list[str] = []
    rest = list(entries)
    while len(rest) > 1 and len(_render_narrative(folded, rest, max_chars)) > max_chars:
        folded.append(_one_line(rest.pop(0), FOLD_SNIPPET_CHARS))
    rendered = _render_narrative(folded, rest, max_chars)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, max_chars - 3)] + "..."


def _narrative_entries(text: str) -> list[str]:
    return [line for line in str(text or "").splitlines() if line.strip()]


def _render_narrative(folded: list[str], rest: list[str], max_chars: int) -> str:
    """折叠桶只用"未折叠条目"之外的剩余预算。

    否则桶本身(最多 12 条快照)可能撑满上限,把尾部的最新条目挤出并截掉——
    而最新条目恰恰是必须完整保留的那部分。
    """
    tail = "\n".join(rest)
    if not folded:
        return tail
    header = f"[... {len(folded)} earlier entries folded]"
    budget = max_chars - len(tail) - 1
    if budget <= len(header):
        return tail
    kept: list[str] = []
    used = len(header)
    for snippet in reversed(folded[-FOLD_BUCKET_MAX_SNIPPETS:]):
        cost = len(snippet) + 3
        if used + cost > budget:
            break
        kept.append(snippet)
        used += cost
    bucket = header + (" " + " | ".join(reversed(kept)) if kept else "")
    return "\n".join([bucket, *rest])


def _render_rows(rows: list[str], dropped: int) -> str:
    if dropped:
        return "\n".join([f"[... {dropped} older records dropped]", *rows])
    return "\n".join(rows)


def _skeleton_row(message: Any) -> str:
    if isinstance(message, ToolMessage):
        return _tool_row(message)
    if isinstance(message, AIMessage):
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            rendered = ", ".join(
                f"{call.get('name')}({_one_line(json.dumps(call.get('args', {}), ensure_ascii=False, default=str), SKELETON_ERROR_CHARS)})"
                for call in calls
            )
            return f"- call: {rendered}"
        return f"- result: {_one_line(_message_text(message), SKELETON_TEXT_CHARS)}"
    if isinstance(message, HumanMessage):
        return f"- user: {_one_line(_message_text(message), SKELETON_TEXT_CHARS)}"
    return f"- {type(message).__name__}: {_one_line(_message_text(message), SKELETON_TEXT_CHARS)}"


def _tool_row(message: ToolMessage) -> str:
    name = str(getattr(message, "name", "") or "tool")
    result = _parsed_result(message)
    if not isinstance(result, dict):
        return f"- {name}: {_one_line(_message_text(message), SKELETON_TEXT_CHARS)}"
    parts = [f"ok={result.get('ok')}"]
    if result.get("exit_code") is not None:
        parts.append(f"exit={result.get('exit_code')}")
    pointer = _artifact_pointer(result)
    if pointer:
        parts.append(f"artifact={pointer}")
    error = result.get("error") or result.get("stderr")
    if error:
        parts.append(f"err={_one_line(str(error), SKELETON_ERROR_CHARS)}")
    return f"- {name}: {' '.join(parts)}"


def _message_snapshot(message: Any) -> dict[str, str]:
    """给 rollup 的消息快照。

    产物指针单独成字段,避免被 content 的字符截断吃掉——落盘结果的
    artifact_path 在 JSON 里可能排在长字段之后,截断会把它切掉。
    """
    snapshot = {
        "type": type(message).__name__,
        "name": str(getattr(message, "name", "") or ""),
        "content": _short_text(_message_text(message), ROLLUP_MESSAGE_CHARS),
    }
    if isinstance(message, ToolMessage):
        result = _parsed_result(message)
        if isinstance(result, dict):
            pointer = _artifact_pointer(result)
            if pointer:
                snapshot["artifact"] = pointer
    return snapshot


def _parsed_result(message: ToolMessage) -> Any:
    try:
        return json.loads(str(message.content))
    except (TypeError, ValueError):
        return None


def _artifact_pointer(result: dict[str, Any]) -> str:
    for key in ("artifact_path", "stdout_path", "stderr_path"):
        value = result.get(key)
        if value:
            return str(value)
    return ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _one_line(text: str, limit: int) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)] + "..."


def _parse_rollup_summary(content: str) -> str:
    """从 rollup 响应里取 summary;叙事条目始终压成单行,便于逐条折叠。"""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    raw = fenced.group(1) if fenced else content
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        return ""
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    return _one_line(str(parsed.get("summary") or ""), NARRATIVE_MAX_CHARS)


def memory_event(memory: dict[str, Any], *, node: str) -> dict[str, Any]:
    working = memory.get("working_memory", {})
    history = memory.get("history_summary_store", {})
    return {
        "type": "memory_snapshot",
        "node": node,
        "rules_count": len(memory.get("rules", {}).get("rules", [])),
        "todo_count": len(working.get("todos", [])),
        "source_count": len(working.get("sources", [])),
        "handoff_count": len(working.get("agent_handoffs", [])),
        "notepad_exists": bool(history.get("notepad_exists")),
        "history_exists": bool(history.get("history_exists")),
        "history_path": history.get("history_path", HISTORY_SUMMARY_FILE),
        "layers": {
            "rules": _event_layer_summary(memory.get("rules", {})),
            "critical_context": _short_text(str(memory.get("critical_context", "")), 420),
            "working_memory": _event_layer_summary(working),
            "history_summary_store": _event_layer_summary(history),
        },
    }


def read_history_summary(state: RuntimeState) -> dict[str, Any]:
    path = state.assert_workspace_path(state.workspace / HISTORY_SUMMARY_FILE)
    if not path.exists():
        return {"ok": True, "path": HISTORY_SUMMARY_FILE, "content": "", "exists": False}
    content = read_text_lossy(path)
    state.record_read(path, complete=True)
    return {"ok": True, "path": HISTORY_SUMMARY_FILE, "content": content, "exists": True}


def persist_history_summary(state: RuntimeState, summary: str) -> dict[str, Any]:
    path = state.assert_workspace_path(state.workspace / HISTORY_SUMMARY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"# NexusAgent History Summary\n\n_Updated: {timestamp}_\n\n{summary.strip()}\n"
    path.write_text(content, encoding="utf-8")
    state.record_read(path, complete=True)
    return {"ok": True, "path": HISTORY_SUMMARY_FILE, "lines": len(content.splitlines())}


def _event_layer_summary(layer: dict[str, Any]) -> str:
    if not layer:
        return "(empty)"
    text = json.dumps(layer, ensure_ascii=False, default=str)
    return _short_text(text, 420)


def _trim_handoffs(handoffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = []
    for handoff in handoffs[-6:]:
        trimmed.append(
            {
                "from_agent": handoff.get("from_agent", ""),
                "to_agent": handoff.get("to_agent", ""),
                "instruction": _short_text(
                    str(handoff.get("instruction", "")),
                    MAX_TEXT_CHARS["agent_handoff_instruction"],
                ),
                "result": _short_text(str(handoff.get("result", "")), MAX_TEXT_CHARS["agent_handoff_result"]),
            }
        )
    return trimmed


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
