"""工具输出落盘:超长结果写盘,inline 只留头尾摘要 + 文件指针。

设计意图:上下文压缩时最容易丢失的原始细节(完整 stdout、报错栈、搜索结果)
不应该只存在于消息转录里。这里把超长输出整体落到 ``.nexusagent/tool-outputs/``,
inline 保留头尾摘要与可回读的路径指针,使模型既能判断结果是否正常,
又能在需要时按路径取回全文。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from nexusagent.core.state import ARTIFACT_OP, RuntimeState

ARTIFACT_DIR = Path(".nexusagent") / "tool-outputs"

# 主要文本字段:按此优先级探测,首个命中的字段作为主载荷(拿到大头字符预算)
PRIMARY_TEXT_FIELDS = ("stdout", "content", "diff", "matches", "output")

SCALAR_STRING_CHARS = 200  # 短标量字符串的保留上限
MIN_FIELD_CHARS = 160  # 摘要字段至少保留的字符数
TEXT_BUDGET_RATIO = 0.6  # 文本摘要合计占 inline 预算的比例
PRIMARY_BUDGET_SHARE = 0.7  # 主载荷在文本预算中的占比
OMITTED_TEMPLATE = "\n[... {count} lines omitted ...]\n"
FULL_OUTPUT_TEMPLATE = "\n[Full output saved to: {path}]"
TRIM_MARKER_TEMPLATE = "[... output trimmed to fit inline budget; full output: {path}]"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

__all__ = ["ARTIFACT_DIR", "PRIMARY_TEXT_FIELDS", "spill_tool_output"]


def spill_tool_output(
    runtime: RuntimeState,
    tool_name: str,
    result: dict,
    inline_chars: int = 2000,
    head_lines: int = 30,
    tail_lines: int = 20,
) -> dict:
    """短结果原样返回;超长结果落盘并返回头尾摘要版。

    返回的 dict 保留短标量字段,把主要文本字段换成
    「头 N 行 + 省略提示 + 尾 M 行 + 完整输出路径」,并新增 ``artifact_path``。
    """
    if not isinstance(result, dict):
        return result
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if len(serialized) <= inline_chars:
        return result

    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    artifact_path = _write_artifact(runtime, tool_name, payload)
    _register_artifact(runtime, tool_name, artifact_path, payload, result)

    text_budget = max(MIN_FIELD_CHARS, int(inline_chars * TEXT_BUDGET_RATIO))
    budgets = _field_budgets(_oversized_keys(result), text_budget)

    summary: dict[str, Any] = {}
    for key, value in result.items():
        if key == "artifact_path":
            continue
        if key in budgets:
            summary[key] = _summarize_text(
                _as_text(value),
                head_lines=head_lines,
                tail_lines=tail_lines,
                max_chars=budgets[key],
                artifact_path=artifact_path,
            )
        else:
            summary[key] = value
    summary["artifact_path"] = artifact_path
    return _enforce_inline_budget(summary, list(budgets), inline_chars)


def _write_artifact(runtime: RuntimeState, tool_name: str, payload: str) -> str:
    """写入完整结果(已序列化文本),返回 workspace 相对路径。"""
    # tool_name 来自模型的 tool_call,做字符白名单清洗以杜绝路径穿越
    safe_name = _UNSAFE_NAME_CHARS.sub("_", tool_name).strip("._-") or "tool"
    directory = runtime.workspace / ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_name}-{time.time_ns()}.json"
    path.write_text(payload, encoding="utf-8")
    return path.relative_to(runtime.workspace).as_posix()


def _register_artifact(
    runtime: RuntimeState,
    tool_name: str,
    artifact_path: str,
    payload: str,
    result: dict,
) -> None:
    """把落盘产物登记进 artifacts 白名单(上限 MAX_ARTIFACTS,由 RuntimeState 维护)。

    元数据在落盘这一刻量取,避免渲染白名单时读盘。体积分两组同时记录:
    source_* 描述原始主文本(模型的体积直觉来自它——3000 行输出与 7 行 JSON
    是两种判断),json_* 描述落盘文件本身。渲染时整组优先 source_*。
    """
    meta: dict[str, Any] = {
        "json_lines": payload.count("\n") + 1,
        "json_bytes": len(payload.encode("utf-8")),
        "source": str(result.get("command") or tool_name),
    }
    source_stats = _source_text_stats(result)
    if source_stats is not None:
        meta["source_lines"], meta["source_bytes"] = source_stats
    runtime.record_touch(artifact_path, op=ARTIFACT_OP, via=tool_name, meta=meta)


def _source_text_stats(result: dict) -> tuple[int, int] | None:
    """原始主文本字段的 (行数, 字节数):按 PRIMARY_TEXT_FIELDS 取首个非空字段。

    两个数字取自同一份 _as_text 结果,保证它们描述的是同一个对象。没有主文本
    字段时返回 None,由渲染层整组退回 json_*。
    """
    for key in PRIMARY_TEXT_FIELDS:
        value = result.get(key)
        if value is None:
            continue
        text = _as_text(value)
        if text.strip():
            return len(text.splitlines()), len(text.encode("utf-8"))
    return None


def _oversized_keys(result: dict) -> list[str]:
    """需要摘要的字段:按 PRIMARY_TEXT_FIELDS 优先级在前,其余超长字段随后。"""
    oversized = {key for key, value in result.items() if key != "artifact_path" and _is_oversized(value)}
    ordered = [key for key in PRIMARY_TEXT_FIELDS if key in oversized]
    ordered.extend(sorted(oversized.difference(ordered)))
    return ordered


def _is_oversized(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) > SCALAR_STRING_CHARS
    if isinstance(value, (list, tuple, dict)):
        return len(json.dumps(value, ensure_ascii=False, default=str)) > SCALAR_STRING_CHARS
    return False


def _field_budgets(keys: list[str], text_budget: int) -> dict[str, int]:
    """主载荷拿大头,其余超长字段平分剩余预算。"""
    if not keys:
        return {}
    if len(keys) == 1:
        return {keys[0]: text_budget}
    primary = max(MIN_FIELD_CHARS, int(text_budget * PRIMARY_BUDGET_SHARE))
    each = max(MIN_FIELD_CHARS, (text_budget - primary) // (len(keys) - 1))
    budgets = {keys[0]: primary}
    for key in keys[1:]:
        budgets[key] = each
    return budgets


def _as_text(value: Any) -> str:
    """把字段值摊平成可裁剪的文本(列表逐项一行,便于头尾截取)。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _summarize_text(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
    max_chars: int,
    artifact_path: str,
) -> str:
    """头 N 行 + 省略提示 + 尾 M 行 + 完整输出路径。

    行数上限之外还有字符预算:行少但单行极长时按字符截断,行多时按预算
    继续裁行,始终把省略行数如实标出。
    """
    lines = text.splitlines()
    footer = FULL_OUTPUT_TEMPLATE.format(path=artifact_path)
    budget = max(0, max_chars - len(footer))
    if not lines:
        return footer.lstrip("\n")

    head = lines[:head_lines]
    tail = lines[-tail_lines:] if tail_lines > 0 else []
    omitted = len(lines) - len(head) - len(tail)

    if omitted <= 0:
        body = "\n".join(lines)
        return (body if len(body) <= budget else body[:budget]) + footer

    marker = OMITTED_TEMPLATE.format(count=omitted)
    body = "\n".join(head) + marker + "\n".join(tail)
    if len(body) <= budget:
        return body + footer

    inner = max(0, budget - len(OMITTED_TEMPLATE.format(count=len(lines))))
    head = _clip_lines(head, inner // 2, from_end=False)
    tail = _clip_lines(tail, inner - inner // 2, from_end=True)
    omitted = max(0, len(lines) - len(head) - len(tail))
    return "\n".join(head) + OMITTED_TEMPLATE.format(count=omitted) + "\n".join(tail) + footer


def _clip_lines(lines: list[str], budget: int, *, from_end: bool) -> list[str]:
    ordered = list(reversed(lines)) if from_end else list(lines)
    kept: list[str] = []
    used = 0
    for line in ordered:
        cost = len(line) + 1
        if kept and used + cost > budget:
            break
        kept.append(line)
        used += cost
    return list(reversed(kept)) if from_end else kept


def _enforce_inline_budget(summary: dict, trimmable: list[str], inline_chars: int) -> dict:
    """兜底:短标量字段过多时继续压缩已摘要字段,保证 inline 契约。

    极端情况下把摘要换成一行指针 —— ``artifact_path`` 始终保留,模型不会
    失去取回全文的线索。
    """
    for _ in range(8):
        if len(json.dumps(summary, ensure_ascii=False, default=str)) <= inline_chars:
            break
        target = ""
        longest = 0
        for key in trimmable:
            length = len(str(summary.get(key, "")))
            if length > longest:
                target, longest = key, length
        if not target or longest <= len(TRIM_MARKER_TEMPLATE):
            break
        summary[target] = TRIM_MARKER_TEMPLATE.format(path=summary.get("artifact_path", ""))
    return summary
