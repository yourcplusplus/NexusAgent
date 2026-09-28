"""滑动窗口的纯函数核心:分组、窗口规划、配对闭合、逐出 ID 生成。

本模块不读 state、不调 LLM、不碰磁盘——输入输出都是普通数据结构,便于单测。
节点侧只负责把 state 里的消息传进来,再把 evict_ids 转成 RemoveMessage。

分组依赖消息 ID 的确定性前缀(`<node>-a<attempt>-<seq>`),该前缀由
`nodes._stamp_message_ids` 盖上;reducer 会保留显式盖章的 ID。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

DEFAULT_KEEP_GROUPS = 4
DEFAULT_WINDOW_RATIO = 0.30
DEFAULT_CHARS_PER_TOKEN = 4
CLOSURE_MAX_PASSES = 3

_GROUP_ID = re.compile(r"^(?P<node>[A-Za-z_][A-Za-z0-9_]*)-a(?P<attempt>\d+)-(?P<seq>\d+)$")


@dataclass(frozen=True)
class MessageGroup:
    """一次节点执行的产出(转录里的一段连续消息)。"""

    key: str
    pinned: bool
    messages: tuple[Any, ...]
    tokens: int


@dataclass(frozen=True)
class EvictionPlan:
    """逐出决策。skipped_reason 非空时 evict_ids 必为空,调用方应跳过本轮压缩。"""

    evict_ids: list[str] = field(default_factory=list)
    skipped_reason: str = ""
    notes: list[str] = field(default_factory=list)
    groups_total: int = 0
    groups_evicted: int = 0
    messages_kept: int = 0
    messages_evicted: int = 0
    tokens_kept: int = 0
    tokens_evicted: int = 0


def group_messages(messages: list[Any]) -> list[MessageGroup]:
    """按确定性 ID 前缀把转录切成执行组,保持转录顺序且组内连续。

    - `<node>-a<attempt>-<seq>` → 归入 `node-a<attempt>` 组,可逐出
    - 转录【首条】HumanMessage → `task` 组并钉住(勿忘:生产中转录起点为空,
      该情形只在测试或未来把 task 种进转录时出现)
    - 其它无法解析的消息 → `solo` 组,可逐出(避免旧格式消息把窗口钉死)

    钉住是【位次】语义而非类型语义:只有首条用户请求被钉住。否则一旦未来把多轮
    用户输入追加进同一份转录,每个 HumanMessage 都钉住会让窗口再也压不下去。
    这也保证保留窗口始终是转录的一段连续区间,不会出现中间带洞。

    按【连续段】切分:键或钉住属性变化即开新组。这保证"保留窗口"总是转录的
    一段干净后缀,而不是一堆带洞的片段。
    """
    runs: list[tuple[str, bool, list[Any]]] = []
    for index, message in enumerate(messages):
        key, is_user_turn = _classify(message)
        pinned = is_user_turn and index == 0
        if runs and runs[-1][0] == key and runs[-1][1] == pinned:
            runs[-1][2].append(message)
            continue
        runs.append((key, pinned, [message]))
    return [
        MessageGroup(key=key, pinned=pinned, messages=tuple(items), tokens=_tokens_of(items))
        for key, pinned, items in runs
    ]


def close_window(keep: list[Any], transcript: list[Any]) -> tuple[list[Any], list[Any]]:
    """把保留窗口闭合到满足配对不变量,返回 (keep, evict)。

    不变量:任一 ToolMessage 与其引用的 AIMessage(其 tool_calls 含该
    tool_call_id)必须同侧——要么都在 keep,要么都在 evict。两类违约都会让
    LLM API 报 400:调用没有结果、或结果引用不存在的调用。

    规则 1(A∈keep 但结果不在 keep)→ 把结果【拉回】keep。
    规则 2(T∈keep 但调用者不在 keep)→ 把 T【踢出】keep。

    规则 1 的取向是"宁可少逐出,不可留下坏消息":若分组有瑕疵险些把调用与结果
    拆开,选择保留更多上下文而不是丢掉调用。两条规则都只增删 ToolMessage,
    而 ToolMessage 不含 tool_calls,故一轮即达不动点;cap 纯属防御。
    """
    keep_keys = {id(message) for message in keep}
    for _ in range(CLOSURE_MAX_PASSES):
        called = {
            str(call.get("id"))
            for message in keep
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
        }
        pulled = [
            message
            for message in transcript
            if isinstance(message, ToolMessage)
            and str(message.tool_call_id) in called
            and id(message) not in keep_keys
        ]
        dropped = [
            message
            for message in keep
            if isinstance(message, ToolMessage) and str(message.tool_call_id) not in called
        ]
        if not pulled and not dropped:
            break
        dropped_keys = {id(message) for message in dropped}
        keep = [message for message in keep if id(message) not in dropped_keys] + pulled
        keep_keys = {id(message) for message in keep}
    evict = [message for message in transcript if id(message) not in keep_keys]
    return keep, evict


def plan_eviction(
    messages: list[Any],
    *,
    keep_groups: int = DEFAULT_KEEP_GROUPS,
    token_limit: int = 0,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
) -> EvictionPlan:
    """算出应当逐出的消息 ID。

    组数约束与 token 约束取更严者;**最新一组永远保留**,即使它单独超出 token
    ceiling(组内配对必须完整,无法切分,而 Phase 1 的落盘已让单组天然有界)。
    """
    groups = group_messages(messages)
    if not groups:
        return EvictionPlan(skipped_reason="no_messages")

    window_tokens = max(1, int(token_limit * window_ratio)) if token_limit > 0 else 0
    kept_indices, notes = _select_window(groups, keep_groups=keep_groups, window_tokens=window_tokens)
    kept_messages = [message for index in sorted(kept_indices) for message in groups[index].messages]

    keep, evict = close_window(kept_messages, list(messages))
    if len(keep) != len(kept_messages):
        notes.append("pairing_closure_adjusted")

    keep_keys = {id(message) for message in keep}
    stats = {
        "groups_total": len(groups),
        "groups_evicted": sum(1 for group in groups if not all(id(m) in keep_keys for m in group.messages)),
        "messages_kept": len(keep),
        "messages_evicted": len(evict),
        "tokens_kept": _tokens_of(keep),
        "tokens_evicted": _tokens_of(evict),
    }

    if not keep:
        return EvictionPlan(skipped_reason="empty_window", notes=notes, **stats)
    unidentified = [message for message in evict if not getattr(message, "id", None)]
    if unidentified:
        notes.append(f"unidentified_messages={len(unidentified)}")
        return EvictionPlan(skipped_reason="unidentified_messages", notes=notes, **stats)
    if not evict:
        return EvictionPlan(skipped_reason="nothing_to_evict", notes=notes, **stats)
    return EvictionPlan(evict_ids=[str(message.id) for message in evict], notes=notes, **stats)


def estimate_window_tokens(messages: list[Any]) -> int:
    """转录窗口的字符量估算:各执行组之和。

    monitor 与状态栏共用这一个口径;两边都不 tokenize、不读盘。
    """
    return sum(group.tokens for group in group_messages(messages))


def estimate_payload_tokens(payload: Any, *, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    """记忆载荷的字符量估算。调用方复用 state 里已有的快照,不重新构建。"""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    return len(text) // chars_per_token


def _select_window(
    groups: list[MessageGroup],
    *,
    keep_groups: int,
    window_tokens: int,
) -> tuple[set[int], list[str]]:
    """从最新一组往前收集保留组下标;返回 (下标集合, 备注)。"""
    notes: list[str] = []
    newest = len(groups) - 1
    kept = {newest}
    counted = 0 if groups[newest].pinned else 1
    tokens = groups[newest].tokens

    for index in range(newest - 1, -1, -1):
        group = groups[index]
        if group.pinned:
            kept.add(index)
            continue
        if counted >= keep_groups:
            notes.append("group_count_limit_reached")
            break
        if window_tokens and tokens + group.tokens > window_tokens:
            notes.append("window_token_ceiling_reached")
            break
        kept.add(index)
        tokens += group.tokens
        counted += 1

    if window_tokens and tokens > window_tokens:
        notes.append("newest_group_exceeds_window")
    kept |= {index for index, group in enumerate(groups) if group.pinned}
    return kept, notes


def _classify(message: Any) -> tuple[str, bool]:
    """返回 (组键, 是否为用户轮次)。是否钉住由调用方按位次决定。"""
    if isinstance(message, HumanMessage):
        return "task", True
    match = _GROUP_ID.match(str(getattr(message, "id", "") or ""))
    if match:
        return f"{match.group('node')}-a{match.group('attempt')}", False
    return "solo", False


def _tokens_of(messages: list[Any] | tuple[Any, ...]) -> int:
    if not messages:
        return 0
    return max(1, sum(_message_chars(message) for message in messages) // DEFAULT_CHARS_PER_TOKEN)


def _message_chars(message: Any) -> int:
    """粗略字符量:content 文本 + 工具调用参数。仅用于阈值判断,不需要精确。"""
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    size = len(content)
    for call in getattr(message, "tool_calls", None) or []:
        size += len(json.dumps(call.get("args", {}), ensure_ascii=False, default=str))
    return size
