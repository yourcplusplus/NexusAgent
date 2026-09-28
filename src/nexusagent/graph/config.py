"""图级运行时配置:环境变量读取与默认值。

这些 getter 原先住在 ``graph/nodes.py``。Phase 4 的 ``status_bar``(算 token 预算)与
``agents/code_agent``(循环内刷新状态栏)都需要读它们,而 ``nodes`` 已经导入了
``code_agent``——继续放在 nodes 里会与两者形成循环导入。故独立成模块,
``nodes`` 仍以原函数名导入,既有调用点与测试引用不变。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from nexusagent.graph.context_window import DEFAULT_KEEP_GROUPS, DEFAULT_WINDOW_RATIO

DEFAULT_CONTEXT_TOKEN_LIMIT = 400000
DEFAULT_STATUS_BAR_TTL_SECONDS = 5.0


def get_context_token_limit() -> int:
    load_dotenv()
    raw = os.getenv("NEXUS_CONTEXT_TOKEN_LIMIT", str(DEFAULT_CONTEXT_TOKEN_LIMIT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_TOKEN_LIMIT
    return value if value > 0 else DEFAULT_CONTEXT_TOKEN_LIMIT


def get_context_keep_groups() -> int:
    load_dotenv()
    raw = os.getenv("NEXUS_CONTEXT_KEEP_GROUPS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_KEEP_GROUPS
    return value if value > 0 else DEFAULT_KEEP_GROUPS


def get_context_window_ratio() -> float:
    load_dotenv()
    raw = os.getenv("NEXUS_CONTEXT_WINDOW_RATIO", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_WINDOW_RATIO
    return value if 0 < value <= 1 else DEFAULT_WINDOW_RATIO


def get_status_bar_ttl_seconds() -> float:
    """状态栏快照的 TTL:TTL 内复用缓存,不重复起 git 子进程、不重扫工作区。"""
    load_dotenv()
    raw = os.getenv("NEXUS_STATUS_BAR_TTL_SECONDS", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_STATUS_BAR_TTL_SECONDS
    return value if value > 0 else DEFAULT_STATUS_BAR_TTL_SECONDS
