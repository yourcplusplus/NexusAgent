from __future__ import annotations

import logging
import re
from typing import Callable, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger("nexusagent.retry")

T = TypeVar("T")

# 通过异常类名/消息子串识别瞬态错误(匹配前会去除全部空白):
# openai/httpx/tavily 的异常层级各不相同且随版本漂移,子串匹配让本模块
# 保持零 SDK 依赖。误判的代价由"只包幂等调用"的原则兜底(见 retry_transient)。
TRANSIENT_ERROR_MARKERS = (
    "ratelimit",
    "apitimeout",
    "apiconnection",
    "timeout",
    "timedout",
    "connection",
    "transport",
    "servererror",
    "internalservererror",
    "badgateway",
    "serviceunavailable",
    "overloaded",
    "econnreset",
    "temporarilyunavailable",
)


def is_transient_error(exc: BaseException) -> bool:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    text = " ".join(
        f"{type(item).__name__.lower()} {str(item).lower()}" for item in chain
    )
    normalized = re.sub(r"\s+", "", text)
    return any(marker in normalized for marker in TRANSIENT_ERROR_MARKERS)


def retry_transient(
    *, attempts: int = 3, initial_wait: float = 1.0, max_wait: float = 8.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """只对瞬态网络/API 错误重试(指数退避 + 抖动)。

    禁止用于非幂等操作(BashTool/FileWriteTool/FileEditTool):重试会重放
    副作用。本装饰器只应包裹幂等的读侧调用;耗尽次数后原样重抛。
    """

    def decorate(func: Callable[..., T]) -> Callable[..., T]:
        return retry(
            retry=retry_if_exception(is_transient_error),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(func)

    return decorate
