from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from nexusagent.core.retry import retry_transient


def create_model() -> ChatOpenAI:
    load_dotenv()

    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")

    missing = [name for name, value in {"API_KEY": api_key, "MODEL": model, "BASE_URL": base_url}.items() if not value]
    if missing:
        raise RuntimeError(f"missing required .env setting(s): {', '.join(missing)}")

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=0,
    )


def invoke_with_retry(runnable: Any, messages: list) -> Any:
    """带瞬态错误重试的 invoke(限流/超时/5xx,指数退避)。

    非瞬态错误立即抛出;调用方保留各自的 try/except 语义。
    """
    return _invoke(runnable, tuple(messages))


@retry_transient(attempts=4)
def _invoke(runnable: Any, messages: tuple) -> Any:
    return runnable.invoke(list(messages))
