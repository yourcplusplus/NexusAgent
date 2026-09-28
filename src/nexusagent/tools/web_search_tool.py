from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

from nexusagent.core.retry import retry_transient


def web_search(
    query: str,
    max_results: int | str = 5,
    include_answer: bool | str = True,
) -> dict[str, Any]:
    """Search the web with Tavily and return a compact structured result."""
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"ok": False, "error": "missing required .env setting: TAVILY_API_KEY"}

    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return {"ok": False, "error": f"tavily-python is not installed: {exc}"}

    try:
        max_value = int(max_results)
    except (TypeError, ValueError):
        max_value = 5
    max_value = max(1, min(max_value, 10))
    answer_value = _coerce_bool(include_answer)

    try:
        client = TavilyClient(api_key=api_key)
        response = _search_with_retry(client, query, max_value, answer_value)
    except Exception as exc:
        return {"ok": False, "query": query, "error": f"{type(exc).__name__}: {exc}"}

    results = []
    for item in response.get("results", []) or []:
        results.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", ""))[:1200],
                "score": item.get("score"),
            }
        )

    return {
        "ok": True,
        "query": query,
        "answer": response.get("answer") or "",
        "results": results,
    }


def build_web_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="WebSearchTool",
        func=web_search,
        description=(
            "Search the web with Tavily. Args: query, optional max_results, optional include_answer. "
            "Returns answer and result sources with title, url, content, and score."
        ),
    )


@retry_transient()
def _search_with_retry(
    client: Any, query: str, max_results: int, include_answer: bool
) -> dict[str, Any]:
    return client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=include_answer,
    )


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}
