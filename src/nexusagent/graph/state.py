from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from nexusagent.core.state import FileEntry, RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class VerificationResult(TypedDict):
    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class SourceItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float


class AgentHandoff(TypedDict, total=False):
    from_agent: str
    to_agent: str
    instruction: str
    result: str


class VerificationCheck(TypedDict, total=False):
    name: str
    passed: bool
    detail: str


class CompressionEvent(TypedDict, total=False):
    before_tokens: int
    after_tokens: int
    removed_messages: int
    summary: str
    next_node: str


class LayeredMemory(TypedDict, total=False):
    rules: dict[str, Any]
    working_memory: dict[str, Any]
    history_summary_store: dict[str, Any]


class CriticalContext(TypedDict, total=False):
    """压缩期强制保留的关键信息:由代码从结构化状态渲染,不经 LLM 转述。

    files 来自 RuntimeState.touched_files;git 来自 agent_status_bar 采集的
    env_status 快照,无快照时渲染为 pending。
    """

    goal: str
    constraints: list[str]
    todos_digest: list[dict[str, Any]]
    files: list[FileEntry]
    git: dict[str, Any]
    artifacts: list[dict[str, Any]]


class EnvStatus(TypedDict, total=False):
    """agent_status_bar 采集的环境快照(Phase 4)。

    git 段只有 repo / branch / tracked,刻意没有 dirty:默认 workspace 位于
    ``.nexusagent/workspaces/`` 且被 .gitignore 忽略,git 的 dirty 统计里是开发者
    自己的未提交改动,渲染给模型会被误认成 Agent 的产出。改了哪些文件由
    delta(盘上实测的增删改)与 Files 白名单(工具层登记)表达。
    """

    git: dict[str, Any]
    delta: dict[str, Any]
    budget: dict[str, Any]
    background: dict[str, Any]
    artifacts: int
    refresh_seq: int
    refreshed_at: str
    age_seconds: float


class NexusGraphState(TypedDict, total=False):
    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    passed: bool
    attempts: int
    max_attempts: int
    final_answer: str
    intent_route: str
    intent_reason: str
    intent_confidence: float
    chat_response: str
    session_id: str
    session_turn: int
    session_context: str
    last_actor_summary: str
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    verifier_summary: str
    verification_checks: list[VerificationCheck]
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_pressure_notified: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    critical_context: CriticalContext
    env_status: EnvStatus
    history_summary: str
    last_error: str
    metadata: dict[str, Any]
