from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage

import nexusagent.graph.status_bar as status_bar
from nexusagent.agents.code_agent import run_code_agent
from nexusagent.core.state import RuntimeState
from nexusagent.graph.context_window import DEFAULT_CHARS_PER_TOKEN
from nexusagent.graph.memory import build_layered_memory, render_critical_context
from nexusagent.graph.nodes import (
    _planner_input,
    _verifier_input,
    agent_status_bar_node,
    agent_status_bar_route,
    planner_node,
    verifier_node,
)
from nexusagent.graph.status_bar import (
    clear_status_cache,
    ensure_env_status,
    render_env_status,
    render_event_summary,
    render_git_board,
    render_status_line,
    with_fresh_env,
)
from nexusagent.graph.workflow import build_workflow

STATUS_BLOCK_TOKEN_BUDGET = 500


def make_state(workspace: Path, **extra) -> dict:
    return {"runtime": RuntimeState(workspace=workspace), "task": "demo", "messages": [], **extra}


def require_git() -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git executable not available")
    return git


def git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    """workspace 本身即仓库根(另一种形态是把 workspace 放在被忽略的子目录里)。"""
    require_git()
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def skip_if_inside_a_repo(workspace: Path) -> None:
    """tmp_path 落在某个仓库内时跳过「非仓库」用例(标准环境是 /tmp,不在仓库内)。"""
    if status_bar._probe_git(workspace)["is_repo"]:
        pytest.skip("pytest tmp_path is inside a git repository")


# ---------- 1. git 探针:只报 repo 与 branch ----------


def test_git_probe_reports_repo_and_branch_without_dirty_counts(git_workspace: Path) -> None:
    """dirty 会有意不报:默认 workspace 被 .gitignore 忽略,那里的改动不是 Agent 的。"""
    (git_workspace / "uncommitted.txt").write_text("scratch\n", encoding="utf-8")
    state = make_state(git_workspace)

    env = ensure_env_status(state["runtime"], state, ttl=60.0)

    assert env["git"]["is_repo"] is True
    assert env["git"]["repo"] == git_workspace.as_posix()
    assert env["git"]["branch"] == git(git_workspace, "rev-parse", "--abbrev-ref", "HEAD")
    # workspace 就是仓库根,不存在「未跟踪」的问题
    assert env["git"]["tracked"] is True
    rendered = render_git_board(env) + render_env_status(env) + render_status_line(env)
    assert "dirty" not in rendered
    assert "uncommitted.txt" not in rendered
    assert "branch=" in rendered


def test_git_board_notes_ignored_workspace(tmp_path: Path) -> None:
    """生产形态:workspace 位于被忽略的子目录下,产物不会出现在该仓库的 git 状态里。"""
    require_git()
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text(".nexusagent/\n", encoding="utf-8")
    workspace = tmp_path / ".nexusagent" / "workspaces" / "workspace-1"
    workspace.mkdir(parents=True)
    state = make_state(workspace)

    env = ensure_env_status(state["runtime"], state, ttl=60.0)
    board = render_git_board(env)

    assert env["git"]["is_repo"] is True
    assert env["git"]["repo"] == tmp_path.as_posix()
    assert env["git"]["tracked"] is False
    assert "workspace not tracked by this repo" in board


def test_git_probe_degrades_outside_any_repo(tmp_path: Path) -> None:
    require_git()
    skip_if_inside_a_repo(tmp_path)
    state = make_state(tmp_path)

    env = ensure_env_status(state["runtime"], state, ttl=60.0)

    assert env["git"] == {"is_repo": False, "reason": "not a git repository"}
    assert render_git_board(env).startswith("[Git] (not a git repo)")
    assert "not a git repo" in render_status_line(env)


def test_git_probe_degrades_without_git_executable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nexusagent.graph.status_bar.shutil.which", lambda _name: None)
    state = make_state(tmp_path)

    env = ensure_env_status(state["runtime"], state, ttl=60.0)

    assert env["git"]["is_repo"] is False
    assert env["git"]["reason"] == "git executable not found"
    assert render_git_board(env).startswith("[Git] (not a git repo)")


def test_git_probe_reports_detached_head(git_workspace: Path) -> None:
    sha = git(git_workspace, "rev-parse", "--short", "HEAD")
    git(git_workspace, "checkout", "-q", "--detach")
    state = make_state(git_workspace)

    env = ensure_env_status(state["runtime"], state, ttl=60.0)

    assert env["git"]["branch"] == f"detached@{sha}"


# ---------- 2. TTL 缓存 ----------


def test_cache_reuses_snapshot_within_ttl(monkeypatch, git_workspace: Path) -> None:
    probes = {"count": 0}
    calls = {"count": 0}
    real_probe = status_bar._probe_git
    real_run = status_bar._git_run

    def counting_probe(workspace: Path) -> dict:
        probes["count"] += 1
        return real_probe(workspace)

    def counting_run(workspace: Path, args: list[str]):
        calls["count"] += 1
        return real_run(workspace, args)

    monkeypatch.setattr(status_bar, "_probe_git", counting_probe)
    monkeypatch.setattr(status_bar, "_git_run", counting_run)
    state = make_state(git_workspace)

    first = ensure_env_status(state["runtime"], state, ttl=60.0)
    after_first = calls["count"]
    second = ensure_env_status(state["runtime"], state, ttl=60.0)

    # TTL 内 git 子进程调用次数不增加(plan 的性能验收:TTL 内 ≤ 1 次探针)
    assert probes["count"] == 1
    assert calls["count"] == after_first
    assert second["refresh_seq"] == first["refresh_seq"]
    assert second["git"] == first["git"]
    # 时效契约:命中缓存返回的快照年龄仍严格小于 TTL
    assert second["age_seconds"] >= first["age_seconds"]
    assert second["age_seconds"] < 60.0


def test_cache_recollects_after_ttl_expires(git_workspace: Path) -> None:
    state = make_state(git_workspace)

    first = ensure_env_status(state["runtime"], state, ttl=60.0)
    second = ensure_env_status(state["runtime"], state, ttl=0.0)

    assert second["refresh_seq"] == first["refresh_seq"] + 1
    assert second["age_seconds"] < 60.0


def test_clear_status_cache_forces_recollection(git_workspace: Path) -> None:
    state = make_state(git_workspace)

    first = ensure_env_status(state["runtime"], state, ttl=60.0)
    clear_status_cache(state["runtime"])
    second = ensure_env_status(state["runtime"], state, ttl=60.0)

    assert second["refresh_seq"] == 1
    assert first["refresh_seq"] == 1


def test_status_cache_stays_out_of_serialized_state(git_workspace: Path) -> None:
    """缓存挂在 runtime 上,而 serialize_state 整体跳过 runtime,不污染 checkpoint 载荷。"""
    from nexusagent.core.checkpoint import serialize_state

    state = make_state(git_workspace, plan_summary="plan")
    with_fresh_env(state)

    payload = serialize_state(state)

    assert "runtime" not in payload
    assert "status_cache" not in str(payload)


# ---------- 3. 工作区 delta ----------


def test_first_snapshot_only_establishes_baseline(tmp_path: Path) -> None:
    (tmp_path / "preexisting.txt").write_text("before task\n", encoding="utf-8")
    state = make_state(tmp_path)

    env = ensure_env_status(state["runtime"], state, ttl=60.0)

    assert env["delta"]["baseline"] is True
    assert env["delta"]["added"] == 0
    assert env["delta"]["samples"] == []
    assert "delta (baseline)" in render_git_board(env)


def test_delta_reports_added_modified_and_deleted(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("keep\n", encoding="utf-8")
    (tmp_path / "gone.txt").write_text("gone\n", encoding="utf-8")
    state = make_state(tmp_path)
    ensure_env_status(state["runtime"], state, ttl=60.0)  # 立基线

    (tmp_path / "new.html").write_text("<html></html>\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("keep changed\n", encoding="utf-8")
    (tmp_path / "gone.txt").unlink()
    env = ensure_env_status(state["runtime"], state, ttl=0.0)

    assert env["delta"]["baseline"] is False
    assert (env["delta"]["added"], env["delta"]["modified"], env["delta"]["deleted"]) == (1, 1, 1)
    assert env["delta"]["samples"] == [
        {"op": "+", "path": "new.html"},
        {"op": "~", "path": "keep.txt"},
        {"op": "-", "path": "gone.txt"},
    ]
    assert "delta +1 ~1 -1" in render_git_board(env)


def test_delta_counts_same_size_rewrite_as_modified(tmp_path: Path) -> None:
    path = tmp_path / "same.txt"
    path.write_text("aaaa\n", encoding="utf-8")
    state = make_state(tmp_path)
    ensure_env_status(state["runtime"], state, ttl=60.0)

    path.write_text("bbbb\n", encoding="utf-8")
    # mtime 显式错开:tmpfs 的时间戳粒度会让紧邻两次写入的 mtime_ns 完全相同,
    # 那样测到的是「粒度边界」而不是本用例要验证的「同尺寸改写靠 mtime 检出」。
    # 该边界的成因与取舍记在 status_bar 模块 docstring 里。
    bumped = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(bumped, bumped))
    env = ensure_env_status(state["runtime"], state, ttl=0.0)

    assert env["delta"]["modified"] == 1
    assert env["delta"]["samples"] == [{"op": "~", "path": "same.txt"}]


def test_delta_ignores_runtime_scratch_and_dependency_dirs(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    ensure_env_status(state["runtime"], state, ttl=60.0)

    (tmp_path / ".nexusagent" / "tool-outputs").mkdir(parents=True)
    (tmp_path / ".nexusagent" / "tool-outputs" / "BashTool-1.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".nexusagent" / "checkpoints").mkdir(parents=True)
    (tmp_path / ".nexusagent" / "checkpoints" / "checkpoint.json").write_text("{}", encoding="utf-8")
    for directory in (".venv/lib", "node_modules/pkg", "__pycache__", ".git"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
        (tmp_path / directory / "junk.txt").write_text("junk\n", encoding="utf-8")
    (tmp_path / "real.html").write_text("<html></html>\n", encoding="utf-8")

    env = ensure_env_status(state["runtime"], state, ttl=0.0)

    assert env["delta"]["added"] == 1
    assert env["delta"]["samples"] == [{"op": "+", "path": "real.html"}]


def test_delta_marks_truncated_scan_as_partial(monkeypatch, tmp_path: Path) -> None:
    """清单被条数上限截断时必须标出:此时增删计数只覆盖扫到的子集。"""
    state = make_state(tmp_path)
    ensure_env_status(state["runtime"], state, ttl=60.0)

    monkeypatch.setattr(status_bar, "MANIFEST_ENTRIES_LIMIT", 2)
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("c\n", encoding="utf-8")
    env = ensure_env_status(state["runtime"], state, ttl=0.0)

    assert env["delta"]["partial"] is True
    assert "(partial scan)" in render_git_board(env)


# ---------- 4. 渲染契约 ----------


def test_git_board_falls_back_to_pending_without_snapshot() -> None:
    assert render_git_board(None) == "[Git] (git: pending)"
    assert render_git_board({}) == "[Git] (git: pending)"
    assert render_env_status(None) == ""
    assert render_env_status({}) == ""


def test_git_board_truncates_long_paths_but_keeps_the_tail() -> None:
    deep = "a" * 90
    env = {
        "git": {"is_repo": True, "repo": f"/{'/'.join([deep] * 4)}/project", "branch": "main", "tracked": True},
        "delta": {
            "baseline": False,
            "added": 1,
            "modified": 0,
            "deleted": 0,
            "samples": [{"op": "+", "path": f"{'/'.join([deep] * 3)}/index.html"}],
        },
    }

    board = render_git_board(env)

    assert "index.html" in board
    assert "project" in board
    assert len(board) < 400


def test_env_status_section_lists_only_what_the_whitelist_does_not() -> None:
    env = {
        "git": {"is_repo": True, "repo": "/tmp/proj", "branch": "main", "tracked": True},
        "delta": {"baseline": False, "added": 1, "modified": 0, "deleted": 0, "samples": [{"op": "+", "path": "a.html"}]},
        "budget": {"tokens": 12000, "limit": 400000, "groups": 7, "pressure": 0.03},
        "artifacts": 3,
        "background": {"count": 1, "samples": [{"path": ".nexusagent/background/job-1.out", "bytes": 2048}]},
    }

    section = render_env_status(env)

    assert "context: 12000/400000 tokens (3.0%), 7 window group(s)" in section
    assert "artifacts: 3 spilled output(s) on disk" in section
    assert "job-1.out (2.0KB)" in section
    # 白名单已有 repo/branch 与 delta,小节里不再重复
    assert "branch=" not in section
    assert "repo=" not in section
    assert "[Git]" not in section


def test_status_block_stays_within_token_budget() -> None:
    deep = "very-long-segment-" * 6
    env = {
        "git": {"is_repo": True, "repo": f"/{'/'.join([deep] * 5)}/proj", "branch": deep, "tracked": False},
        "delta": {
            "baseline": False,
            "added": 9,
            "modified": 9,
            "deleted": 9,
            "samples": [{"op": "+", "path": f"{deep}/{index}.html"} for index in range(9)],
        },
        "budget": {"tokens": 399999, "limit": 400000, "groups": 42, "pressure": 0.999},
        "artifacts": 100,
        "background": {"count": 7, "samples": [{"path": f"{deep}/job-{i}.out", "bytes": 1048576} for i in range(9)]},
    }

    block = render_git_board(env) + "\n" + render_env_status(env)
    line = render_status_line(env)

    assert len(block) // DEFAULT_CHARS_PER_TOKEN <= STATUS_BLOCK_TOKEN_BUDGET
    assert len(line) // DEFAULT_CHARS_PER_TOKEN < 100


def test_render_event_summary_carries_git_and_budget() -> None:
    env = {
        "git": {"is_repo": True, "repo": "/tmp/proj", "branch": "main", "tracked": True},
        "delta": {"baseline": False, "added": 2, "modified": 1, "deleted": 0, "samples": [{"op": "+", "path": "a.html"}]},
        "budget": {"tokens": 12000, "limit": 400000, "groups": 7, "pressure": 0.03},
        "age_seconds": 0.25,
    }

    text = render_event_summary(env)

    assert "[Git] repo=/tmp/proj branch=main" in text
    assert "delta +2 ~1 -0 +a.html" in text
    assert "context: 12000/400000 tokens" in text
    assert "snapshot age: 0.25s" in text


# ---------- 5. 图拓扑与节点 ----------


def test_topology_routes_monitor_and_compressor_through_status_bar() -> None:
    edges = {(edge.source, edge.target) for edge in build_workflow().get_graph().edges}

    assert ("context_monitor", "agent_status_bar") in edges
    assert ("context_monitor", "context_compressor") in edges
    assert ("context_compressor", "agent_status_bar") in edges
    assert ("agent_status_bar", "verifier") in edges
    assert ("agent_status_bar", "planner") in edges
    assert ("agent_status_bar", "final") in edges
    # 压缩器不再直接连目标节点,路由职责已并入状态栏
    assert ("context_compressor", "verifier") not in edges
    assert ("context_compressor", "planner") not in edges


def test_status_bar_node_sets_snapshot_and_emits_event(monkeypatch, tmp_path: Path) -> None:
    events: list[dict] = []
    monkeypatch.setattr("nexusagent.graph.nodes._get_writer", lambda: events.append)
    state = make_state(tmp_path, context_next_node="verifier")

    result = agent_status_bar_node(state)

    assert events and events[0]["type"] == "status_bar"
    assert events[0]["git"] == result["env_status"]["git"]
    assert result["env_status"]["git"]["is_repo"] is False
    # 节点只通过返回值交付更新,不改调用方传入的 state
    assert "env_status" not in state


def test_status_bar_route_follows_context_next_node() -> None:
    assert agent_status_bar_route({}) == "verifier"
    assert agent_status_bar_route({"context_next_node": "planner"}) == "planner"
    assert agent_status_bar_route({"context_next_node": "final"}) == "final"


def test_planner_and_verifier_ensure_a_fresh_snapshot_on_direct_entry(monkeypatch, tmp_path: Path) -> None:
    """首个 planner 先于 status_bar 节点运行,验证节点自身也会补采快照。"""

    class FakeBoundModel:
        def invoke(self, messages):  # type: ignore[no-untyped-def]
            return AIMessage(content="plan ready")

    class FakeModel:
        def bind_tools(self, tools):  # type: ignore[no-untyped-def]
            return FakeBoundModel()

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())

    planned = planner_node({"task": "demo", "runtime": RuntimeState(workspace=tmp_path), "attempts": 0, "max_attempts": 3})
    verified = verifier_node(
        {
            "runtime": RuntimeState(workspace=tmp_path),
            "task": "demo",
            "todos": [{"id": "todo-1", "content": "verify", "status": "in_progress", "note": ""}],
        }
    )

    for update in (planned, verified):
        env = update["env_status"]
        assert env["git"]["is_repo"] is False
        assert env["delta"]["baseline"] is True
        assert env["age_seconds"] < status_bar.get_status_bar_ttl_seconds()


# ---------- 6. 循环内 TTL 注入 ----------


def _tool_call_once_model(captured: list[list], *, calls_until_done: int = 1):
    class FakeBoundModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages):  # type: ignore[no-untyped-def]
            self.calls += 1
            captured.append(list(messages))
            if self.calls <= calls_until_done:
                return AIMessage(
                    content="using a tool",
                    tool_calls=[{"name": "BogusTool", "args": {}, "id": f"tc{self.calls}"}],
                )
            return AIMessage(content="done")

    class FakeModel:
        def bind_tools(self, tools):  # type: ignore[no-untyped-def]
            return FakeBoundModel()

    return FakeModel()


def _status_messages(messages: list) -> list:
    return [
        message
        for message in messages
        if isinstance(message, SystemMessage) and str(message.content).startswith("[STATUS ")
    ]


def test_planner_loop_injects_status_message_only_after_ttl(monkeypatch, tmp_path: Path) -> None:
    captured: list[list] = []
    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: _tool_call_once_model(captured))
    monkeypatch.setattr("nexusagent.graph.status_bar.get_status_bar_ttl_seconds", lambda: 0.0)

    result = planner_node({"task": "demo", "runtime": RuntimeState(workspace=tmp_path), "attempts": 0})

    injected = _status_messages(captured[-1])
    assert len(injected) == 1
    assert "[STATUS " in str(injected[0].content)
    # 注入只进本地 messages,不进转录:否则 _last_ai_content 会把它当成节点产出
    assert _status_messages(result["messages"]) == []
    assert result["metadata"]["planner_raw"] == "done"


def test_planner_loop_does_not_inject_within_ttl(monkeypatch, tmp_path: Path) -> None:
    captured: list[list] = []
    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: _tool_call_once_model(captured))

    result = planner_node({"task": "demo", "runtime": RuntimeState(workspace=tmp_path), "attempts": 0})

    assert _status_messages(captured[-1]) == []
    assert _status_messages(result["messages"]) == []
    assert result["metadata"]["planner_raw"] == "done"


def test_verifier_loop_injects_status_message_after_ttl(monkeypatch, tmp_path: Path) -> None:
    captured: list[list] = []
    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: _tool_call_once_model(captured))
    monkeypatch.setattr("nexusagent.graph.status_bar.get_status_bar_ttl_seconds", lambda: 0.0)

    result = verifier_node(
        {
            "runtime": RuntimeState(workspace=tmp_path),
            "task": "demo",
            "todos": [{"id": "todo-1", "content": "verify", "status": "in_progress", "note": ""}],
        }
    )

    assert len(_status_messages(captured[-1])) == 1
    assert _status_messages(result["messages"]) == []


def test_code_agent_loop_injects_status_message_after_ttl(monkeypatch, tmp_path: Path) -> None:
    captured: list[list] = []
    monkeypatch.setattr("nexusagent.agents.code_agent.create_model", lambda: _tool_call_once_model(captured))
    monkeypatch.setattr("nexusagent.graph.status_bar.get_status_bar_ttl_seconds", lambda: 0.0)
    state = make_state(
        tmp_path,
        todos=[{"id": "todo-1", "content": "build", "status": "in_progress", "note": ""}],
        plan_summary="plan",
    )

    result = run_code_agent(state, "build it", writer=lambda _: None, max_loops=3)

    assert len(_status_messages(captured[-1])) == 1
    assert _status_messages(result["messages"]) == []


# ---------- 7. 端到端:整图跑通 ----------


def test_workflow_run_routes_through_status_bar_with_fresh_snapshots(monkeypatch, tmp_path: Path) -> None:
    """monitor → compressor(空转跳过) → status_bar → verifier → final 全程可跑,
    且每次进入 LLM 节点前的快照年龄都小于 TTL。"""
    verdict = json.dumps(
        {
            "passed": True,
            "reason": "ok",
            "checks": [{"name": "html", "passed": True, "detail": "ok"}],
            "recommended_next_instruction": "",
        }
    )

    class FakeBoundModel:
        def invoke(self, messages):  # type: ignore[no-untyped-def]
            return AIMessage(content=verdict)

    class FakeModel:
        def bind_tools(self, tools):  # type: ignore[no-untyped-def]
            return FakeBoundModel()

    monkeypatch.setattr("nexusagent.graph.nodes.create_model", lambda: FakeModel())
    monkeypatch.setattr("nexusagent.graph.memory.create_model", lambda: FakeModel())
    # 阈值压到 1,强制走压缩分支;此时窗口只有 1 组,压缩器会以 nothing_to_evict 空转跳过
    monkeypatch.setenv("NEXUS_CONTEXT_TOKEN_LIMIT", "1")

    runtime = RuntimeState(workspace=tmp_path)

    def run_graph() -> tuple[list[dict], list[dict]]:
        events: list[dict] = []
        updates: list[dict] = []
        for mode, event in build_workflow().stream(
            {
                "task": "demo",
                "runtime": runtime,
                "messages": [],
                "attempts": 0,
                "max_attempts": 1,
            },
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                events.append(event)
            else:
                updates.append(event)
        return events, updates

    events, updates = run_graph()

    kinds = [event.get("type") for event in events]
    assert "context_monitor" in kinds
    assert "context_compression" in kinds
    assert "status_bar" in kinds
    assert kinds.index("context_compression") < kinds.index("status_bar")
    assert any("verifier" in update for update in updates)
    assert any("final" in update for update in updates)

    # 压缩器确实被走到了:首轮以「无可逐出」空转跳过,之后真的逐出了一组
    compression_events = [event for event in events if event.get("type") == "context_compression"]
    assert compression_events[0]["skipped_reason"] == "nothing_to_evict"
    assert compression_events[0]["evicted_messages"] == 0
    evicted = [event for event in compression_events if event["evicted_messages"] > 0]
    assert evicted, "1 token 阈值下应当出现一次真实逐出"
    # 判定性 fallback 兜底(假模型不返回 summary 字段),指针保住了
    assert evicted[-1]["fallback"] is True
    assert (tmp_path / "HISTORY_SUMMARY.md").exists()

    ttl = status_bar.get_status_bar_ttl_seconds()
    snapshots = [event for event in events if event.get("type") == "status_bar"]
    assert snapshots
    for snapshot in snapshots:
        assert snapshot["age_seconds"] < ttl
        assert "dirty" not in json.dumps(snapshot)

    verifier_update = next(update["verifier"] for update in updates if "verifier" in update)
    assert verifier_update["env_status"]["refresh_seq"] >= 1
    # TTL 内复用同一份快照,所以一次快跑只有首次采集,delta 停在基线是正确行为
    assert verifier_update["env_status"]["delta"]["baseline"] is True

    # 第二轮:先在盘上改动,再把 TTL 归零(等价于「距上次采集已超时」);此时清单
    # 还在缓存里,delta 才有可比基线。清缓存不行——那会连基线一起丢掉。
    (tmp_path / "out.html").write_text("<html></html>\n", encoding="utf-8")
    monkeypatch.setenv("NEXUS_STATUS_BAR_TTL_SECONDS", "0.0001")
    _, second_updates = run_graph()

    # delta 的语义是「距上次刷新之间变了什么」,由越过 TTL 的第一次采集消费掉;
    # 这一轮里那一次就是 planner 的入口采集(它先于状态栏节点运行)。
    planned = next(update["planner"] for update in second_updates if "planner" in update)
    delta = planned["env_status"]["delta"]
    assert delta["baseline"] is False
    # out.html 必在样本里;条数不写死——首轮压缩落下的 HISTORY_SUMMARY.md 同属新增
    assert {"op": "+", "path": "out.html"} in delta["samples"]
    assert delta["added"] >= 1
    # 变化被消费后,同一轮里后续采集对比的是同一份清单,自然归零
    second_env = next(update["verifier"]["env_status"] for update in second_updates if "verifier" in update)
    assert second_env["delta"]["added"] == 0


# ---------- 8. prompt 注入 ----------


def test_prompts_carry_git_board_and_env_section(tmp_path: Path) -> None:
    state = make_state(
        tmp_path,
        env_status={
            "git": {"is_repo": True, "repo": "/tmp/proj", "branch": "main", "tracked": True},
            "delta": {"baseline": False, "added": 1, "modified": 0, "deleted": 0, "samples": [{"op": "+", "path": "a.html"}]},
            "budget": {"tokens": 12000, "limit": 400000, "groups": 7, "pressure": 0.03},
            "artifacts": 2,
        },
        acceptance_criteria=["done"],
        todos=[],
    )
    memory = build_layered_memory(state, node="planner")

    prompts = {
        "planner": _planner_input(state, memory),
        "verifier": _verifier_input(state, memory),
        "critical_context": render_critical_context(state),
    }

    for name, text in prompts.items():
        assert "[Git] repo=/tmp/proj branch=main" in text, name
        assert "delta +1 ~0 -0 +a.html" in text, name

    for name in ("planner", "verifier"):
        assert "Environment status:" in prompts[name], name
        assert "context: 12000/400000 tokens" in prompts[name], name


def test_env_section_is_omitted_when_no_snapshot_yet(tmp_path: Path) -> None:
    state = make_state(tmp_path, acceptance_criteria=["done"], todos=[])
    memory = build_layered_memory(state, node="planner")

    text = _planner_input(state, memory)

    assert "[Git] (git: pending)" in text
    assert "Environment status:" not in text


# ---------- 9. CLI 分支 ----------


def test_formatter_renders_status_bar_panel(capsys) -> None:
    """经 print_event 走一遍分发,确保 custom_event 分支真的接上了。"""
    from nexusagent.cli.formatter import print_event

    print_event(
        {
            "type": "custom_event",
            "event": {
                "type": "status_bar",
                "git": {"is_repo": True, "repo": "/tmp/proj", "branch": "main", "tracked": True},
                "delta": {"baseline": False, "added": 2, "modified": 1, "deleted": 0, "samples": [{"op": "+", "path": "a.html"}]},
                "budget": {"tokens": 12000, "limit": 400000, "groups": 7, "pressure": 0.03},
                "age_seconds": 0.1,
            },
        }
    )

    output = capsys.readouterr().out
    assert "Environment Status" in output
    assert "branch=main" in output
    assert "delta +2 ~1 -0" in output


def test_event_summary_renders_status_bar_event() -> None:
    from nexusagent.cli.event_summary import summarize_event

    summary = summarize_event(
        {
            "type": "custom_event",
            "event": {
                "type": "status_bar",
                "git": {"is_repo": True, "repo": "/tmp/proj", "branch": "main", "tracked": True},
                "delta": {"baseline": True, "added": 0, "modified": 0, "deleted": 0, "samples": []},
                "budget": {"tokens": 0, "limit": 400000, "groups": 0, "pressure": 0.0},
                "age_seconds": 0.0,
            },
        }
    )

    assert summary.title == "Environment Status"
    assert summary.category == "environment"
    assert "branch=main" in summary.body
