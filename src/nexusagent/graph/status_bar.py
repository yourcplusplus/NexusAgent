"""环境状态栏:每次进入 LLM 节点前刷新 git / 工作区 / 预算快照。

几条边界都是刻意的:

- **git 段只报 repo 与 branch,不报 dirty。** 默认 workspace 位于
  ``<项目根>/.nexusagent/workspaces/`` 下且被 .gitignore 忽略,``git status`` 会向上
  找到项目仓库,它的 dirty 统计里装的是开发者自己的未提交改动。把它渲染给模型,
  模型会把这些改动误认成自己的产出。「改了哪些文件」由工作区 delta 与 Files 白名单
  承担——前者是盘上实测,后者是工具层登记,两者都不依赖 git。
- **TTL 缓存是唯一的成本闸门。** git 子进程与工作区扫描都只在 TTL 过期时发生,
  所以「每个节点转移都刷新」与「TTL 内子进程调用 ≤ 1 次」并不矛盾。
- **只读。** 不 init、不 commit、不写仓库,因此结果可以被安全缓存;非 git 工作区
  与缺失 git 可执行文件都降级为一条 ``is_repo: False``,不抛异常。
- **delta 用 (字节数, mtime_ns) 判断,不读内容。** 代价是一个已知边界:时间戳粒度
  较粗的文件系统(tmfs 实测两次紧邻写入的 mtime_ns 可以完全相同)上,「同长度且在
  同一个时间戳刻度内完成的改写」检测不到。要闭合这个缝就得哈希内容,而这是每个
  节点转移都要跑的热路径;工具经手的写入另有 Files 白名单这条更准的信号,故接受。
- **清单有 400 条上限,截断时明说。** 超大工作区上增删计数只覆盖扫到的子集,渲染层
  会标 ``(partial scan)``,不默认它全准。

渲染不截断,体积靠构造期上限控制(与白名单同一条约定):路径、样本条数、整块
字数都在构造时就有界,而不是渲染后再切。
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage

from nexusagent.core.state import RuntimeState
from nexusagent.graph.config import get_context_token_limit, get_status_bar_ttl_seconds
from nexusagent.graph.context_window import (
    estimate_payload_tokens,
    estimate_window_tokens,
    group_messages,
)

GIT_TIMEOUT_SECONDS = 5.0
MANIFEST_ENTRIES_LIMIT = 400
SAMPLE_LIMIT = 5
PATH_MAX_CHARS = 60
SECTION_MAX_CHARS = 1200
STATUS_LINE_MAX_CHARS = 320
BACKGROUND_DIR = Path(".nexusagent") / "background"

# 工作区 delta 的扫描排除项。与 checkpoint.workspace_manifest 的差异是有意的:
# 这里在遍历时就剪掉整棵子树(而非先枚举再过滤),且把 .nexusagent 整块排除——
# 工具落盘每次调用都新增一个文件,计进 delta 只是噪声,它们的条数与体积已由
# [Artifacts] 白名单和预算块表达。
SKIP_DIR_NAMES = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".nexusagent"}
)

_CACHE_KEY = "env"
_GIT_KEY = "git"
_MANIFEST_KEY = "workspace"
_SEQ_KEY = "seq"


def ensure_env_status(
    runtime: RuntimeState,
    state: dict[str, Any],
    *,
    ttl: float | None = None,
) -> dict[str, Any]:
    """返回环境快照;TTL 内复用缓存,过期才重新采集(git 子进程 + 工作区扫描)。

    返回体始终带 ``age_seconds``:本次返回距上次真实采集的秒数。它按毫秒向下截断,
    因此 ``age_seconds < TTL`` 恒成立——调用方可以据此断言时效契约,而不必关心
    本次到底是命中缓存还是重新采集。
    """
    now = time.monotonic()
    ttl_value = get_status_bar_ttl_seconds() if ttl is None else ttl
    cache = _cache_of(runtime)
    entry = cache.get(_CACHE_KEY)
    if not isinstance(entry, dict) or now - float(entry.get("at", 0.0)) >= ttl_value:
        value = _collect(runtime, state, cache, now=now, ttl=ttl_value)
        entry = {"at": now, "value": value}
        cache[_CACHE_KEY] = entry
    value = dict(entry["value"])
    value["age_seconds"] = math.floor(max(0.0, now - float(entry["at"])) * 1000) / 1000
    return value


def with_fresh_env(state: dict[str, Any], *, runtime: RuntimeState | None = None) -> dict[str, Any]:
    """返回带最新 env_status 的状态副本(节点入口用,TTL 内为空操作)。"""
    resolved = runtime if runtime is not None else state.get("runtime")
    if resolved is None:
        return dict(state)
    return {**state, "env_status": ensure_env_status(resolved, state)}


def env_seq(state: dict[str, Any]) -> int:
    """当前快照的刷新序号;缺失或未采集时为 0。

    循环内用它判断「自上一次注入后是否真的重新采集过」,从而做到一次刷新只注入一条
    状态——文件路径与进程存活都不是可靠信号,自增序号才是。
    """
    env = state.get("env_status")
    if not isinstance(env, dict):
        return 0
    seq = env.get("refresh_seq")
    return int(seq) if isinstance(seq, int) else 0


def append_status_update(
    messages: list[Any],
    *,
    runtime: RuntimeState,
    state: dict[str, Any],
    last_seq: int,
) -> int:
    """TTL 到期时向循环内 messages 追加一条 ``[STATUS ...]`` SystemMessage。

    只进调用方的本地循环 messages,不进节点的 produced_messages:后者会被
    ``_last_ai_content`` 当成节点产出读取(verifier 会拿状态栏文本去解析 JSON),
    也会作为转录的一部分被盖上 ID 并参与窗口逐出。状态的新鲜度由事件流留痕,
    不靠转录——转录里的旧状态反而会与最新状态并存,变成噪声。

    成功注入时顺带把新快照写回 ``state["env_status"]``:调用方传进来的都是本节点的
    工作副本,这样同一节点内后续构建(如嵌套的 codeAgent)能看到最新快照。

    返回本次生效的 refresh_seq,调用方持有它即可实现「一次刷新只注入一条」。
    """
    fresh = ensure_env_status(runtime, state)
    seq = int(fresh.get("refresh_seq") or 0)
    if seq == last_seq:
        return last_seq
    if isinstance(state, dict):
        state["env_status"] = fresh
    messages.append(SystemMessage(content=render_status_line(fresh)))
    return seq


def render_git_board(env_status: Any) -> str:
    """白名单 ``[Git]`` 板块:repo / branch / 工作区 delta,单行有界。

    从未采集过快照时回落到 ``(git: pending)``——生产路径上 planner 与 verifier
    都在节点入口保证快照存在,该回落只在直接调用渲染器的场景(测试、未来的
    新节点忘了注入)出现。
    """
    git = _section(env_status, "git")
    delta = _delta_text(_section(env_status, "delta"))
    if not git:
        return "[Git] (git: pending)"
    if not git.get("is_repo"):
        parts = ["(not a git repo)"]
    else:
        parts = [f"repo={_truncate_path(str(git.get('repo', '')))} branch={git.get('branch', '')}"]
        if git.get("tracked") is False:
            # 默认 workspace 就是这一档:它的产物不会出现在该仓库的 git 状态里
            parts.append("workspace not tracked by this repo")
    if delta:
        parts.append(delta)
    return "[Git] " + " | ".join(parts)


def render_env_status(env_status: Any) -> str:
    """prompt 里的「Environment status」小节,只放白名单没有的部分。

    白名单 ``[Git]`` 已承载 repo/branch 与工作区 delta,这里再放一遍等于同一事实
    在同一个 prompt 里出现两次(Phase 2 明确避免过的形态)。所以本节只放预算、
    产物计数与后台任务输出——它们每次节点转移都重算,不需要靠白名单续命。
    """
    if not isinstance(env_status, dict) or not env_status:
        return ""
    lines: list[str] = []
    budget = _section(env_status, "budget")
    if budget and budget.get("limit"):
        lines.append(
            f"context: {budget.get('tokens', 0)}/{budget.get('limit', 0)} tokens "
            f"({_percent(budget.get('pressure'))}), {budget.get('groups', 0)} window group(s)"
        )
    artifacts = env_status.get("artifacts")
    if isinstance(artifacts, int) and artifacts > 0:
        lines.append(f"artifacts: {artifacts} spilled output(s) on disk")
    background = _section(env_status, "background")
    if background and background.get("count"):
        lines.append(f"background: {_background_text(background)}")
    if not lines:
        return ""
    text = "Environment status:\n" + "\n".join(lines)
    if len(text) > SECTION_MAX_CHARS:
        return text[: SECTION_MAX_CHARS - 3] + "..."
    return text


def render_status_line(env_status: Any) -> str:
    """循环内追加的 ``[STATUS ...]`` 单行:自包含,不依赖白名单就在眼前。

    与 prompt 小节的分工不同,这里会带上 branch:它是循环内唯一的环境锚点,
    上一次注入可能已是很久以前的消息。
    """
    git = _section(env_status, "git")
    if git and git.get("is_repo"):
        head = f"branch={git.get('branch', '')}"
    elif git:
        head = "not a git repo"
    else:
        head = "git: pending"
    parts = [head]
    delta = _delta_text(_section(env_status, "delta"))
    if delta:
        parts.append(delta)
    budget = _section(env_status, "budget")
    if budget and budget.get("limit"):
        parts.append(
            f"ctx {budget.get('tokens', 0)}/{budget.get('limit', 0)} tok ({_percent(budget.get('pressure'))})"
        )
    artifacts = env_status.get("artifacts") if isinstance(env_status, dict) else None
    if isinstance(artifacts, int) and artifacts > 0:
        parts.append(f"artifacts {artifacts}")
    background = _section(env_status, "background")
    if background and background.get("count"):
        parts.append(f"jobs {background['count']}")
    stamp = str((env_status or {}).get("refreshed_at", "")) if isinstance(env_status, dict) else ""
    line = f"[STATUS {stamp}] " + " | ".join(parts)
    if len(line) > STATUS_LINE_MAX_CHARS:
        return line[: STATUS_LINE_MAX_CHARS - 3] + "..."
    return line


def render_event_summary(env_status: Any) -> str:
    """CLI 面板与 TUI 摘要共用的多行文本:git 板 + 环境小节 + 快照年龄。

    两个 CLI 模块各有各的外壳(Panel / EventSummary),但字段到文本的映射只在这里
    一份,免得同一个事件在两种呈现里慢慢长歪。
    """
    lines = [render_git_board(env_status)]
    section = render_env_status(env_status)
    if section:
        lines.extend(section.splitlines()[1:])
    age = env_status.get("age_seconds") if isinstance(env_status, dict) else None
    if age is not None:
        lines.append(f"snapshot age: {age}s")
    return "\n".join(lines)


def clear_status_cache(runtime: RuntimeState) -> None:
    """丢弃该 runtime 的快照缓存(测试与需要强制重采的场景)。"""
    cache = getattr(runtime, "status_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _collect(
    runtime: RuntimeState,
    state: dict[str, Any],
    cache: dict[str, Any],
    *,
    now: float,
    ttl: float,
) -> dict[str, Any]:
    cache[_SEQ_KEY] = int(cache.get(_SEQ_KEY, 0)) + 1
    return {
        "git": _git_snapshot(runtime, cache, now=now, ttl=ttl),
        "delta": _workspace_delta(runtime, cache, now=now, ttl=ttl),
        "budget": _budget(state),
        "background": _background_snapshot(runtime),
        "artifacts": _artifact_count(runtime),
        "refresh_seq": cache[_SEQ_KEY],
        "refreshed_at": _utc_stamp(),
    }


# ---------- git 探针(只读) ----------


def _git_snapshot(runtime: RuntimeState, cache: dict[str, Any], *, now: float, ttl: float) -> dict[str, Any]:
    entry = cache.get(_GIT_KEY)
    if isinstance(entry, dict) and now - float(entry.get("at", 0.0)) < ttl:
        return dict(entry["value"])
    value = _probe_git(Path(runtime.workspace))
    cache[_GIT_KEY] = {"at": now, "value": value}
    return dict(value)


def _probe_git(workspace: Path) -> dict[str, Any]:
    """只读探针:仓库顶层与当前分支;任何失败都降级成 ``is_repo: False``。"""
    if shutil.which("git") is None:
        return {"is_repo": False, "reason": "git executable not found"}
    toplevel = _git_output(workspace, ["rev-parse", "--show-toplevel"])
    if not toplevel:
        return {"is_repo": False, "reason": "not a git repository"}
    repo = Path(toplevel)
    branch = _git_output(workspace, ["rev-parse", "--abbrev-ref", "HEAD"]) or ""
    if branch in {"", "HEAD"}:
        short = _git_output(workspace, ["rev-parse", "--short", "HEAD"]) or ""
        branch = f"detached@{short}" if short else "(detached)"
    return {
        "is_repo": True,
        "repo": repo.as_posix(),
        "branch": branch,
        "tracked": _workspace_tracked(workspace, repo),
    }


def _workspace_tracked(workspace: Path, repo: Path) -> bool:
    """workspace 是仓库根、或其相对路径未被 ignore → True。

    探测失败一律按「已跟踪」处理:凭一次失败给模型一个否定结论,比不给结论更糟。
    """
    try:
        relative = workspace.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return False
    if relative in {"", "."}:
        return True
    result = _git_run(workspace, ["check-ignore", "-q", "--", relative])
    if result is None:
        return True
    return result.returncode != 0


def _git_output(workspace: Path, args: list[str]) -> str:
    result = _git_run(workspace, args)
    if result is None or result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_run(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """只读 git 调用;无 git、超时、非仓库、权限问题都返回 None 而不抛。"""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


# ---------- 工作区 delta ----------


def _workspace_delta(runtime: RuntimeState, cache: dict[str, Any], *, now: float, ttl: float) -> dict[str, Any]:
    entry = cache.get(_MANIFEST_KEY)
    if isinstance(entry, dict) and now - float(entry.get("at", 0.0)) < ttl:
        return dict(entry["delta"])
    previous = entry.get("manifest") if isinstance(entry, dict) else None
    previous_partial = bool(entry.get("partial")) if isinstance(entry, dict) else False
    current, partial = _scan_manifest(Path(runtime.workspace))
    delta = _diff_manifest(previous, current, partial=partial or previous_partial)
    cache[_MANIFEST_KEY] = {"at": now, "manifest": current, "partial": partial, "delta": delta}
    return dict(delta)


def _scan_manifest(workspace: Path) -> tuple[dict[str, tuple[int, int]], bool]:
    """工作区清单 ``{相对 posix 路径: (字节数, mtime_ns)}`` 与「是否被条数上限截断」。

    遍历时剪掉 SKIP_DIR_NAMES 整棵子树,并以条数上限兜底。checkpoint 的
    ``workspace_manifest`` 在这个位置不可用:它先 ``sorted(rglob("*"))`` 枚举全树,
    limit 只截结果、不省遍历,放在每个节点转移的路径上太贵。

    截断标记会一路传到渲染层:超大工作区上增删计数只覆盖扫到的部分,不能默认它全准。
    """
    entries: dict[str, tuple[int, int]] = {}
    if not workspace.exists():
        return entries, False
    truncated = False
    stack = [workspace]
    while stack:
        if len(entries) >= MANIFEST_ENTRIES_LIMIT:
            truncated = True
            break
        directory = stack.pop()
        try:
            children = list(os.scandir(directory))
        except OSError:
            continue
        for child in children:
            if len(entries) >= MANIFEST_ENTRIES_LIMIT:
                truncated = True
                break
            if child.name in SKIP_DIR_NAMES:
                continue
            try:
                if child.is_dir(follow_symlinks=False):
                    stack.append(Path(child.path))
                    continue
                if not child.is_file(follow_symlinks=False):
                    continue
                stat = child.stat()
            except OSError:
                continue
            entries[Path(child.path).relative_to(workspace).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return entries, truncated


def _diff_manifest(
    previous: dict[str, tuple[int, int]] | None,
    current: dict[str, tuple[int, int]],
    *,
    partial: bool = False,
) -> dict[str, Any]:
    """两次清单的差;首次采集只立基线(否则会把任务开始前的存量报成「新增」)。

    比较 (字节数, mtime_ns) 二元组而非只看尺寸:同长度覆盖写要靠 mtime 才看得出来。
    已知边界见模块 docstring——时间戳粒度粗的文件系统上,同一刻度内的同长度改写会漏;
    任一测清单被条数上限截断时(partial),增删计数都只覆盖扫到的子集。
    """
    if previous is None:
        return {"baseline": True, "partial": partial, "added": 0, "modified": 0, "deleted": 0, "samples": []}
    added = sorted(set(current) - set(previous))
    deleted = sorted(set(previous) - set(current))
    modified = sorted(path for path in set(current) & set(previous) if current[path] != previous[path])
    samples = [{"op": "+", "path": path} for path in added]
    samples += [{"op": "~", "path": path} for path in modified]
    samples += [{"op": "-", "path": path} for path in deleted]
    return {
        "baseline": False,
        "partial": partial,
        "added": len(added),
        "modified": len(modified),
        "deleted": len(deleted),
        "samples": samples[:SAMPLE_LIMIT],
    }


# ---------- 预算 / 后台任务 / 产物 ----------


def _budget(state: dict[str, Any]) -> dict[str, Any]:
    """token 预算:优先用 monitor 或压缩器刚算出的计数,缺失时按同一口径现算。

    现算这条只在首个 planner 出现(它先于 context_monitor 运行),口径与
    ``estimate_context_tokens`` 一致:窗口字符量 + 已有记忆快照字符量。
    """
    messages = list(state.get("messages", []) or [])
    tokens = state.get("context_token_count")
    if not isinstance(tokens, int) or tokens <= 0:
        tokens = estimate_window_tokens(messages) + estimate_payload_tokens(state.get("memory_snapshot") or {})
    limit = state.get("context_token_limit")
    if not isinstance(limit, int) or limit <= 0:
        limit = get_context_token_limit()
    return {
        "tokens": int(tokens),
        "limit": int(limit),
        "groups": len(group_messages(messages)),
        "pressure": round(tokens / limit, 3) if limit > 0 else 0.0,
    }


def _background_snapshot(runtime: RuntimeState) -> dict[str, Any]:
    """后台任务输出概览。

    只数目录下的落盘文件,不声称「活跃」:``_run_background`` 没把 pid 持久化,
    进程存活与否无法从盘上判断,给出一个做不到的判断比不给更糟。
    """
    directory = Path(runtime.workspace) / BACKGROUND_DIR
    if not directory.is_dir():
        return {"count": 0, "samples": []}
    try:
        paths = sorted(path for path in directory.iterdir() if path.is_file())
    except OSError:
        return {"count": 0, "samples": []}
    samples = []
    for path in paths[-SAMPLE_LIMIT:]:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        samples.append({"path": _relative_posix(path, Path(runtime.workspace)), "bytes": size})
    return {"count": len(paths), "samples": samples}


def _artifact_count(runtime: RuntimeState) -> int:
    artifacts = getattr(runtime, "artifacts", None)
    return len(artifacts) if isinstance(artifacts, dict) else 0


# ---------- 渲染辅助 ----------


def _delta_text(delta: Any) -> str:
    if not isinstance(delta, dict) or not delta:
        return ""
    if delta.get("baseline"):
        return "delta (baseline)"
    counts = f"delta +{delta.get('added', 0)} ~{delta.get('modified', 0)} -{delta.get('deleted', 0)}"
    # 清单被条数上限截断时明确标出:计数只覆盖扫到的子集,不能让模型默认它全准
    if delta.get("partial"):
        counts += " (partial scan)"
    samples = _samples_text(delta.get("samples"))
    return f"{counts} {samples}" if samples else counts


def _samples_text(samples: Any) -> str:
    if not isinstance(samples, list):
        return ""
    parts = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        path = _truncate_path(str(sample.get("path", "")))
        if not path:
            continue
        op = str(sample.get("op", ""))
        parts.append(f"{op}{path}" if op else path)
    return " ".join(parts)


def _background_text(background: dict[str, Any]) -> str:
    samples = background.get("samples")
    rendered = []
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            path = _truncate_path(str(sample.get("path", "")))
            if path:
                rendered.append(f"{path} ({_bytes_text(sample.get('bytes'))})")
    count = background.get("count", 0)
    head = f"{count} job output(s)"
    return f"{head} - {', '.join(rendered)}" if rendered else head


def _truncate_path(path: str, *, limit: int = PATH_MAX_CHARS) -> str:
    """过长路径保留尾部:末两段才是能定位文件的部分,头部用省略号。"""
    if len(path) <= limit:
        return path
    parts = [part for part in path.split("/") if part]
    tail = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else path)
    if len(tail) > limit - 4:
        tail = tail[-(limit - 4) :]
    return f".../{tail}"


def _bytes_text(value: Any) -> str:
    if not isinstance(value, int):
        return "?"
    if value < 1024:
        return f"{value}B"
    return f"{value / 1024:.1f}KB"


def _percent(pressure: Any) -> str:
    try:
        return f"{float(pressure) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _section(env_status: Any, key: str) -> dict[str, Any]:
    if not isinstance(env_status, dict):
        return {}
    value = env_status.get(key)
    return value if isinstance(value, dict) else {}


def _relative_posix(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_of(runtime: RuntimeState) -> dict[str, Any]:
    """快照缓存挂在 runtime 上:``serialize_state`` 整体跳过 runtime,不污染 checkpoint。

    拿不到可写缓存时退回一个一次性 dict——退化成「每次都重采」而不是抛错。
    """
    cache = getattr(runtime, "status_cache", None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    try:
        runtime.status_cache = cache
    except (AttributeError, TypeError):
        pass
    return cache
