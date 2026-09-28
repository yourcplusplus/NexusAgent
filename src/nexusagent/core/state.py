from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NotRequired, TypedDict

from nexusagent.core.approval import ApprovalDecision, ApprovalRequest, normalize_approval_mode
from nexusagent.core.checkpoint import normalize_checkpoint_mode
from nexusagent.core.trace import normalize_trace_mode

MAX_TOUCHED_FILES = 50
MAX_ARTIFACTS = 100
ARTIFACT_OP = "artifact"


class FileEntry(TypedDict):
    """白名单里的文件足迹:path 为 workspace 相对 posix 路径,at 为完整 UTC ISO 时间戳。

    meta 为可选补充元数据,落盘产物用它携带行数/字节/来源,避免渲染时读盘。
    """

    path: str
    op: str
    at: str
    via: str
    meta: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    mtime_ns: int
    complete: bool


@dataclass
class RuntimeState:
    workspace: Path
    read_files: dict[Path, FileSnapshot] = field(default_factory=dict)
    touched_files: dict[str, FileEntry] = field(default_factory=dict)
    artifacts: dict[str, FileEntry] = field(default_factory=dict)
    # 状态栏快照缓存(TTL 内复用 git 子进程结果与工作区清单)。它是运行时缓存而非
    # 配置,且 serialize_state 整体跳过 runtime,所以不会进入 checkpoint 载荷。
    status_cache: dict[str, Any] = field(default_factory=dict)
    approval_mode: str = "inline"
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision | bool] | None = None
    bash_default_timeout_seconds: int = 120
    bash_max_timeout_seconds: int = 600
    bash_max_output_chars: int = 6000
    bash_env_file: Path | None = None
    checkpoint_mode: str = "light"
    resume_from: Path | None = None
    trace_mode: str = "on"
    trace_id: str | None = None

    def __post_init__(self) -> None:
        self.approval_mode = normalize_approval_mode(self.approval_mode)
        self.checkpoint_mode = normalize_checkpoint_mode(self.checkpoint_mode)
        self.trace_mode = normalize_trace_mode(self.trace_mode)

    def record_read(self, path: Path, *, complete: bool) -> None:
        stat = path.stat()
        resolved = path.resolve()
        self.read_files[resolved] = FileSnapshot(
            path=resolved,
            mtime_ns=stat.st_mtime_ns,
            complete=complete,
        )

    def snapshot_for(self, path: Path) -> FileSnapshot | None:
        return self.read_files.get(path.resolve())

    def record_touch(
        self,
        path: str | Path,
        *,
        op: str,
        via: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        """登记一次文件足迹,供 critical_context 白名单渲染。

        op == ARTIFACT_OP 的落盘产物进 artifacts(上限 MAX_ARTIFACTS),其余
        工作集足迹进 touched_files(上限 MAX_TOUCHED_FILES)。路径归一化为
        workspace 相对 posix 字符串;重复触碰同一路径会刷新记录并移到队尾
        (LRU),超出上限时淘汰该字典中最久未触碰的条目。meta 为可选补充
        元数据(如落盘产物的行数/字节/来源)。
        """
        if op == ARTIFACT_OP:
            store, limit = self.artifacts, MAX_ARTIFACTS
        else:
            store, limit = self.touched_files, MAX_TOUCHED_FILES
        key = self.relative_posix(path)
        store.pop(key, None)
        entry = FileEntry(
            path=key,
            op=op,
            at=datetime.now(timezone.utc).isoformat(),
            via=via,
        )
        if meta:
            entry["meta"] = meta
        store[key] = entry
        while len(store) > limit:
            store.pop(next(iter(store)))

    def relative_posix(self, path: str | Path) -> str:
        """workspace 相对 posix 路径;workspace 之外保留原样(不抛错)。"""
        raw = Path(str(path).replace("\\", "/"))
        candidate = raw if raw.is_absolute() else self.workspace / raw
        try:
            return candidate.resolve().relative_to(self.workspace.resolve()).as_posix()
        except ValueError:
            return raw.as_posix()

    def assert_workspace_path(self, path: Path) -> Path:
        resolved = path.resolve()
        workspace = self.workspace.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"path must stay inside workspace: {workspace}")
        return resolved
