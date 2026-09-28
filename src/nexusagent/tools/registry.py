from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from nexusagent.core.state import RuntimeState
from nexusagent.tools.bash_tool import bash_tool_description, run_bash
from nexusagent.tools.file_tools import edit_file, read_file, write_file
from nexusagent.tools.grep_tool import grep
from nexusagent.tools.notepad_tool import append_notepad, read_notepad
from nexusagent.tools.web_search_tool import build_web_search_tool

# 模型偶发用训练语料里的通用工具名发起调用。两组各 10 任务的真实评测里,
# 幻觉名全部是 "Bash"(B 组 43 次 / A 组 31 次,无第二名字),故别名表只收
# 实证出现过的条目——空泛的"顺手多加几个"只会掩盖未来的新幻觉。
TOOL_ALIASES = {"Bash": "BashTool"}


def resolve_tool_name(name: str) -> str:
    """执行侧把幻觉工具名归一到真实注册名;未知名原样返回(由调用方报 unknown)。"""
    return TOOL_ALIASES.get(str(name or ""), str(name or ""))


def note_alias(result: Any, requested_name: str) -> None:
    """别名命中时在成功结果里注明真名,让模型当轮就修正后续调用。

    只在成功时加注:失败结果不该被这行提示冲淡(它本身已带 error)。
    """
    resolved = resolve_tool_name(requested_name)
    if resolved == requested_name or not isinstance(result, dict) or result.get("ok") is not True:
        return
    result.setdefault("note", f"tool '{requested_name}' resolved to '{resolved}'; call '{resolved}' directly next time")


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            func=lambda file_path, offset=0, limit=2000: read_file(state, file_path, offset, limit),
            description=(
                "Read a UTF-8 text file inside the workspace. Content lines are prefixed "
                "with 1-based line numbers ('<n>\\t<line>'). Supports offset and limit; "
                "a truncated read returns next_offset to continue from."
            ),
        ),
        StructuredTool.from_function(
            name="FileWriteTool",
            func=lambda file_path, content: write_file(state, file_path, content),
            description=(
                "Create a new file, or fully rewrite an existing file you have read this "
                "session. Overwriting an unread or changed-since-read file fails with "
                "error_code NOT_READ / STALE_READ; re-read the file first."
            ),
        ),
        StructuredTool.from_function(
            name="FileEditTool",
            func=lambda file_path, old_text, new_text: edit_file(state, file_path, old_text, new_text),
            description=(
                "Edit an existing workspace file by replacing exactly one unique old_text "
                "snippet (must occur once). File must be read and unchanged since. Failures "
                "return structured error_code: NOT_READ / STALE_READ / NOT_FOUND / "
                "AMBIGUOUS_MATCH, with occurrences (line numbers) or near_matches to guide retry."
            ),
        ),
        StructuredTool.from_function(
            name="GrepTool",
            func=lambda pattern, path=".", glob=None, head_limit=50, ignore_case=False: grep(
                state, pattern, path, glob, head_limit, ignore_case
            ),
            description="Search workspace text files by regex pattern and return matching lines.",
        ),
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=None, run_in_background=False: run_bash(
                state, command, timeout_seconds, run_in_background
            ),
            description=bash_tool_description(),
        ),
        StructuredTool.from_function(
            name="NotepadReadTool",
            func=lambda: read_notepad(state),
            description="Read the durable workspace notepad from NOTEPAD.md.",
        ),
        StructuredTool.from_function(
            name="NotepadAppendTool",
            func=lambda heading, content: append_notepad(state, heading, content),
            description="Append a durable markdown note to NOTEPAD.md. Args: heading, content.",
        ),
    ]


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            func=lambda file_path, offset=0, limit=2000: read_file(state, file_path, offset, limit),
            description=(
                "Read a UTF-8 text file inside the workspace. Content lines are prefixed "
                "with 1-based line numbers ('<n>\\t<line>'). Supports offset and limit; "
                "a truncated read returns next_offset to continue from."
            ),
        ),
        StructuredTool.from_function(
            name="GrepTool",
            func=lambda pattern, path=".", glob=None, head_limit=50, ignore_case=False: grep(
                state, pattern, path, glob, head_limit, ignore_case
            ),
            description="Search workspace text files by regex pattern and return matching lines.",
        ),
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=None, run_in_background=False: run_bash(
                state, command, timeout_seconds, run_in_background
            ),
            description=bash_tool_description(),
        ),
        StructuredTool.from_function(
            name="NotepadReadTool",
            func=lambda: read_notepad(state),
            description="Read the durable workspace notepad from NOTEPAD.md.",
        ),
        build_web_search_tool(),
    ]
