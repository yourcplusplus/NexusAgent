from __future__ import annotations

import json
import random
import re
from pathlib import Path

from nexusagent.core.state import RuntimeState
from nexusagent.tools.bash_tool import _preview_output, run_bash
from nexusagent.tools.output_sink import spill_tool_output


def make_state(tmp_path: Path) -> RuntimeState:
    return RuntimeState(workspace=tmp_path)


def inline_size(payload: dict) -> int:
    """调用方塞进 ToolMessage 的实际字节规模。"""
    return len(json.dumps(payload, ensure_ascii=False))


def big_stdout(line_count: int = 2500) -> str:
    return "\n".join(f"line {i:05d} " + "x" * 30 for i in range(line_count))


# ---------- 短结果:零开销直通 ----------


def test_short_result_is_returned_unchanged_same_object(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    result = {"ok": True, "exit_code": 0, "stdout": "hello"}

    spilled = spill_tool_output(state, "BashTool", result)

    assert spilled is result
    assert not (tmp_path / ".nexusagent" / "tool-outputs").exists()


def test_non_dict_result_is_returned_unchanged(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    payload = ["not", "a", "dict"]

    assert spill_tool_output(state, "BashTool", payload) is payload


# ---------- 超长输出落盘 ----------


def test_large_stdout_spills_and_keeps_head_and_tail(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    stdout = big_stdout()
    assert len(stdout) > 100_000

    spilled = spill_tool_output(
        state,
        "BashTool",
        {"ok": True, "exit_code": 0, "command": "pytest -x", "stdout": stdout, "stderr": ""},
    )

    assert inline_size(spilled) <= 2000
    # 头尾都保留,中间标出省略行数
    assert spilled["stdout"].startswith("line 00000")
    assert "line 02499" in spilled["stdout"]
    assert "lines omitted" in spilled["stdout"]
    # 指针:结构化字段 + 文本尾部
    assert spilled["artifact_path"].startswith(".nexusagent/tool-outputs/")
    assert spilled["artifact_path"] in spilled["stdout"]
    # 短标量字段原样保留
    assert spilled["ok"] is True
    assert spilled["exit_code"] == 0
    assert spilled["command"] == "pytest -x"

    # 盘上文件真实存在,且能被 BashTool 的 tail 回读
    assert (tmp_path / spilled["artifact_path"]).exists()
    tailed = run_bash(state, f"tail -5 {spilled['artifact_path']}", timeout_seconds=5)
    assert tailed["ok"] is True
    assert tailed["stdout"].strip()


def test_artifact_file_is_valid_json_with_same_keys(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    result = {"ok": True, "exit_code": 0, "command": "ls", "stdout": big_stdout(400)}

    spilled = spill_tool_output(state, "BashTool", result)

    data = json.loads((tmp_path / spilled["artifact_path"]).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert set(data) == set(result)
    # 落盘的是全文,不是摘要
    assert data["stdout"] == result["stdout"]


def test_single_huge_line_stays_within_inline_budget(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    spilled = spill_tool_output(state, "FileReadTool", {"ok": True, "content": "y" * 40000})

    assert inline_size(spilled) <= 2000
    assert spilled["artifact_path"] in spilled["content"]


def test_grep_matches_list_is_flattened_within_budget(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    matches = [{"path": f"f{i}.py", "line": i, "text": "z" * 80} for i in range(500)]

    spilled = spill_tool_output(
        state,
        "GrepTool",
        {"ok": True, "pattern": "TODO", "matches": matches, "truncated": True},
    )

    assert inline_size(spilled) <= 2000
    assert isinstance(spilled["matches"], str)
    assert spilled["pattern"] == "TODO"


# ---------- 路径穿越防护 ----------


def test_tool_name_cannot_escape_workspace(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    spilled = spill_tool_output(state, "../../evil", {"ok": True, "stdout": "q" * 5000})

    artifact_path = spilled["artifact_path"]
    assert ".." not in artifact_path
    assert artifact_path.startswith(".nexusagent/tool-outputs/")
    assert re.fullmatch(r"evil-\d+\.json", Path(artifact_path).name)
    resolved = (tmp_path / artifact_path).resolve()
    assert resolved.is_relative_to(tmp_path.resolve())
    assert resolved.exists()
    assert not list(tmp_path.parent.glob("evil-*.json"))


# ---------- _preview_output 契约 ----------


def test_preview_output_passes_short_text_through() -> None:
    text = "hello\nworld\n"

    assert _preview_output(text, max_chars=6000, artifact_path="p.log") == text


def test_preview_output_small_budget_degrades_to_char_truncation() -> None:
    """极小预算下退化为纯字符截断(test_tools 的 max_output_chars=10 场景依赖此行为)。"""
    preview = _preview_output("x" * 50 + "\n", max_chars=10, artifact_path=".nexusagent/bash-outputs/s.log")

    assert preview == "x" * 10


def test_preview_output_reports_exact_omitted_line_count() -> None:
    line_count = 5000
    text = "\n".join(f"line {i:05d} " + "y" * 30 for i in range(line_count))

    preview = _preview_output(text, max_chars=6000, artifact_path=".nexusagent/bash-outputs/s.log")

    lines = preview.splitlines()
    marker = next(line for line in lines if "lines omitted" in line)
    omitted = int(re.search(r"(\d+) lines omitted", marker).group(1))
    shown = len(lines) - 2  # 去掉省略提示行与落盘路径行

    assert preview.startswith("line 00000")
    assert f"line {line_count - 1:05d}" in preview
    assert shown == 50  # 头 30 + 尾 20
    assert omitted == line_count - shown


def test_preview_output_respects_char_budget_on_random_inputs() -> None:
    rng = random.Random(20260926)
    budgets = (1, 2, 10, 20, 50, 200, 1000, 6000)

    for _ in range(1000):
        line_count = rng.randint(1, 400)
        text = "\n".join("q" * rng.randint(1, 300) for _ in range(line_count))
        max_chars = rng.choice(budgets)
        artifact_path = rng.choice([None, ".nexusagent/bash-outputs/stdout-1.log"])

        preview = _preview_output(text, max_chars=max_chars, artifact_path=artifact_path)

        assert len(preview) <= max_chars, (line_count, max_chars, artifact_path, len(preview))


def test_bash_tool_truncated_output_shows_head_and_tail(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path, bash_max_output_chars=400)
    (tmp_path / "emit.py").write_text(
        "for i in range(200):\n    print('line ' + str(i).zfill(3))\n",
        encoding="utf-8",
    )

    result = run_bash(state, "python emit.py", timeout_seconds=10)

    assert result["ok"] is True
    assert result["stdout_truncated"] is True
    assert len(result["stdout"]) <= 400
    assert result["stdout"].startswith("line 000")
    assert "line 199" in result["stdout"]
    assert "lines omitted" in result["stdout"]
    assert result["stdout_path"] in result["stdout"]
    assert (tmp_path / result["stdout_path"]).exists()
