from __future__ import annotations

import os
from pathlib import Path

from nexusagent.core.state import RuntimeState
from nexusagent.tools.file_tools import edit_file, read_file, write_file


def make_state(tmp_path: Path) -> RuntimeState:
    return RuntimeState(workspace=tmp_path)


# ---------- FileReadTool ----------


def test_read_file_aligns_line_numbers_and_reports_next_offset(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "doc.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = read_file(state, "doc.md", offset=0, limit=2)

    assert result["ok"] is True
    lines = result["content"].splitlines()
    assert lines[0] == f"{1:>6}\talpha"
    assert lines[1] == f"{2:>6}\tbeta"
    assert result["complete"] is False
    assert result["next_offset"] == 2


def test_read_file_complete_read_has_no_next_offset(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "doc.md").write_text("alpha\n", encoding="utf-8")

    result = read_file(state, "doc.md")

    assert result["ok"] is True
    assert result["complete"] is True
    assert "next_offset" not in result


# ---------- FileWriteTool ----------


def test_write_file_unread_returns_structured_not_read(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "app.py").write_text("print('old')\n", encoding="utf-8")

    result = write_file(state, "app.py", "print('new')\n")

    assert result["ok"] is False
    assert result["error_code"] == "NOT_READ"
    assert "not been read" in result["error"]
    assert "FileReadTool" in result["hint"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "print('old')\n"


def test_write_file_stale_read_after_external_change(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    target = tmp_path / "app.py"
    target.write_text("print('v1')\n", encoding="utf-8")
    assert read_file(state, "app.py")["ok"] is True
    target.write_text("print('v2')\n", encoding="utf-8")
    os.utime(target, ns=(0, 0))

    result = write_file(state, "app.py", "print('v3')\n")

    assert result["ok"] is False
    assert result["error_code"] == "STALE_READ"
    assert target.read_text(encoding="utf-8") == "print('v2')\n"


def test_write_file_rejects_directory_target(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "adir").mkdir()

    result = write_file(state, "adir", "x")

    assert result["ok"] is False
    assert result["error_code"] == "NOT_A_FILE"


def test_write_file_reports_write_failed_and_keeps_file(monkeypatch, tmp_path: Path) -> None:
    state = make_state(tmp_path)
    target = tmp_path / "app.py"
    target.write_text("keep\n", encoding="utf-8")
    assert read_file(state, "app.py")["ok"] is True

    def boom(self: Path, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    result = write_file(state, "app.py", "new\n")

    assert result["ok"] is False
    assert result["error_code"] == "WRITE_FAILED"
    assert "disk full" in result["error"]
    assert target.read_text(encoding="utf-8") == "keep\n"


# ---------- FileEditTool ----------


def test_edit_file_unread_returns_structured_not_read(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "app.py").write_text("print('old')\n", encoding="utf-8")

    result = edit_file(state, "app.py", "old", "new")

    assert result["ok"] is False
    assert result["error_code"] == "NOT_READ"
    assert "FileReadTool" in result["hint"]


def test_edit_file_rejects_empty_old_text(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "app.py").write_text("print('x')\n", encoding="utf-8")
    assert read_file(state, "app.py")["ok"] is True

    result = edit_file(state, "app.py", "", "new")

    assert result["ok"] is False
    assert result["error_code"] == "EMPTY_OLD_TEXT"


def test_edit_file_ambiguous_match_lists_occurrence_lines(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    target = tmp_path / "dup.txt"
    target.write_text("header\nTODO fix\nmiddle\nTODO fix\nfooter\n", encoding="utf-8")
    assert read_file(state, "dup.txt")["ok"] is True

    result = edit_file(state, "dup.txt", "TODO fix", "TODO done")

    assert result["ok"] is False
    assert result["error_code"] == "AMBIGUOUS_MATCH"
    assert "matched 2 times" in result["error"]
    assert result["occurrences"][0]["line"] == 2
    assert result["occurrences"][0]["preview"] == "TODO fix"
    assert result["occurrences"][1]["line"] == 4
    assert "TODO fix" in target.read_text(encoding="utf-8")


def test_edit_file_not_found_detects_indentation_mismatch(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "code.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert read_file(state, "code.py")["ok"] is True

    # 文件里是 4 空格缩进,old_text 写成 8 空格 → 匹配失败但去空白后一致
    result = edit_file(state, "code.py", "        return 1", "        return 2")

    assert result["ok"] is False
    assert result["error_code"] == "NOT_FOUND"
    assert result["likely_cause"] == "whitespace_or_indentation_mismatch"
    assert "near_matches" not in result


def test_edit_file_not_found_offers_near_matches(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "notes.md").write_text("the quick brown fox\njumps over\n", encoding="utf-8")
    assert read_file(state, "notes.md")["ok"] is True

    result = edit_file(state, "notes.md", "the quik brown fox", "x")

    assert result["ok"] is False
    assert result["error_code"] == "NOT_FOUND"
    assert "the quick brown fox" in result["near_matches"]
