from __future__ import annotations

import json

from nexusagent.core.session import (
    SESSION_SUMMARY_FILE,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
    session_file,
)


def test_session_create_save_and_context(tmp_path) -> None:
    session = load_or_create_session(tmp_path)
    turn = append_user_turn(session, "帮我创建一个 app.py")
    append_assistant_turn(session, turn=turn, route="workflow", content="created app.py")
    save_session(tmp_path, session)

    assert session_file(tmp_path).exists()
    assert (tmp_path / SESSION_SUMMARY_FILE).exists()

    saved = json.loads(session_file(tmp_path).read_text(encoding="utf-8"))
    assert saved["turn_index"] == 1
    assert saved["last_route"] == "workflow"

    context = build_session_context(tmp_path, saved)
    assert "帮我创建一个 app.py" in context
    assert "created app.py" in context


def test_session_history_is_compacted(tmp_path) -> None:
    session = load_or_create_session(tmp_path)

    for idx in range(25):
        turn = append_user_turn(session, f"user turn {idx}")
        append_assistant_turn(session, turn=turn, route="chat", content=f"assistant turn {idx}")

    save_session(tmp_path, session)
    saved = json.loads(session_file(tmp_path).read_text(encoding="utf-8"))

    assert len(saved["recent_turns"]) <= 18
    assert "user turn 0" in saved["summary"]
