from __future__ import annotations

from datetime import datetime
from typing import Any

from nexusagent.core.state import RuntimeState
from nexusagent.tools.file_tools import read_text_lossy

NOTEPAD_FILE = "NOTEPAD.md"


def read_notepad(state: RuntimeState) -> dict[str, Any]:
    path = state.assert_workspace_path(state.workspace / NOTEPAD_FILE)
    if not path.exists():
        return {"ok": True, "path": NOTEPAD_FILE, "content": "", "exists": False}
    content = read_text_lossy(path)
    state.record_read(path, complete=True)
    return {"ok": True, "path": NOTEPAD_FILE, "content": content, "exists": True}


def append_notepad(state: RuntimeState, heading: str, content: str) -> dict[str, Any]:
    if not content.strip():
        return {"ok": False, "error": "content must not be empty"}
    path = state.assert_workspace_path(state.workspace / NOTEPAD_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_text_lossy(path) if path.exists() else "# NexusAgent Notepad\n"
    title = heading.strip() or "Note"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {title}\n\n_Recorded: {timestamp}_\n\n{content.strip()}\n"
    updated = existing.rstrip() + "\n" + entry
    path.write_text(updated, encoding="utf-8")
    state.record_read(path, complete=True)
    return {"ok": True, "path": NOTEPAD_FILE, "heading": title, "lines": len(updated.splitlines())}
