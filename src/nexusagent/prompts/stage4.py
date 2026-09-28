# 阶段 4:上下文压缩(滑动窗口 Rollup)。
CONTEXT_ROLLUP_PROMPT = """You are the sliding-window rollup step in NexusAgent.

The oldest part of the message transcript is being EVICTED to keep the window
small. Summarize ONLY the evicted messages into one incremental narrative entry,
so work can continue without them.

Input keys:
- evicted_messages: the messages being dropped (content already truncated)
- current_narrative: existing narrative, for continuity ONLY

Rules:
- Do NOT restate current_narrative. Produce only the new entry for this eviction.
- Report what was attempted, what the results were, and what is still open.
- Keep artifact / stdout_path / stderr_path pointers VERBATIM. They are how the
  full output is read back after eviction; a dropped or paraphrased pointer
  loses the data permanently.
- Keep failing command exit codes and unresolved errors.
- Preserve CONCRETE FACTS VERBATIM, never summarize them into categories:
  exact file paths, literal string/number constants, config values, token-like
  strings, names and identifiers seen in the evicted messages. If the evicted
  content contains short literal values (keys, codes, single characters),
  copy them exactly. Concrete values are unrecoverable after eviction; prose
  about them is not.
- Do not invent facts that are absent from evicted_messages.
- Output ONE short line: no preamble, no bullet list, no markdown fences.

Return only JSON: {"summary": "<one-line incremental narrative entry>"}
"""
