# NexusAgent 项目篇视频文稿

> 从零到一，手搓一个 Code Agent —— 六个阶段，六次进化

---

## 项目概览

### 核心目标

实现一个自己的 Code Agent，类似 Claude Code 的 mini 版本。但重点不是功能对齐，而是**循序渐进地增加复杂度**，每一步都配合视频讲解，让你真正理解每个设计决策背后的原因。

### 技术栈

| 组件 | 选型 | 为什么 |
|------|------|--------|
| LLM 调用 | `langchain` + `langchain-openai` | 工具绑定、消息管理开箱即用 |
| 工作流引擎 | `langgraph` | 状态图 + 条件路由，天然适合 Plan→Execute→Verify 循环 |
| CLI 框架 | `typer` | 类型安全的命令行参数 |
| TUI 界面 | `textual` | 现代终端 UI，支持异步事件 |
| 搜索 | `tavily-python` | WebSearch 工具后端 |

### 项目架构（最终形态）

```
src/nexusagent/
├── agents/              # 多 Agent 实现
│   ├── code_agent.py    # 代码实现专家
│   └── search_agent.py  # 网络搜索专家
├── cli/                 # 用户交互层
│   ├── app.py           # typer CLI 入口
│   ├── formatter.py     # 输出格式化
│   ├── event_summary.py # 事件摘要
│   └── tui/             # textual TUI 界面
│       ├── app.py       # 主 TUI 应用
│       ├── approval.py  # 人类审批弹窗
│       └── logo.py      # Logo 渲染
├── core/                # 核心基础设施
│   ├── agent.py         # 工作流编排 & 事件流
│   ├── approval.py      # 人类审批机制
│   ├── checkpoint.py    # 断点保存 & 恢复
│   ├── session.py       # 多轮会话管理
│   ├── state.py         # 运行时状态
│   ├── paths.py         # 路径工具
│   └── trace.py         # 链路追踪 & 日志
├── graph/               # LangGraph 状态图
│   ├── workflow.py      # 图构建（入口 + 复杂流）
│   ├── nodes.py         # 所有图节点
│   ├── state.py         # 图状态定义
│   └── memory.py        # 分层记忆 & 压缩
├── prompts/             # 各阶段 Prompt
│   ├── stage2.py        # Plan & Execute
│   ├── stage3.py        # MultiAgent
│   └── stage4.py        # Context Compression
├── providers/           # LLM 提供商
│   └── openai_provider.py
└── tools/               # 工具集
    ├── registry.py      # 工具注册
    ├── bash_tool.py     # Shell 命令
    ├── file_tools.py    # 文件读写编辑
    ├── grep_tool.py     # 文本搜索
    ├── notepad_tool.py  # 持久化笔记
    ├── todo_tool.py     # 任务管理
    └── web_search_tool.py # 网络搜索
```

---

## 阶段一：ReAct 实现基础功能

### 🎯 设计目标

用最简单的 ReAct（Reasoning + Acting）循环，让 Agent 能**听懂指令、创建文件、执行代码**。这是整个项目的地基。

### 🏗️ 架构设计

**ReAct 循环**是最朴素的 Agent 模式：模型思考 → 调用工具 → 观察结果 → 再思考，如此循环直到任务完成。

```
┌──────────────────────────────────────┐
│           User Task                   │
│  "帮我创建一个贪吃蛇游戏并执行"         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│         Actor (ReAct Loop)            │
│                                      │
│  System Prompt + Task                │
│         │                            │
│         ▼                            │
│   ┌───────────┐    ┌──────────────┐  │
│   │  LLM Think │───▶│  Tool Call   │  │
│   └───────────┘    └──────┬───────┘  │
│         ▲                  │         │
│         │                  ▼         │
│   ┌───────────┐    ┌──────────────┐  │
│   │  Observe   │◀───│ Tool Result  │  │
│   └───────────┘    └──────────────┘  │
│         │                            │
│         ▼                            │
│   (loop until no tool calls)         │
└──────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│          Final Answer                 │
└──────────────────────────────────────┘
```

**Agent 节点：**
- **actor**：拥有 FileReadTool、FileWriteTool、FileEditTool、GrepTool、BashTool

**工具清单：**
| 工具 | 功能 | 为什么需要 |
|------|------|-----------|
| FileReadTool | 读取工作区文件 | Agent 需要先读再改 |
| FileWriteTool | 创建/覆写文件 | 生成代码文件 |
| FileEditTool | 精确替换文件片段 | 局部修改而非全量重写 |
| GrepTool | 正则搜索文件内容 | 定位代码位置 |
| BashTool | 执行 Shell 命令 | 运行代码、安装依赖 |

### 📝 Prompt 设计要点

ReAct 阶段的 Actor Prompt 需要告诉模型三件事：

1. **你是谁**：你是 NexusAgent 的 actor 节点，负责实现代码
2. **你可以用什么**：列出工具及使用规则
3. **你应该怎么做**：
   - 先想清楚再动手
   - 用 FileWriteTool 创建新文件
   - 用 FileReadTool 先读再改
   - 用 FileEditTool 做局部修改
   - 用 BashTool 运行和测试代码
   - 工作区路径用相对路径
   - 结尾给一个简洁的总结

```python
ACTOR_PROMPT = """You are the actor node in NexusAgent's LangGraph workflow.

You implement the current plan using tools. Work inside the workspace only.

Rules:
- Before starting work for a todo, call TodoUpdateTool with status "in_progress".
- After finishing that todo, call TodoUpdateTool with status "completed".
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool to run tests and demos.
- BashTool already runs inside the workspace. Never run "cd /workspace".
- End with a concise summary of files changed and commands run.
"""
```

### 🎬 演示效果

```bash
nexusagent "帮我创建一个简易的贪吃蛇游戏代码，并执行出来给我看"
```

Agent 会：
1. 思考 → 调用 FileWriteTool 创建 `snake.py`
2. 思考 → 调用 BashTool 执行 `python snake.py`
3. 观察输出 → 如有报错则思考修复 → 调用 FileEditTool 修改
4. 最终返回结果总结

### 💡 讲解要点

- **ReAct 的本质**：就是一个 while 循环，直到模型不再调用工具
- **Tool Binding**：LangChain 的 `model.bind_tools()` 如何把 Python 函数变成 LLM 可调用的工具
- **消息流**：SystemMessage → HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(...) → Final AIMessage
- **工作区隔离**：所有文件操作都在 workspace 目录内，避免污染宿主系统

### 🔍 核心代码

**工具注册 — [registry.py:13](src/nexusagent/tools/registry.py#L13)**

```python
def build_tools(state: RuntimeState) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(name="FileReadTool",  func=lambda file_path, offset=0, limit=2000: read_file(state, file_path, offset, limit), ...),
        StructuredTool.from_function(name="FileWriteTool", func=lambda file_path, content: write_file(state, file_path, content), ...),
        StructuredTool.from_function(name="FileEditTool",  func=lambda file_path, old_text, new_text: edit_file(state, file_path, old_text, new_text), ...),
        StructuredTool.from_function(name="GrepTool",      func=lambda pattern, path=".", ...: grep(state, pattern, path, ...), ...),
        StructuredTool.from_function(name="BashTool",      func=lambda command, timeout_seconds=None, ...: run_bash(state, command, ...), ...),
        StructuredTool.from_function(name="NotepadReadTool",  func=lambda: read_notepad(state), ...),
        StructuredTool.from_function(name="NotepadAppendTool", func=lambda heading, content: append_notepad(state, heading, content), ...),
    ]
```

> 每个工具闭包捕获 `state`（含 workspace 路径），所有文件操作限制在 workspace 内。`model.bind_tools(build_tools(state))` 把这些函数变成 LLM 可调用的 tool schema。

### 🤖 Vibe Coding Prompt

> 以下是你跟着视频一起写代码时，输入给 AI 编程助手的 Prompt。直接复制使用即可。

**Step 1: 初始化项目结构**

```
帮我创建一个 Python 项目 NexusAgent，用 typer 做 CLI。项目结构如下：

src/nexusagent/
├── __init__.py
├── __main__.py
├── core/
│   ├── __init__.py
│   ├── state.py          # RuntimeState 数据类
│   └── paths.py          # 工作区路径工具
├── tools/
│   ├── __init__.py
│   ├── registry.py       # build_tools 注册函数
│   ├── file_tools.py     # FileReadTool, FileWriteTool, FileEditTool
│   ├── grep_tool.py      # GrepTool
│   └── bash_tool.py      # BashTool
├── providers/
│   ├── __init__.py
│   └── openai_provider.py  # create_model() 工厂函数
└── cli/
    ├── __init__.py
    └── app.py            # typer 入口，nexusagent 命令

核心要求：
1. RuntimeState 包含 workspace: Path 字段，所有工具操作限制在 workspace 内
2. FileReadTool(file_path, offset, limit) — 读取文件，先做路径安全检查
3. FileWriteTool(file_path, content) — 创建/覆写文件
4. FileEditTool(file_path, old_text, new_text) — 替换唯一文本片段，匹配多个则报错
5. GrepTool(pattern, path, glob, head_limit, ignore_case) — 正则搜索
6. BashTool(command, timeout_seconds) — 在 workspace 内执行命令，有超时控制
7. build_tools(state) 返回 StructuredTool 列表，供 model.bind_tools() 使用
8. create_model() 使用 langchain_openai 的 ChatOpenAI，从 .env 读取 OPENAI_API_KEY
9. CLI 入口：nexusagent <task> --workspace <path>，workspace 默认自动创建
```

**Step 2: 实现 ReAct 循环**

```
在 src/nexusagent/core/agent.py 中实现一个 ReAct 循环：

def stream_agent_events(task, *, workspace, max_loops=10) -> Iterator[dict]:
    """
    1. 创建 RuntimeState(workspace)
    2. 构建消息列表：SystemMessage(ACTOR_PROMPT) + HumanMessage(task)
    3. model.bind_tools(build_tools(state))
    4. for _ in range(max_loops):
         response = agent.invoke(messages)
         messages.append(response)
         yield {"type": "ai_message", "content": ...}
         if no tool_calls: break
         for call in tool_calls:
             yield {"type": "tool_call", "name": ..., "args": ...}
             result = execute_tool(call)
             tool_message = ToolMessage(content=json.dumps(result), ...)
             messages.append(tool_message)
             yield {"type": "tool_result", "name": ..., "result": ...}
    5. yield {"type": "final_answer", "content": last_ai_content}
    """

ACTOR_PROMPT 的内容：
"""You are the actor node in NexusAgent's ReAct workflow.

You implement the user's task using tools. Work inside the workspace only.

Rules:
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool to run commands and test results.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- End with a concise summary of files changed and commands run.
"""

CLI 层在 app.py 中调用 stream_agent_events，用 rich 实时打印每个事件。
事件类型：tool_call → 显示工具名和参数，tool_result → 显示结果，final_answer → 显示最终回答
```

**Step 3: 测试运行**

```bash
nexusagent "帮我创建一个简易的贪吃蛇游戏代码，并执行出来给我看" -w ./workspace
```

---

## 阶段二：改为 LangGraph —— Plan → Execute → Verify

### 🎯 设计目标

ReAct 循环太"盲目"了——Agent 想到哪做到哪，没有规划，没有验证。引入 LangGraph，实现 **计划 → 执行 → 检查** 的结构化循环。

### 🏗️ 架构设计

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Planner   │  ← 制定计划、写验收标准
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Actor    │  ← 按计划执行，使用工具
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Verifier   │  ← 运行验收命令，判断是否通过
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   passed?   │
                    └──┬──────┬───┘
                  yes  │      │  no
                       ▼      ▼
                ┌─────────┐  ┌─────────┐
                │  Final   │  │ Planner │  ← 重新规划修复方案
                └────┬────┘  └─────────┘
                     │            │
                     ▼            │
                ┌─────────┐      │
                │   END   │◀─────┘
                └─────────┘  (max_attempts 次后也到 END)
```

**LangGraph 核心概念：**
- **StateGraph**：用 `NexusGraphState`（TypedDict）定义图的共享状态
- **add_node**：每个节点是一个函数，接收 state，返回 state 更新
- **add_conditional_edges**：根据 state 的某个字段决定路由方向
- **compile()**：编译为可执行的图

**Agent 节点：**
| 节点 | 职责 | 工具 |
|------|------|------|
| Planner | 分析任务、制定计划、定义验收标准 | TodoWriteTool |
| Actor | 按计划逐步执行、创建/修改文件 | FileReadTool、FileWriteTool、FileEditTool、GrepTool、BashTool |
| Verifier | 运行验收命令、检查产出物、判断通过/失败 | BashTool、FileReadTool、GrepTool |

### 📝 Prompt 设计

**Planner Prompt** — 把用户任务转化为结构化计划：

```python
PLANNER_PROMPT = """You are the planner node in NexusAgent's LangGraph workflow.

Your job is to turn the user's task into a concrete engineering plan. Return a
compact JSON object with these keys:
- plan_summary: short summary of the implementation goal
- todos: list of concrete todo strings
- acceptance_criteria: list of requirements the verifier can judge
- verification_commands: list of shell commands to run inside the workspace

Rules:
- Prefer TDD for coding tasks: write tests first, then implementation, then demo.
- Verification commands must be cross-platform Python commands when possible.
"""
```

**Verifier Prompt** — 判断任务是否完成：

```python
VERIFIER_PROMPT = """You are verifier, a model-based reviewer node.

You decide whether the user's task is complete by inspecting state and using
read-only tools. Return only JSON with these keys:
  passed: boolean
  reason: short human-readable explanation
  checks: list of {name, passed, detail}
  recommended_next_instruction: what planner should fix, or empty string
"""
```

### 🎬 演示效果

```
任务：帮我实现一个 Conway's Game of Life，要求：
1. 写测试用例
2. 写实现代码
3. 跑通所有测试
4. 跑一个 demo 展示

--- Planner 输出 ---
📋 Plan: Implement Conway's Game of Life with TDD
  ✅ todo-1: Write test_game_of_life.py with edge cases
  ✅ todo-2: Implement game_of_life.py
  ✅ todo-3: Run pytest and fix failures
  ✅ todo-4: Run demo with --demo --steps 3

--- Actor 执行 ---
🔧 FileWriteTool → test_game_of_life.py
🔧 FileWriteTool → game_of_life.py
🔧 BashTool → python -m pytest -q (3 passed)
🔧 BashTool → python game_of_life.py --demo --steps 3

--- Verifier 检查 ---
✅ All checks passed
  - tests_pass: pytest exits 0
  - demo_runs: demo output present
  - rules_correct: blinker and still-life patterns verified

--- Final ---
✅ Task completed: Game of Life implemented with TDD, all tests pass.
```

### 💡 讲解要点

- **为什么 ReAct 不够**：缺少规划 → 容易遗漏；缺少验证 → 不知道做没做对
- **LangGraph 的 StateGraph**：如何把 `TypedDict` 变成图的共享状态
- **条件路由**：`verifier_route` 函数决定 passed → final，failed → planner
- **attempts 机制**：最多重试 `max_attempts` 次，防止无限循环
- **Plan 结构化**：planner 输出 JSON（todos + acceptance_criteria + verification_commands），而不是自由文本

### 🔍 核心代码

**图构建 — [workflow.py:24](src/nexusagent/graph/workflow.py#L24)**

```python
def build_complex_workflow():
    graph = StateGraph(NexusGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("context_monitor", context_monitor_node)   # 阶段四加入
    graph.add_node("context_compressor", context_compressor_node) # 阶段四加入
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges("context_monitor", context_monitor_route, {...})
    graph.add_conditional_edges("context_compressor", context_compressor_route, {...})
    graph.add_edge("verifier", "context_monitor")
    graph.add_edge("final", END)
    return graph.compile()
```

> 阶段二的初始版是 `START → planner → actor → verifier → (passed? → final / planner) → END`。后续阶段逐步插入 context_monitor、context_compressor 节点。

**状态定义 — [state.py:60](src/nexusagent/graph/state.py#L60)**

```python
class NexusGraphState(TypedDict, total=False):
    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]  # ← 自动合并，不覆盖
    plan_summary: str;  todos: list[TodoItem]
    acceptance_criteria: list[str];  verification_commands: list[str]
    passed: bool                    # ← verifier 判断结果
    attempts: int                   # ← 重试次数
    # --- 阶段三: MultiAgent ---
    research_notes: str;  sources: list[SourceItem];  agent_handoffs: list[AgentHandoff]
    code_agent_summary: str;  verifier_summary: str
    # --- 阶段四: Context Engineer ---
    context_should_compress: bool;  context_next_node: str
    compression_events: list[CompressionEvent];  memory_snapshot: LayeredMemory
    # --- 阶段六: Session ---
    intent_route: str;  chat_response: str;  session_id: str;  session_context: str
```

> `messages` 用 `Annotated[list[BaseMessage], add_messages]` 让 LangGraph 自动合并而非覆盖。每个节点返回 dict 只更新需要的字段。

**Verifier 路由 — [nodes.py:412](src/nexusagent/graph/nodes.py#L412)**

```python
def verifier_route(state: NexusGraphState) -> str:
    if state.get("passed"):          return "final"   # 通过 → 结束
    if state.get("attempts", 0) >= state.get("max_attempts", 3):  return "final"   # 达到上限 → 认输
    return "planner"                                   # 失败 → 重试
```

**Verifier 节点 — [nodes.py:221](src/nexusagent/graph/nodes.py#L221)**

```python
def verifier_node(state: NexusGraphState) -> dict[str, Any]:
    verifier = model.bind_tools(build_read_only_tools(state["runtime"]))  # ← 只读工具！
    for _ in range(8):
        response = verifier.invoke(messages)
        if not tool_calls: break
        # ...执行工具，收集结果
    parsed = _extract_json(_last_ai_content(produced_messages))  # ← 解析 JSON
    passed = bool(parsed.get("passed"))
    attempts = state.get("attempts", 0) + 1
    return {"passed": passed, "attempts": attempts, "last_error": ..., "context_next_node": verifier_route({...})}
```

> Verifier 用 `build_read_only_tools`（只有 FileReadTool、GrepTool、BashTool、WebSearchTool，没有写工具），返回 JSON 结构化判断 pass/fail。

### 🤖 Vibe Coding Prompt

**Step 1: 定义图状态**

```
在 src/nexusagent/graph/state.py 中定义 LangGraph 图的共享状态：

class TodoItem(TypedDict):
    id: str
    content: str
    status: str       # "pending" | "in_progress" | "completed" | "blocked"
    note: str

class VerificationResult(TypedDict):
    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str

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

注意 messages 字段用 Annotated[list[BaseMessage], add_messages]，
这样 LangGraph 会自动合并消息而不是覆盖。
```

**Step 2: 实现三个节点**

```
在 src/nexusagent/graph/nodes.py 中实现三个核心节点：

1. planner_node(state) -> dict:
   - 如果 state 没有 todos，用 LLM 生成计划（返回 JSON: plan_summary, todos, acceptance_criteria, verification_commands）
   - 如果已有 todos 且 verifier 失败过，根据 last_error 修订计划
   - 用 model.bind_tools([TodoWriteTool]) 让 LLM 结构化输出
   - 返回 state 更新：plan_summary, todos, acceptance_criteria, verification_commands

2. actor_node(state) -> dict:
   - 用 model.bind_tools(build_tools(state) + [TodoUpdateTool]) 创建 agent
   - 输入：ACTOR_PROMPT + 当前计划 + 任务
   - ReAct 循环（max_loops=10），每一步 yield 事件
   - 返回 state 更新：messages, last_actor_summary

3. verifier_node(state) -> dict:
   - 用 model.bind_tools(build_read_only_tools(state)) 创建只读 agent
   - 输入：VERIFIER_PROMPT + 计划 + 验收标准 + 验证命令 + 最近 actor 输出
   - 返回 JSON: {passed: bool, reason: str, checks: [{name, passed, detail}], recommended_next_instruction: str}
   - 同时运行 verification_commands，收集 VerificationResult
   - 返回 state 更新：passed, attempts(+1), verification_results, verification_checks, last_error(如果失败), todos(更新状态)

路由函数：
- verifier_route(state) -> str:
    if passed: return "final"
    if attempts >= max_attempts: return "final"
    return "planner"
```

**Step 3: 构建工作流图**

```
在 src/nexusagent/graph/workflow.py 中构建 LangGraph 图：

from langgraph.graph import StateGraph, START, END

def build_workflow():
    graph = StateGraph(NexusGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {
        "final": "final",
        "planner": "planner",
    })
    graph.add_edge("final", END)
    return graph.compile()

final_node 只是把 passed/failed 状态格式化为 final_answer 文本。

同时在 src/nexusagent/prompts/stage2.py 中写好：
- PLANNER_PROMPT（返回 JSON 计划）
- ACTOR_PROMPT（按计划执行工具）
- VERIFIER_PROMPT（返回 JSON 验收结果）
- FINAL_PROMPT（总结最终结果）
```

**Step 4: 更新 CLI 和 agent.py**

```
更新 src/nexusagent/core/agent.py：
- 不再直接用 ReAct 循环
- 改为调用 build_workflow().stream(inputs, stream_mode=["updates", "custom"])
- 解析图事件，yield 统一格式的事件给 CLI 层

更新 src/nexusagent/cli/app.py：
- 添加 --max-attempts 参数（默认 3）
- 显示每个节点的输出：📋 Planner, 🔧 Actor, ✅/❌ Verifier, 📝 Final
```

**Step 5: 测试运行**

```bash
nexusagent "帮我实现一个 Conway's Game of Life，要求 TDD：先写测试，再写实现，最后跑 demo" --max-attempts 3
```

---

## 阶段三：MultiAgent —— 专家分工协作

### 🎯 设计目标

一个 Actor 既要写代码又要查资料，能力边界模糊。把 Actor 拆成两个专家：**codeAgent**（专注实现）和 **searchAgent**（专注搜索），由 Planner 作为 Supervisor 协调调度。

### 🏗️ 架构设计

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
               ┌───────────────────────┐
               │     Planner (Supervisor)│
               │  TodoWriteTool          │
               │  CallSearchAgentTool    │
               │  CallCodeAgentTool      │
               └───────┬───────────────┘
                       │
              ┌────────┼────────┐
              │        │        │
              ▼        ▼        ▼
     ┌────────────┐ ┌──────────┐ │
     │ searchAgent │ │ codeAgent│ │
     │ WebSearch   │ │ 文件/Shell │ │
     └──────┬─────┘ └────┬─────┘ │
              │            │      │
              └──────┬─────┘      │
                     │            │
                     ▼            │
              ┌─────────────┐    │
              │   Planner   │◀───┘
              │  (continue) │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Verifier   │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │   passed?   │
              └──┬──────┬───┘
            yes  │      │  no
                 ▼      ▼
           ┌────────┐ ┌─────────┐
           │ Final  │ │ Planner │
           └───┬────┘ └─────────┘
               │           │
               ▼           │
           ┌───────┐       │
           │  END  │◀──────┘
           └───────┘
```

**关键设计 —— Agent Handoff（交接）：**

Planner 通过 `CallSearchAgentTool` 和 `CallCodeAgentTool` 两个工具来调度专家。每次调用就是一次 Handoff：

```python
class AgentHandoff(TypedDict, total=False):
    from_agent: str    # 谁发起的
    to_agent: str      # 交给谁
    instruction: str   # 具体指令
    result: str        # 执行结果摘要
```

这些 Handoff 记录会保存在 State 里，下一个 Agent 可以看到之前的交接上下文。

**Agent 节点：**
| 节点 | 职责 | 工具 |
|------|------|------|
| Planner（Supervisor） | 分析任务、规划、调度专家 | TodoWriteTool、CallSearchAgentTool、CallCodeAgentTool |
| searchAgent | 搜索外部资料 | WebSearchTool |
| codeAgent | 实现/修改代码 | FileReadTool、FileWriteTool、FileEditTool、GrepTool、BashTool、TodoUpdateTool、NotepadAppendTool、NotepadReadTool |
| Verifier | 验证结果 | BashTool（只读）、FileReadTool、GrepTool |

### 📝 Prompt 设计

**Planner Prompt** — 从执行者变成调度者：

```python
PLANNER_PROMPT = """You are the planner/supervisor node in NexusAgent stage 3.

You coordinate specialist agents through tools. You cannot directly edit files
or search the web yourself; delegate specialist work through tool calls.

Available tools:
- TodoWriteTool: publish or revise the plan, todos, acceptance criteria.
- CallSearchAgentTool: delegate web/document research.
- CallCodeAgentTool: delegate file/code implementation.

Rules:
- Always call TodoWriteTool before delegating new work.
- For tasks that require current facts, call CallSearchAgentTool before CallCodeAgentTool.
- If the verifier failed, revise the plan and delegate only the missing fix.
- End with a concise supervisor summary after the needed specialist calls.
"""
```

**SearchAgent Prompt** — 专注搜索，不做实现：

```python
SEARCH_AGENT_PROMPT = """You are searchAgent, a focused research specialist.

Your only external capability is WebSearchTool. Search for reliable information
needed by the planner and codeAgent.

Rules:
- Use WebSearchTool for factual research.
- Prefer official or encyclopedia-style sources.
- Return a concise research summary and list useful source URLs.
- Do not write files or produce application code.
"""
```

**CodeAgent Prompt** — 专注实现，带 Notepad 机制：

```python
CODE_AGENT_PROMPT = """You are codeAgent, a focused implementation specialist.

Rules:
- You must update todo progress explicitly via TodoUpdateTool.
- Use NotepadAppendTool to record durable findings, decisions, important files,
  blockers, and next-step context that should survive compression.
- Use NotepadReadTool when you need to recover prior notes.
- Incorporate research notes and source URLs when the task asks for researched content.
- End with a concise summary of files changed and checks run.
"""
```

### 🎬 演示效果

```
任务：查阅明日方舟阿米娅的资料，编写一个 HTML 介绍页面

--- Planner ---
📋 Plan: Research Amiya, then build HTML page
  ✅ todo-1: Search Amiya character profile
  ✅ todo-2: Create amiya_profile.html with researched content

🔄 Handoff: planner → searchAgent
  "Search for Amiya from Arknights: character profile, story, abilities"

--- searchAgent ---
🔍 WebSearchTool: "明日方舟 阿米娅 角色 资料"
🔍 WebSearchTool: "Arknights Amiya character profile"
📋 Found: 阿米娅是罗德岛公开领袖... (sources: 3 URLs)

🔄 Handoff: searchAgent → planner (result: research summary)

🔄 Handoff: planner → codeAgent
  "Create amiya_profile.html using the research notes. Must include at least 2 source links."

--- codeAgent ---
🔧 FileWriteTool → amiya_profile.html
📋 NotepadAppendTool: "Created amiya_profile.html, includes 3 source links"

--- Verifier ---
✅ checks:
  - html_exists: amiya_profile.html present ✓
  - source_citations: 3 source URLs found ✓
  - content_quality: Contains name, story, abilities sections ✓
```

### 💡 讲解要点

- **为什么拆 Agent**：单一 Agent 上下文太杂，搜索和编码思维模式不同，拆开后 Prompt 更精准
- **Supervisor 模式**：Planner 不干活，只分配活。这是 MultiAgent 最常见的协调模式
- **Agent Handoff**：Planner → searchAgent → Planner → codeAgent 的交接链，每次交接都带 instruction 和 result
- **Notepad 机制**：codeAgent 可以用 NotepadAppendTool 写持久笔记，这样即使 Context 被压缩，关键信息也不会丢
- **WebSearchTool**：用 Tavily API 实现搜索，返回结构化结果（title、url、content、score）

### 🔍 核心代码

**Planner 的 Agent 调度工具 — [nodes.py:473](src/nexusagent/graph/nodes.py#L473)**

```python
def _build_planner_tools(state, writer):
    return [
        StructuredTool.from_function(name="TodoWriteTool", ...),          # 写计划
        StructuredTool.from_function(name="CallSearchAgentTool",          # 调度搜索
            func=lambda instruction: _call_search_agent_tool(state, writer, instruction), ...),
        StructuredTool.from_function(name="CallCodeAgentTool",            # 调度编码
            func=lambda instruction: _call_code_agent_tool(state, writer, instruction), ...),
    ]
```

> Planner 不直接操作文件，只通过 `CallSearchAgentTool` / `CallCodeAgentTool` 两个工具调度专家。

**SearchAgent Handoff — [nodes.py:536](src/nexusagent/graph/nodes.py#L536)**

```python
def _call_search_agent_tool(state, writer, instruction):
    writer({"type": "handoff", "from": "planner", "to": "searchAgent", "instruction": instruction})
    result = run_search_agent(state, instruction, writer=writer)
    state["research_notes"] = _join_notes(state.get("research_notes", ""), result.get("summary", ""))
    state["sources"] = _dedupe_sources(existing_sources + result.get("sources", []))
    state["agent_handoffs"] = list(state.get("agent_handoffs", [])) + [
        {"from_agent": "planner", "to_agent": "searchAgent", "instruction": instruction, "result": result.get("summary", "")}
    ]
    return {"ok": True, "summary": ..., "sources": ..., "queries": ...}
```

> 每次 Handoff 记录 `from → to + instruction + result`，写入 `state["agent_handoffs"]`，后续 Agent 可看到完整交接链。

**CodeAgent Handoff — [nodes.py:558](src/nexusagent/graph/nodes.py#L558)**

```python
def _call_code_agent_tool(state, writer, instruction):
    writer({"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": instruction})
    result = run_code_agent(state, instruction, writer=writer)
    state["todos"] = result.get("todos", state.get("todos", []))
    state["code_agent_summary"] = result.get("summary", "")
    state["agent_handoffs"] = list(state.get("agent_handoffs", [])) + [
        {"from_agent": "planner", "to_agent": "codeAgent", "instruction": instruction, "result": ...}
    ]
    return {"ok": True, "summary": ..., "todos": ...}
```

**SearchAgent 实现 — [search_agent.py:17](src/nexusagent/agents/search_agent.py#L17)**

```python
def run_search_agent(state, instruction, *, writer=None, max_loops=4):
    search_agent = model.bind_tools([build_web_search_tool()])  # ← 只有一个工具
    messages = [SystemMessage(SEARCH_AGENT_PROMPT), HumanMessage(...)]
    for _ in range(max_loops):
        response = search_agent.invoke(messages)
        if not tool_calls: break
        # 执行 WebSearchTool，收集 queries + sources + answers
    return {"ok": True, "summary": ..., "queries": ..., "sources": _dedupe_sources(sources)}
```

**CodeAgent 实现 — [code_agent.py:21](src/nexusagent/agents/code_agent.py#L21)**

```python
def run_code_agent(state, instruction, *, writer=None, max_loops=10):
    memory = build_layered_memory({**state, "todos": todos}, node="codeAgent")  # ← 注入分层记忆
    code_agent = model.bind_tools(build_tools(runtime) + [_build_todo_update_tool(todos)])
    messages = [SystemMessage(CODE_AGENT_PROMPT), HumanMessage(_code_agent_input(state, instruction, memory))]
    for _ in range(max_loops):
        response = code_agent.invoke(messages)
        if not tool_calls: break
        for call in tool_calls:
            if call.name == "TodoUpdateTool": result = update_todo(todos, ...)  # ← 特殊处理
            else: result = tool.invoke(args)
    return {"ok": True, "summary": ..., "todos": todos, "messages": ..., "tool_events": ...}
```

### 🤖 Vibe Coding Prompt

**Step 1: 创建 search_agent.py**

```
在 src/nexusagent/agents/search_agent.py 中实现搜索专家 Agent：

def run_search_agent(state, instruction, *, writer=None, max_loops=4) -> dict:
    """
    1. 创建 model，bind_tools([WebSearchTool])
    2. 消息：SystemMessage(SEARCH_AGENT_PROMPT) + HumanMessage(任务+指令+已有研究笔记)
    3. ReAct 循环 max_loops 次：
       - response = agent.invoke(messages)
       - 提取 tool_calls，执行 WebSearchTool
       - 收集 queries, sources, answers
       - writer 写入事件（type: tool_call, search_results）
    4. 返回 {ok: True, summary, queries, sources, messages, tool_events}
    """

同时在 src/nexusagent/tools/web_search_tool.py 中实现 WebSearchTool：
- 调用 tavily-python 的 TavilyClient.search()
- 返回 {ok, query, answer, results: [{title, url, content, score}]}
- 需要 TAVILY_API_KEY 环境变量
- 如果没有 API key，返回 {ok: False, error: "missing TAVILY_API_KEY"}

SEARCH_AGENT_PROMPT：
"""You are searchAgent, a focused research specialist.

Your only external capability is WebSearchTool. Search for reliable information
needed by the planner and codeAgent.

Rules:
- Use WebSearchTool for factual research.
- Prefer official or encyclopedia-style sources when available.
- Return a concise research summary and list the useful source URLs.
- Do not write files or produce application code.
"""
```

**Step 2: 创建 code_agent.py**

```
在 src/nexusagent/agents/code_agent.py 中实现代码专家 Agent：

def run_code_agent(state, instruction, *, writer=None, max_loops=10) -> dict:
    """
    1. 创建 model，bind_tools(build_tools(state) + [TodoUpdateTool])
    2. 构建 layered memory 快照（后续阶段实现，先留接口）
    3. 消息：SystemMessage(CODE_AGENT_PROMPT) + HumanMessage(任务+指令+session上下文+memory)
    4. ReAct 循环 max_loops 次：
       - response = agent.invoke(messages)
       - 提取 tool_calls，逐个执行
       - 如果是 TodoUpdateTool，更新 todos 并持久化
       - writer 写入事件
    5. 返回 {ok: True, summary, todos, messages, tool_events}
    """

CODE_AGENT_PROMPT：
"""You are codeAgent, a focused implementation specialist.

You implement the planner's instruction inside the workspace using file and
shell tools.

Rules:
- You must update todo progress explicitly.
- Before starting a todo, call TodoUpdateTool with status "in_progress".
- After finishing that todo, call TodoUpdateTool with status "completed".
- If a todo is impossible, call TodoUpdateTool with status "blocked" and explain.
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool for non-interactive checks.
- Use NotepadAppendTool to record durable findings, decisions, important files,
  blockers, and next-step context that should survive compression.
- Use NotepadReadTool when you need to recover prior notes.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- Incorporate research notes and source URLs when the task asks for researched content.
- End with a concise summary of files changed and checks run.
"""
```

**Step 3: 改造 planner_node 为 Supervisor**

```
改造 src/nexusagent/graph/nodes.py 中的 planner_node：

1. Planner 的工具从 TodoWriteTool 变为三个：
   - TodoWriteTool: 发布/修订计划
   - CallSearchAgentTool: 委托搜索任务给 searchAgent
   - CallCodeAgentTool: 委托实现任务给 codeAgent

2. CallSearchAgentTool 的实现：
   def _call_search_agent_tool(state, writer, instruction):
       writer({"type": "handoff", "from": "planner", "to": "searchAgent", "instruction": instruction})
       result = run_search_agent(state, instruction, writer=writer)
       # 更新 state: research_notes, sources, agent_handoffs
       return result

3. CallCodeAgentTool 的实现：
   def _call_code_agent_tool(state, writer, instruction):
       writer({"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": instruction})
       result = run_code_agent(state, instruction, writer=writer)
       # 更新 state: todos, code_agent_summary, agent_handoffs, messages
       return result

4. AgentHandoff 数据结构：
   class AgentHandoff(TypedDict, total=False):
       from_agent: str
       to_agent: str
       instruction: str
       result: str

5. 在 NexusGraphState 中新增字段：
   - research_notes: str
   - sources: list[SourceItem]
   - agent_handoffs: list[AgentHandoff]
   - code_agent_summary: str

6. 更新 PLANNER_PROMPT（在 src/nexusagent/prompts/stage3.py）：
   """You are the planner/supervisor node in NexusAgent stage 3.

   You coordinate specialist agents through tools. You cannot directly edit files
   or search the web yourself; delegate specialist work through tool calls.

   Available tools:
   - TodoWriteTool: publish or revise the plan, todos, acceptance criteria.
   - CallSearchAgentTool: delegate web/document research.
   - CallCodeAgentTool: delegate file/code implementation.

   Rules:
   - Always call TodoWriteTool before delegating new work.
   - For tasks that require current facts, call CallSearchAgentTool before CallCodeAgentTool.
   - If the verifier failed, revise the plan and delegate only the missing fix.
   - End with a concise supervisor summary after the needed specialist calls.
   """

7. 从图中移除 actor 节点，planner 直接通过工具调用 searchAgent 和 codeAgent。
   图结构变为：START → planner → verifier → (passed? → final / planner) → END
```

**Step 4: 更新 VERIFIER_PROMPT**

```
在 src/nexusagent/prompts/stage3.py 中更新 VERIFIER_PROMPT：

"""You are verifier, a model-based reviewer node.

You decide whether the user's task is complete by inspecting state and using
read-only tools. You may read files, grep, run safe shell checks, and search
the web. You must not modify files.

Rules:
- Check the actual workspace, not only the previous agent summaries.
- Read NOTEPAD.md with NotepadReadTool when prior durable context matters.
- Run the provided verification commands when they are relevant.
- For researched content, confirm the output cites useful sources.
- Return only JSON with these keys:
  passed: boolean
  reason: short human-readable explanation
  checks: list of {name, passed, detail}
  recommended_next_instruction: what planner should ask a specialist to fix, or
    an empty string when passed
"""
```

**Step 5: 测试运行**

```bash
# 需要设置 TAVILY_API_KEY
export TAVILY_API_KEY=your_key_here
nexusagent "查阅明日方舟阿米娅的资料，编写一个HTML介绍页面，至少包含2个来源链接"
```

---

## 阶段四：引入 Context Engineer

### 🎯 设计目标

Agent 执行长程任务时，消息历史会越来越长，最终撑爆上下文窗口。引入 Context Engineering 的三大措施：

1. **压缩机制**：当 token 数接近上限时，自动压缩消息历史
2. **Notepad 持久笔记**：关键信息写入 NOTEPAD.md，不依赖消息历史
3. **分层 Memory**：Rules 层 + Working Memory 层 + History Summary 层

### 🏗️ 架构设计

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Planner   │
                    └──────┬──────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   Context Monitor      │  ← 检测 token 数
               └───────┬───────────────┘
                       │
              ┌────────┴────────┐
              │ should_compress?│
              └──┬──────────┬───┘
              no  │          │  yes
                  │          ▼
                  │   ┌──────────────────┐
                  │   │Context Compressor │  ← LLM 压缩消息历史
                  │   └────────┬─────────┘
                  │            │
                  ▼            ▼
           ┌─────────────┐  ┌─────────────┐
           │  Verifier    │  │  Planner    │  ← 压缩后重新进入 planner
           └──────┬──────┘  └─────────────┘
                  │
           ┌──────┴──────┐
           │   passed?   │
           └──┬──────┬───┘
         yes  │      │  no
              ▼      ▼
        ┌────────┐ ┌─────────┐
        │ Final  │ │ Planner │
        └───┬────┘ └─────────┘
            │           │
            ▼           │
        ┌───────┐       │
        │  END  │◀──────┘
        └───────┘
```

**三层 Memory 架构：**

```
┌─────────────────────────────────────────────────┐
│  Rules Layer (固定规则)                           │
│  - 只在 workspace 内操作                          │
│  - 使用相对路径                                   │
│  - TODO.md = 工作计划状态                          │
│  - NOTEPAD.md = 持久笔记                          │
│  - HISTORY_SUMMARY.md = 压缩历史                  │
│  - Agent 不直接写 Memory，由 Runtime 组装          │
├─────────────────────────────────────────────────┤
│  Working Memory (当前任务状态)                     │
│  - 当前节点、任务、session 信息                     │
│  - plan_summary、todos、acceptance_criteria       │
│  - research_notes、sources                        │
│  - agent_handoffs（最近 6 条）                     │
│  - code_agent_summary、verifier_summary           │
│  - last_error、attempts                           │
├─────────────────────────────────────────────────┤
│  History Summary Store (压缩历史)                  │
│  - HISTORY_SUMMARY.md 的内容摘要                   │
│  - NOTEPAD.md 的内容摘要                           │
│  - context_summary（上一轮压缩结果）               │
│  - compression_events（最近 3 次压缩记录）          │
└─────────────────────────────────────────────────┘
```

**Context Monitor 节点的逻辑：**

```python
def context_monitor_node(state):
    # 1. 估算当前 token 数
    token_count = estimate_context_tokens(state)
    # 2. 判断是否需要压缩
    should_compress = token_count > limit
    # 3. 记录下一个应该去的节点
    next_node = state.get("context_next_node", "verifier")
    return {
        "context_token_count": token_count,
        "context_should_compress": should_compress,
        "context_next_node": next_node,
    }
```

**Context Compressor 节点：**

用 LLM 做摘要压缩，保留关键信息，删除冗余的 tool 调用和长输出：

```python
CONTEXT_COMPRESSION_PROMPT = """You are the context_compressor node.

Keep everything needed to resume work:
- user task and active goal
- current plan, todos, acceptance criteria, verification commands
- completed work and current files/artifacts
- important tool findings and command results
- research notes and source URLs
- latest verifier failure and recommended next step

Remove redundant transcript detail:
- repeated tool calls
- long stdout/stderr
- duplicate search snippets
- stale intermediate reasoning
"""
```

**Notepad 持久笔记：**

```python
# NOTEPAD.md 示例
# NexusAgent Notepad

## Key Files
_Recorded: 2025-01-15 14:30:00_

- amiya_profile.html: main deliverable
- style.css: extracted styles

## Design Decisions
_Recorded: 2025-01-15 14:35:00_

- Using flexbox layout for responsive design
- Color scheme: dark blue + orange accent per Arknights theme
```

### 🎬 演示效果

```
任务：帮我搭建一个完整的 Flask 后台管理系统，
     包含用户认证、REST API、数据库模型、前端模板

--- Round 1: Planner → searchAgent → codeAgent ---
... (大量文件创建和工具调用) ...

📊 Context Monitor: tokens=385,421 / 400,000 → should_compress=True
🔄 Routing to Context Compressor...

--- Context Compression ---
📉 Before: 385,421 tokens, 47 messages
📈 After:  52,138 tokens, 8 messages
📝 Summary: "Created Flask app with auth/API/models/templates. User model
   complete, login/logout working. API endpoints for CRUD operational.
   Still need: admin panel, unit tests, deployment config."

--- Round 2: Compressed context → Planner continues ---
... (Planner sees compressed summary, knows where to continue) ...

📊 Context Monitor: tokens=89,234 / 400,000 → OK, proceed to verifier
```

### 💡 讲解要点

- **Context Engineer 的本质**：不是简单的"删消息"，而是**决定什么信息以什么形式存在于什么位置**
- **三层 Memory**：Rules（不变规则）、Working Memory（当前任务上下文）、History Summary（压缩历史）
- **压缩不是删除**：Compressor 用 LLM 做摘要，保留关键信息，只删除冗余的 tool 输出
- **Notepad 是"第二大脑"**：Agent 主动写入，不依赖消息历史。即使所有消息被压缩，Notepad 里的关键信息还在
- **MAX_TEXT_CHARS**：每个字段都有截断上限，防止单个字段撑爆上下文

### 🔍 核心代码

**三层 Memory 构建 — [memory.py:39](src/nexusagent/graph/memory.py#L39)**

```python
def build_layered_memory(state, *, node="graph") -> dict:
    notepad = read_notepad(runtime)          # 读 NOTEPAD.md
    history = read_history_summary(runtime)  # 读 HISTORY_SUMMARY.md

    working_memory = {
        "node": node, "task": ..., "todos": ..., "plan_summary": ...,
        "research_notes": _short_text(..., 1600),     # ← 截断！
        "agent_handoffs": _trim_handoffs(...),         # ← 只保留最近 6 条
        "code_agent_summary": _short_text(..., 1000),  # ← 截断！
        ...
    }
    history_summary_store = {
        "history_summary": _short_text(..., 2200),
        "notepad": _short_text(..., 1800),              # ← 从磁盘读 NOTEPAD.md
        "context_summary": _short_text(..., 1600),
        "compression_events": state.get("compression_events", [])[-3:],  # ← 只保留最近 3 次
    }
    return {"rules": dict(RULES_LAYER), "working_memory": working_memory, "history_summary_store": history_summary_store}
```

> Rules Layer 硬编码不变规则；Working Memory 从 state 实时构建，每个字段 `_short_text` 截断；History Summary Store 从磁盘读 NOTEPAD.md 和 HISTORY_SUMMARY.md。

**Context Monitor — [nodes.py:327](src/nexusagent/graph/nodes.py#L327)**

```python
def context_monitor_node(state) -> dict:
    token_limit = get_context_token_limit()   # 默认 400000
    token_count = estimate_context_tokens(state)  # ← 用模型 tokenizer 精确计算
    should_compress = token_count >= token_limit
    next_node = state.get("context_next_node") or "verifier"
    return {"context_token_count": token_count, "context_should_compress": should_compress, "context_next_node": next_node}
```

**Token 估算 — [nodes.py:460](src/nexusagent/graph/nodes.py#L460)**

```python
def estimate_context_tokens(state) -> int:
    messages = list(state.get("messages", []))
    payload = build_layered_memory(state, node="context_monitor")
    payload_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str))
    try:
        model = create_model()
        return int(model.get_num_tokens_from_messages(messages + [payload_message]))  # ← 精确
    except Exception:
        return max(1, len(text) // 4)  # ← fallback: 4 字符 ≈ 1 token
```

**Context Compressor — [nodes.py:356](src/nexusagent/graph/nodes.py#L356)**

```python
def context_compressor_node(state) -> dict:
    before_messages = list(state.get("messages", []))
    compressed = _compress_context_with_model(state)  # ← LLM 做摘要
    summary = _format_compressed_context(compressed, state)
    summary_message = AIMessage(content=summary)
    persist_history_summary(state["runtime"], summary)  # ← 持久化到 HISTORY_SUMMARY.md

    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary_message],  # ← 清空旧消息，替换为摘要
        "context_summary": summary,
        "context_should_compress": False,   # ← 刚压缩过
        "research_notes": _short_text(state.get("research_notes", ""), 1200),  # ← 截断长字段
        "compression_events": [...prev, {"before_tokens": before, "after_tokens": after, "removed_messages": len(before)}],
    }
```

**LLM 压缩核心 — [nodes.py:616](src/nexusagent/graph/nodes.py#L616)**

```python
def _compress_context_with_model(state) -> dict:
    memory = build_layered_memory(state, node="context_compressor")
    payload = {
        "context_summary": state.get("context_summary", ""),
        "memory": memory,
        "messages": [_message_snapshot(msg) for msg in state.get("messages", [])],  # ← 所有消息
    }
    messages = [SystemMessage(CONTEXT_COMPRESSION_PROMPT), HumanMessage(content=json.dumps(payload, ...))]
    response = create_model().invoke(messages)
    parsed = _extract_json(str(response.content))
    if parsed: return parsed
    return _fallback_compression(state, error=...)  # ← LLM 失败时用规则 fallback
```

> 压缩流程：全量消息+Memory → LLM → 结构化摘要 → `RemoveMessage(REMOVE_ALL)` 清空 → 一条 AIMessage 替换 → 截断所有长字段 → 持久化到 HISTORY_SUMMARY.md。

**Notepad 工具 — [notepad_tool.py:21](src/nexusagent/tools/notepad_tool.py#L21)**

```python
def append_notepad(state, heading, content) -> dict:
    path = state.assert_workspace_path(state.workspace / NOTEPAD_FILE)
    existing = read_text_lossy(path) if path.exists() else "# NexusAgent Notepad\n"
    entry = f"\n## {heading}\n\n_Recorded: {timestamp}_\n\n{content.strip()}\n"
    updated = existing.rstrip() + "\n" + entry
    path.write_text(updated, encoding="utf-8")
    return {"ok": True, "path": NOTEPAD_FILE, "heading": heading, "lines": len(updated.splitlines())}
```

> 追加式写入 NOTEPAD.md。Agent 主动调用。压缩后 Memory 从磁盘读取此文件，关键信息不丢失。

### 🤖 Vibe Coding Prompt

**Step 1: 实现分层 Memory**

```
在 src/nexusagent/graph/memory.py 中实现三层 Memory 系统：

1. Rules Layer（固定规则层）：
RULES_LAYER = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}

2. Working Memory（当前任务状态层）：
def build_layered_memory(state, *, node="graph") -> dict:
    runtime = state["runtime"]
    notepad = read_notepad(runtime)        # 读 NOTEPAD.md
    history = read_history_summary(runtime) # 读 HISTORY_SUMMARY.md

    working_memory = {
        "node": node,
        "task": state.get("task", ""),
        "session_id": ...,
        "session_turn": ...,
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": _short_text(state.get("research_notes", ""), 1600),
        "sources": [...],  # 只保留 title 和 url
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs", [])),  # 最近 6 条
        "code_agent_summary": _short_text(state.get("code_agent_summary", ""), 1000),
        "verifier_summary": _short_text(state.get("verifier_summary", ""), 1000),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

3. History Summary Store（压缩历史层）：
    history_summary_store = {
        "history_path": "HISTORY_SUMMARY.md",
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history_summary, 2200),
        "notepad_path": "NOTEPAD.md",
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), 1800),
        "context_summary": _short_text(state.get("context_summary", ""), 1600),
        "compression_events": state.get("compression_events", [])[-3:],
    }

    return {"rules": dict(RULES_LAYER), "working_memory": working_memory, "history_summary_store": history_summary_store}

关键辅助函数：
- _short_text(text, limit): 超长文本截断，末尾加 "..."
- _trim_handoffs(handoffs): 只保留最近 6 条交接记录
- format_layered_memory_for_prompt(memory): json.dumps 格式化

在 NexusGraphState 中新增字段：
- context_summary: str
- context_token_count: int
- context_token_limit: int
- context_should_compress: bool
- context_next_node: str
- compression_events: list[CompressionEvent]
- memory_snapshot: LayeredMemory
- history_summary: str
```

**Step 2: 实现 Context Monitor 节点**

```
在 src/nexusagent/graph/nodes.py 中新增 context_monitor_node：

def context_monitor_node(state) -> dict:
    """
    1. 估算当前 token 数：
       token_count = model.get_num_tokens_from_messages(messages + [memory_payload])
       如果异常，fallback 为 len(text) // 4
    2. 判断是否需要压缩：
       should_compress = token_count > context_token_limit (默认 400000)
    3. context_next_node 由上游节点设置（planner 后设为 "verifier"，
       verifier 失败后设为 "planner"）
    4. 返回：
       {
           "context_token_count": token_count,
           "context_should_compress": should_compress,
           "context_next_node": state.get("context_next_node", "verifier"),
       }
    """

def context_monitor_route(state) -> str:
    if state.get("passed"):
        return "final"
    if state.get("context_should_compress"):
        return "context_compressor"
    return state.get("context_next_node", "verifier")
```

**Step 3: 实现 Context Compressor 节点**

```
在 src/nexusagent/graph/nodes.py 中新增 context_compressor_node：

def context_compressor_node(state) -> dict:
    """
    1. 用 LLM 压缩消息历史，保留关键信息：
       - 调用 create_model().invoke([
           SystemMessage(CONTEXT_COMPRESSION_PROMPT),
           HumanMessage(当前所有消息 + 分层 memory 快照)
         ])
       - LLM 返回 JSON: {summary, active_goal, completed_work, open_todos,
           important_files, tool_findings, sources, next_steps, risks}
    2. 替换消息历史为压缩后的摘要：
       - 用 RemoveMessage(id=REMOVE_ALL_MESSAGES) 清除所有旧消息
       - 添加一条 AIMessage(content=summary) 作为新的上下文起点
    3. 持久化到 HISTORY_SUMMARY.md
    4. 截断各字段的文本长度（_short_text）
    5. 返回压缩事件：
       {
           "messages": [RemoveMessage, AIMessage(summary)],
           "context_summary": summary,
           "context_token_count": 新 token 数,
           "context_should_compress": False,  # 刚压缩过，不需要再压缩
           "research_notes": 截断后,
           "agent_handoffs": 截断后,
           ... 其他截断字段,
           "history_summary": summary,
           "compression_events": [...prev, 新事件],
       }
    """

CONTEXT_COMPRESSION_PROMPT（在 src/nexusagent/prompts/stage4.py）：
"""You are the context_compressor node in NexusAgent stage 4.

Your job is to compress the graph context so the task can continue with a much
smaller message window.

Keep everything needed to resume work:
- user task and active goal
- current plan, todos, acceptance criteria, verification commands
- completed work and current files/artifacts
- important tool findings and command results
- research notes and source URLs
- latest verifier failure and recommended next step
- risks, blockers, and assumptions

Remove redundant transcript detail:
- repeated tool calls
- long stdout/stderr
- duplicate search snippets
- stale intermediate reasoning

Return only JSON with these keys:
- summary
- active_goal
- completed_work
- open_todos
- important_files
- tool_findings
- sources
- next_steps
- risks
"""
```

**Step 4: 更新工作流图**

```
在 src/nexusagent/graph/workflow.py 中更新图结构：

def build_complex_workflow():
    graph = StateGraph(NexusGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("context_monitor", context_monitor_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges("context_monitor", context_monitor_route, {
        "context_compressor": "context_compressor",
        "verifier": "verifier",
        "planner": "planner",
        "final": "final",
    })
    graph.add_conditional_edges("context_compressor", context_compressor_route, {
        "verifier": "verifier",
        "planner": "planner",
        "final": "final",
    })
    graph.add_edge("verifier", "context_monitor")  # 验证后也过 monitor
    graph.add_edge("final", END)
    return graph.compile()

def context_compressor_route(state) -> str:
    # 压缩后去哪里？由 context_next_node 决定
    return state.get("context_next_node", "verifier")
```

**Step 5: 在各节点中注入 Memory**

```
在每个节点的输入中注入分层 Memory：

# planner_node
memory = build_layered_memory(working_state, node="planner")
writer(memory_event(memory, node="planner"))
messages = [SystemMessage(PLANNER_PROMPT), HumanMessage(_planner_input(working_state, memory))]

# code_agent
memory = build_layered_memory(state, node="codeAgent")
writer(memory_event(memory, node="codeAgent"))
messages = [SystemMessage(CODE_AGENT_PROMPT), HumanMessage(_code_agent_input(state, instruction, memory))]

# verifier_node 同理注入 memory

其中 _planner_input 和 _code_agent_input 会把 format_layered_memory_for_prompt(memory)
拼接到 HumanMessage 中，让每个 Agent 都能看到完整的分层记忆。
```

**Step 6: 测试运行**

```bash
# 设置一个较低的 token limit 来更容易触发压缩
export NEXUS_CONTEXT_TOKEN_LIMIT=50000
nexusagent "帮我搭建一个Flask后台管理系统，包含用户认证、REST API、数据库模型、前端模板"
```

---

## 阶段五：引入 Harness Engineer

### 🎯 设计目标

Agent 在生产环境中需要更多"安全网"和"可观测性"。引入 Harness Engineering 的三大措施：

1. **人类在环审批**：高风险命令（pip install、curl 等）需要人类确认
2. **Checkpoint 检查点**：定期保存状态，中断后可恢复
3. **Trace 链路追踪**：详细记录每一步的执行日志

### 🏗️ 架构设计

**1. 人类审批机制（Human-in-the-Loop）**

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│ codeAgent │────▶│  BashTool     │────▶│ classify_risk│
└──────────┘     │  run_bash()   │     └──────┬───────┘
                 └──────────────┘            │
                                      ┌──────┴──────┐
                                      │  risk_level  │
                                      └──┬──────┬───┘
                                    safe │      │ risky
                                         ▼      ▼
                                    ┌────────┐ ┌──────────────┐
                                    │ 直接执行│ │approval_handler│
                                    └────────┘ └──────┬───────┘
                                                       │
                                                ┌──────┴──────┐
                                                │   人类决策   │
                                                └──┬──────┬───┘
                                              approve │      │ deny
                                                    ▼      ▼
                                               ┌────────┐ ┌────────┐
                                               │  执行   │ │  拒绝   │
                                               └────────┘ └────────┘
```

**风险命令分类：**

```python
RISK_PATTERNS = [
    (r"pip\s+install\b", "Python package installation"),
    (r"uv\s+add\b", "Project dependency change"),
    (r"npm\s+install\b", "Node package installation"),
    (r"(?:curl|wget)\b", "Network download command"),
    (r"uvicorn\b", "Long-running development server"),
    (r"python\s+-m\s+http\.server\b", "Long-running development server"),
]
```

**三种审批模式：**
- `inline`：每条风险命令都在终端弹窗确认
- `auto`：自动批准所有命令
- `deny`：直接拒绝所有风险命令

**2. Checkpoint 检查点机制**

```
┌────────────────────────────────────────────────┐
│           .nexusagent/checkpoints/               │
├────────────────────────────────────────────────┤
│ checkpoint.json   ← 最新状态快照               │
│ state.json        ← 完整图状态 (strict mode)    │
│ events.jsonl      ← 事件日志 (strict mode)      │
│ RECOVERY.md       ← 人类可读的恢复指南          │
│ .git/             ← workspace 的 git 快照       │
│ workspace_manifest.json ← 文件清单             │
└────────────────────────────────────────────────┘
```

**Checkpoint 模式：**
- `light`（默认）：只保存 checkpoint.json + RECOVERY.md + git 快照，轻量高效
- `strict`：额外保存完整 state.json 和 events.jsonl，完整但更慢
- `off`：不保存检查点

**恢复流程：**
```bash
nexusagent --resume .nexusagent/workspaces/workspace-xxx
```

**3. Trace 链路追踪**

```
.nexusagent/traces/{trace_id}/
├── trace.json          ← 总览（task、status、耗时、统计）
├── events.jsonl        ← 每一步事件的详细日志
└── timeline.md         ← 人类可读的时间线
```

**追踪统计：**
```json
{
  "trace_id": "abc123",
  "task": "搭建Flask后台",
  "status": "finished",
  "duration_ms": 45230,
  "node_visits": {"planner": 3, "codeAgent": 2, "verifier": 2},
  "tool_calls": 12,
  "failed_tool_calls": 1,
  "approval_count": 2,
  "checkpoint_count": 5,
  "handoff_count": 3,
  "timeline": [
    "run_start: task='搭建Flask后台'",
    "node:planner visit=1",
    "tool_call: TodoWriteTool",
    "handoff: planner→codeAgent",
    "tool_call: BashTool risk=Python package installation → approved",
    "tool_result: ok=True",
    "checkpoint_saved: light",
    "node:verifier visit=1",
    "node:final status=passed"
  ]
}
```

### 🎬 演示效果

```
任务：帮我搭建一个 Flask 后台管理系统

--- codeAgent ---
🔧 BashTool: pip install flask flask-sqlalchemy flask-login

⚠️  Human Approval Required
    Risk: Python package installation
    Command: pip install flask flask-sqlalchemy flask-login

    [Y] Approve  [N] Deny

> Y

✅ Approved. Executing...
📦 Successfully installed flask-3.0.0 flask-sqlalchemy-3.1.1 ...

--- codeAgent continues ---
🔧 FileWriteTool → app.py
🔧 FileWriteTool → models.py
🔧 FileWriteTool → routes.py

💾 Checkpoint saved (light mode)
   Node: codeAgent, Files: 3, Git: abc123

--- Ctrl+C interrupted ---
💾 Checkpoint saved (status=interrupted)
📋 Recovery guide written to RECOVERY.md

--- Resume ---
$ nexusagent --resume .nexusagent/workspaces/workspace-xxx

🔄 Resuming from checkpoint...
   Last node: codeAgent
   Files: 3
   Attempts: 1/3

--- Continues from where it left off ---
```

### 💡 讲解要点

- **为什么需要 Harness**：Agent 不是玩具，它会在你的电脑上执行命令。没有安全网就是"裸奔"
- **风险分类是关键**：不是所有命令都需要审批，只有"有副作用"的命令才拦截
- **三种审批模式的取舍**：开发时用 auto，生产用 inline，受限环境用 deny
- **Checkpoint 的 light vs strict**：light 适合日常，strict 适合调试。light 只保存 git 快照 + 元数据，开销很小
- **Trace 的价值**：不是给自己看的，是给"未来的自己"和"调试时的自己"看的。出问题时，Trace 是唯一能回溯的线索

### 🔍 核心代码

**命令风险分类 — [approval.py:44](src/nexusagent/core/approval.py#L44)**

```python
RISK_PATTERNS = [
    (r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b", "Python package installation"),
    (r"(?:^|&&|\|\||;)\s*uv\s+add\b", "Project dependency change with uv add"),
    (r"(?:^|&&|\|\||;)\s*npm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b", "Network download command"),
    (r"(?:^|&&|\|\||;)\s*uvicorn\b", "Long-running development server"),
    ...
]

def classify_command_risk(command: str) -> str | None:
    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return reason
    return None  # 安全命令，不需要审批
```

**审批三模式分流 — [bash_tool.py:224](src/nexusagent/tools/bash_tool.py#L224)**

```python
def _resolve_approval(state, command) -> dict | None:
    risk_reason = classify_command_risk(command)
    if risk_reason is None:
        return None  # ← 安全命令，直接放行

    request = make_approval_request(command, risk_reason)
    base = {"requires_approval": True, "risk_reason": risk_reason, "command": command}

    if state.approval_mode == "auto":
        return {**base, "approved": True}                     # ← auto: 自动批准
    if state.approval_mode == "deny" or state.approval_handler is None:
        return {**base, "ok": False, "approved": False,       # ← deny: 直接拒绝
                "error": f"human approval required: {risk_reason}"}

    decision = state.approval_handler(request)                 # ← inline: 等人类决策
    approved = decision.approved if isinstance(decision, ApprovalDecision) else bool(decision)
    if approved: return {**base, "approved": True}
    return {**base, "ok": False, "approved": False, "error": f"human rejected: {risk_reason}"}
```

> 核心流程：`classify_command_risk` → 有风险？→ 三模式分流：auto 放行 / deny 拒绝 / inline 调用 `approval_handler` 阻塞等待。

**BashTool 执行入口 — [bash_tool.py:154](src/nexusagent/tools/bash_tool.py#L154)**

```python
def run_bash(state, command, timeout_seconds=None, run_in_background=False):
    if not command.strip(): return {"ok": False, "error": "empty command"}  # 1. 校验
    timeout = _coerce_timeout(timeout_seconds)                                # 2. 超时
    handled = _handle_tail_command(state, normalized_command)                  # 3. 特殊命令
    blocked = _looks_dangerous(normalized_command)                             # 4. 危险拦截
    if blocked: return {"ok": False, "error": f"blocked: {blocked}"}
    approval = _resolve_approval(state, normalized_command)                    # 5. ★ 人类审批 ★
    if approval is not None and not approval.get("approved"): return approval
    completed = subprocess.run(normalized_command, cwd=state.workspace, ...)  # 6. 执行
    return {"ok": completed.returncode == 0, "exit_code": ..., **(approval or {})}
```

**Checkpoint 保存 — [checkpoint.py:47](src/nexusagent/core/checkpoint.py#L47)**

```python
class CheckpointManager:
    def save(self, state, *, status="running", latest_node=None, event=None):
        if not self.enabled: return None
        if event is not None and self.mode == "strict":
            self._append_event(event)                          # strict: 追加事件日志
        if self.mode == "strict":
            _write_json(self.root / STATE_FILE, serialize_state(state))  # strict: 完整状态

        manifest = workspace_manifest(self.workspace)         # 文件清单
        git_commit, git_error = snapshot_workspace_git(...)    # git 快照
        _write_json(self.root / CHECKPOINT_FILE, payload)      # 检查点元数据
        (self.root / RECOVERY_FILE).write_text(build_recovery_markdown(payload))  # 恢复指南
        return checkpoint_saved_event(payload)
```

> **light**：checkpoint.json + RECOVERY.md + git commit。**strict**：额外保存 state.json + events.jsonl。**off**：不保存。

**主循环中的 Checkpoint + Trace 集成 — [agent.py:291](src/nexusagent/core/agent.py#L291)**

```python
current_state = dict(inputs)
manager = CheckpointManager(state, task=...)
trace = TraceRecorder(state, task=...)
trace.start(current_state)
manager.save(current_state, status="started")  # ← 初始检查点

for mode, event in workflow.stream(inputs, stream_mode=["updates", "custom"]):
    if mode == "custom":
        trace.record_custom_event(event)
        if _custom_event_needs_checkpoint(event):  # ← 失败或需审批时保存
            manager.save(current_state, ...)
    else:
        trace.record_graph_update(event)
        manager.save(current_state, ...)           # ← 每个图节点更新都保存

except KeyboardInterrupt:
    manager.save(current_state, status="interrupted")  # ← 中断也保存
    trace.end(status="interrupted", ...)
```

### 🤖 Vibe Coding Prompt

**Step 1: 实现人类审批机制**

```
在 src/nexusagent/core/approval.py 中实现命令风险分类和审批：

1. 风险命令正则分类：
RISK_PATTERNS = [
    (r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b", "Python package installation"),
    (r"(?:^|&&|\|\||;)\s*uv\s+add\b", "Project dependency change with uv add"),
    (r"(?:^|&&|\|\||;)\s*uv\s+sync\b", "Dependency synchronization with uv sync"),
    (r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b", "Python package installation with uv pip"),
    (r"(?:^|&&|\|\||;)\s*npm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*pnpm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b", "Network download command"),
    (r"(?:^|&&|\|\||;)\s*uvicorn\b", "Long-running development server"),
    (r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b", "Long-running development server"),
]

def classify_command_risk(command: str) -> str | None:
    """匹配则返回风险原因字符串，否则返回 None（安全命令）"""

2. 审批请求和决策数据类：
@dataclass(frozen=True)
class ApprovalRequest:
    id: str           # "approval-{uuid4 hex[:8]}"
    command: str
    risk_reason: str
    tool_name: str = "BashTool"

@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""

3. 三种审批模式：
VALID_APPROVAL_MODES = {"inline", "auto", "deny"}

def normalize_approval_mode(mode: str | None) -> str:
    # 默认 "inline"，无效值也 fallback 到 "inline"

4. 在 BashTool 的 run_bash 中集成：
   - 执行前调用 classify_command_risk(command)
   - 如果有风险，根据 approval_mode 处理：
     - "auto": 直接放行，result 加 requires_approval=True 标记
     - "deny": 直接拒绝，result.ok=False
     - "inline": 调用 approval_handler(request) 等待人类决策
```

**Step 2: 实现 Checkpoint 检查点**

```
在 src/nexusagent/core/checkpoint.py 中实现断点保存和恢复：

class CheckpointManager:
    def __init__(self, runtime, task=""):
        self.workspace = runtime.workspace
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        # mode: "light" | "strict" | "off"
        self.root = workspace / ".nexusagent" / "checkpoints"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def save(self, state, *, status="running", latest_node=None, event=None):
        """保存检查点：
        1. 创建 self.root 目录
        2. 如果 strict 模式：追加事件到 events.jsonl，保存完整 state.json
        3. 生成 workspace 文件清单（workspace_manifest）
        4. git commit 工作区快照（snapshot_workspace_git）
        5. 保存 checkpoint.json（元数据 + 状态摘要）
        6. 生成 RECOVERY.md（人类可读的恢复指南）
        7. 返回 checkpoint_saved_event 或 None（如果 disabled）
        """

    @classmethod
    def load_resume_inputs(cls, runtime, task=None, max_attempts=3):
        """从检查点恢复：
        1. 读取 checkpoint.json
        2. 如果有 git commit，恢复工作区文件
        3. 重建 inputs 字典（task, runtime, messages, attempts 等）
        4. 返回 (inputs, resume_event)
        """

def resume_command(workspace: Path) -> str:
    """生成恢复命令字符串：nexusagent --resume <workspace>"""

def build_recovery_markdown(payload) -> str:
    """生成 RECOVERY.md 内容，包含：任务、状态、文件清单、git commit、恢复命令"""

三种 Checkpoint 模式对比：
- light: 只保存 checkpoint.json + RECOVERY.md + git 快照（每次节点切换保存）
- strict: 额外保存 state.json + events.jsonl（每个事件都追加）
- off: 完全不保存
```

**Step 3: 实现 Trace 链路追踪**

```
在 src/nexusagent/core/trace.py 中实现执行追踪：

class TraceRecorder:
    def __init__(self, runtime, task=""):
        self.workspace = runtime.workspace
        self.mode = normalize_trace_mode(runtime.trace_mode)
        self.trace_id = runtime.trace_id 或生成新 ID
        self.root = workspace / ".nexusagent" / "traces" / self.trace_id
        self.node_visits: dict[str, int] = {}   # 节点访问计数
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0

    def start(self, inputs, *, resumed=False, resume_event=None):
        """记录 run_start 事件"""

    def record_custom_event(self, event):
        """记录自定义事件，同时更新统计：
        - type=tool_call → tool_calls++
        - type=tool_result + ok=False → failed_tool_calls++
        - type=tool_result + requires_approval → approval_count++
        - type=handoff → handoff_count++
        - type=checkpoint_saved → checkpoint_count++
        """

    def record_graph_update(self, event):
        """记录图节点更新，统计 node_visits"""

    def end(self, *, status, latest_node, final_state) -> dict | None:
        """结束追踪，生成 trace.json 和 timeline.md：
        trace.json 包含：
        - trace_id, task, status, started_at, ended_at, duration_ms
        - node_visits, tool_calls, failed_tool_calls
        - approval_count, checkpoint_count, handoff_count
        - timeline_head（前 20 条）, timeline_tail（后 80 条）, timeline_omitted

        timeline.md 是人类可读的时间线摘要
        """

追踪文件结构：
.nexusagent/traces/{trace_id}/
├── trace.json     ← 统计概览
├── events.jsonl   ← 每条事件一行 JSON
└── timeline.md    ← 人类可读时间线
```

**Step 4: 集成到 agent.py**

```
更新 src/nexusagent/core/agent.py 中的 stream_agent_events：

def stream_agent_events(task, *, workspace, max_attempts=3,
                        approval_mode="inline", approval_handler=None,
                        checkpoint_mode="light", resume_workspace=None,
                        trace_mode="on"):

    # 1. 创建 RuntimeState（新增 approval_mode, approval_handler, checkpoint_mode, trace_mode 字段）
    state = create_runtime(workspace, approval_mode=approval_mode,
                           approval_handler=approval_handler,
                           checkpoint_mode=checkpoint_mode,
                           resume_from=resume_workspace,
                           trace_mode=trace_mode)

    # 2. 创建 CheckpointManager 和 TraceRecorder
    manager = CheckpointManager(state, task=task)
    trace = TraceRecorder(state, task=task)

    # 3. 记录开始事件，保存初始检查点
    trace.start(inputs)
    manager.save(current_state, status="started", latest_node="start")

    # 4. 运行工作流，每个事件都记录
    for mode, event in workflow.stream(inputs, stream_mode=["updates", "custom"]):
        if mode == "custom":
            trace.record_custom_event(event)
            if _custom_event_needs_checkpoint(event):
                manager.save(current_state, status="running", latest_node=latest_node, event=event)
            yield {"type": "custom_event", "event": event}
        else:
            trace.record_graph_update(event)
            manager.save(current_state, status="running", latest_node=latest_node, event=event)
            yield {"type": "graph_event", "event": event}

    # 5. 结束追踪，保存最终检查点
    manager.save(current_state, status="finished", latest_node=latest_node)
    trace.end(status="finished", latest_node=latest_node, final_state=current_state)

    # 6. 支持 KeyboardInterrupt 中断恢复
    except KeyboardInterrupt:
        manager.save(current_state, status="interrupted", latest_node=latest_node)
        trace.end(status="interrupted", latest_node=latest_node, final_state=current_state)
```

**Step 5: 更新 CLI 参数**

```
在 src/nexusagent/cli/app.py 中新增参数：

@app.command()
def main(
    ctx: typer.Context,
    workspace: Annotated[Path | None, Option("--workspace", "-w")] = None,
    max_attempts: Annotated[int, Option("--max-attempts")] = 3,
    approval_mode: Annotated[Literal["inline", "auto", "deny"], Option("--approval-mode")] = "inline",
    checkpoint_mode: Annotated[Literal["light", "strict", "off"], Option("--checkpoint-mode")] = "light",
    trace_mode: Annotated[Literal["on", "off"], Option("--trace-mode")] = "on",
    resume: Annotated[Path | None, Option("--resume")] = None,
):
    ...
```

**Step 6: 测试运行**

```bash
# 触发审批 + checkpoint + trace
nexusagent "帮我搭建一个Flask后台管理系统，包含用户认证" \
  --approval-mode inline \
  --checkpoint-mode light \
  --trace-mode on

# 中断后恢复
# Ctrl+C 中断后...
nexusagent --resume .nexusagent/workspaces/workspace-xxx

# 查看 trace
cat .nexusagent/workspaces/workspace-xxx/.nexusagent/traces/*/timeline.md
```

---

## 阶段六：引入 Claw 交互层

### 🎯 设计目标

前面五个阶段的输出都是终端文本，不够直观。用 Textual 实现一个 TUI（Terminal User Interface），让 Agent 的执行过程可视化。同时支持多轮对话和飞书 API 接入。

### 🏗️ 架构设计

**TUI 界面布局：**

```
┌─────────────────────────────────────────────────────┐
│  🐾 NexusAgent                    session: abc123    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  📋 Plan                                            │
│  ├── ✅ todo-1: Research Amiya profile              │
│  ├── 🔄 todo-2: Create amiya_profile.html           │
│  └── ⬜ todo-3: Verify HTML output                  │
│                                                     │
│  🔄 Handoff: planner → searchAgent                  │
│  🔍 WebSearchTool: "明日方舟 阿米娅"                   │
│  📊 3 results found                                 │
│                                                     │
│  🔄 Handoff: planner → codeAgent                    │
│  🔧 FileWriteTool → amiya_profile.html              │
│  📝 NotepadAppend: "Created HTML with 3 sources"     │
│                                                     │
├─────────────────────────────────────────────────────┤
│  💬 Input:  _____________________________________   │
└─────────────────────────────────────────────────────┘
```

**Intent Router —— 区分聊天和任务：**

```
┌─────────────┐
│ User Input  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Intent Router     │  ← LLM 判断意图
│ (confidence≥0.55) │
└──────┬───────┬───┘
       │       │
   "chat"   "workflow"
       │       │
       ▼       ▼
┌──────────┐ ┌──────────┐
│  Chat     │ │ Planner  │
│ Responder │ │ 流程     │
│ (轻量回复) │ │ (完整流程)│
└──────────┘ └──────────┘
```

```python
INTENT_ROUTER_PROMPT = """Classify the user input as "chat" or "workflow".
- chat: greetings, simple questions, conversational follow-ups
- workflow: tasks that need file operations, code, research, or multi-step work
If uncertain, choose workflow.
"""
```

**多轮对话 Session：**

```python
# session.json
{
  "session_id": "abc123",
  "turn_index": 3,
  "recent_turns": [
    {"turn": 1, "role": "user", "content": "帮我创建贪吃蛇"},
    {"turn": 2, "role": "assistant", "route": "workflow", "summary": "Created snake.py"},
    {"turn": 3, "role": "user", "content": "加一个计分功能"}
  ]
}
```

**飞书接入（展望）：**

```
用户 (飞书) → 飞书 Bot API → NexusAgent Backend → Agent 执行 → 结果回传飞书
```

### 🎬 演示效果

```
$ nexusagent

 🐾 NexusAgent v0.6.0 — Stage 6: TUI + Session

 Session: workspace-20250115-abc123
 Mode: inline approval | Checkpoint: light | Trace: on

 > 你好

 🗨️  Chat: 你好！我在。你可以继续提问，或者直接描述一个需要我完成的任务。

 > 帮我查一下明日方舟阿米娅，然后写一个 HTML 介绍页

 📋 Plan: Research & Build Amiya Profile Page
   ✅ todo-1: Search Amiya character info
   ⬜ todo-2: Build amiya_profile.html

 🔄 searchAgent searching...
 🔍 "明日方舟 阿米娅" → 5 results
 📋 Research: 阿米娅是罗德岛领袖...

 🔄 codeAgent working...
 🔧 FileWriteTool → amiya_profile.html (2.3KB)
 📝 Notepad: recorded key design decisions

 ✅ Verified: HTML file present, 3 sources cited

 💾 Checkpoint saved
 📊 Trace: 8.2s, 6 tool calls, 1 approval, 2 handoffs
```

### 💡 讲解要点

- **Intent Router 的意义**：不是所有输入都需要跑完整工作流。"你好"走 chat 分支，秒回；"帮我写代码"走 workflow 分支，完整执行
- **Session 是"记忆"的基础**：多轮对话靠 session 维持上下文，不是靠撑爆消息窗口
- **TUI 让过程可观测**：每个节点的状态、每次工具调用、每次 Handoff，都实时展示
- **事件驱动架构**：Agent 的每个操作都发 event，TUI 订阅 event 并渲染。解耦了逻辑和展示

### 🔍 核心代码

**Intent Router — [nodes.py:85](src/nexusagent/graph/nodes.py#L85)**

```python
def intent_router_node(state) -> dict:
    route = "workflow"  # ← 默认走 workflow
    try:
        response = create_model().invoke([SystemMessage(INTENT_ROUTER_PROMPT), HumanMessage(...)])
        parsed = _extract_json(str(response.content)) or {}
        candidate = str(parsed.get("route", "")).strip().lower()
        parsed_confidence = _coerce_confidence(parsed.get("confidence"))
        if candidate in {"chat", "workflow"} and parsed_confidence >= 0.55:  # ← 置信度阈值
            route = candidate
    except Exception:
        route = "workflow"  # ← 异常 fallback 到 workflow
    return {"intent_route": route, "intent_reason": reason, "intent_confidence": confidence}
```

> LLM 判断意图返回 JSON `{route, reason, confidence}`。confidence < 0.55 或异常时默认 workflow。

**入口工作流 — [workflow.py:49](src/nexusagent/graph/workflow.py#L49)**

```python
def build_entry_workflow():
    graph = StateGraph(NexusGraphState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("chat_responder", chat_responder_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges("intent_router", intent_route_fn, {
        "chat_responder": "chat_responder",
        "planner": END,        # ← 路由到 planner 时，控制权交给主工作流
    })
    graph.add_edge("chat_responder", END)
    return graph.compile()
```

> 入口图只做路由判断。chat → 直接回复；workflow → END，控制权交给 `build_complex_workflow`。

**多轮对话 Session — [agent.py:153](src/nexusagent/core/agent.py#L153)**

```python
def stream_session_events(task, *, session_workspace=None, ...):
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, task)        # ← 记录用户输入
    save_session(workspace, session)
    session_context = build_session_context(workspace, session)  # ← 构建会话上下文

    # 第一步：运行入口图（intent_router）
    for mode, event in build_entry_workflow().stream(entry_state, ...):
        if event.get("type") == "intent_decision": route = event.get("route")

    if route == "chat":
        append_assistant_turn(session, turn=turn, route="chat", content=response)
        save_session(workspace, session)
        return

    # 第二步：运行主工作流
    for event in _stream_complex_workflow(task=task, session=session, ...):
        yield event

    append_assistant_turn(session, turn=turn, route="workflow", content=final_answer)
    save_session(workspace, session)
```

> 每次 turn 记录到 session.json。`build_session_context` 提取最近 10 轮对话摘要 + workspace 文件清单，注入后续节点 prompt。

**TUI 审批弹窗 — [tui/approval.py:34](src/nexusagent/cli/tui/approval.py#L34)**

```python
class ApprovalGate:
    """线程同步：BashTool 线程等待 → TUI 弹窗 → 用户点击 → 释放"""
    request: ApprovalRequest
    decision: ApprovalDecision | None = None

    def resolve(self, approved: bool):
        self.decision = ApprovalDecision(approved=approved, reason=...)
        self._ready.set()  # ← 释放 BashTool 线程

    def wait(self) -> ApprovalDecision:
        self._ready.wait()  # ← 阻塞直到用户操作
        return self.decision

class ApprovalModal(ModalScreen[bool]):
    BINDINGS = [("y", "approve", "Approve"), ("n", "deny", "Deny")]
    # 显示：工具名、风险原因、工作区、完整命令
```

> BashTool 线程 → `approval_handler(request)` → 创建 `ApprovalGate` → TUI 线程弹出 `ApprovalModal` → 用户点击 → `gate.resolve()` → `gate.wait()` 返回 → 继续或拒绝。

### 🤖 Vibe Coding Prompt

**Step 1: 实现 Intent Router 入口图**

```
在 src/nexusagent/graph/workflow.py 中新增入口工作流图：

def build_entry_workflow():
    """意图路由图：判断用户输入是聊天还是任务"""
    graph = StateGraph(NexusGraphState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("chat_responder", chat_responder_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges("intent_router", intent_route_fn, {
        "chat_responder": "chat_responder",
        "planner": END,  # 路由到 planner 时，交给主工作流
    })
    graph.add_edge("chat_responder", END)
    return graph.compile()

在 src/nexusagent/graph/nodes.py 中实现两个新节点：

def intent_router_node(state) -> dict:
    """
    1. 用 LLM 判断意图：chat 还是 workflow
    2. 输入：INTENT_ROUTER_PROMPT + 用户输入 + session 上下文
    3. LLM 返回 JSON: {"route": "chat"|"workflow", "reason": "...", "confidence": 0.0-1.0}
    4. 如果 confidence < 0.55 或返回值无效，默认 workflow
    5. 返回 {intent_route, intent_reason, intent_confidence}
    """

INTENT_ROUTER_PROMPT = """You are the intent router for NexusAgent.

Classify the user's latest input into exactly one route:
- chat: greetings, thanks, identity/help questions, ordinary conceptual Q&A,
  or conversational messages that do not need workspace access.
- workflow: any request that needs creating/editing/reading files, running commands,
  installing packages, searching the web, checking the current project, verifying a
  result, or producing a concrete deliverable.

When session context is provided, use it only to understand whether the latest
input is a continuation of prior coding work. A short follow-up like "继续",
"修一下", or "运行测试" should be workflow if it refers to prior workspace work.

Return only JSON with this shape:
{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}

If uncertain, choose workflow.
"""

def chat_responder_node(state) -> dict:
    """
    轻量聊天分支，不调用任何工具：
    1. 用 LLM 直接回复
    2. 输入：CHAT_RESPONDER_PROMPT + 用户输入 + session 上下文
    3. 返回 {chat_response, final_answer}
    """

CHAT_RESPONDER_PROMPT = """You are NexusAgent's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the workflow route.

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""

def intent_route_fn(state) -> str:
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"
```

**Step 2: 实现多轮对话 Session**

```
在 src/nexusagent/core/session.py 中实现会话管理：

SESSION_ROOT = ".nexusagent/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000

def load_or_create_session(workspace: Path) -> dict:
    """加载或创建 session.json，包含 session_id, turn_index, recent_turns"""

def append_user_turn(session, content: str) -> int:
    """记录用户输入，返回 turn 编号"""

def append_assistant_turn(session, *, turn, route, content, summary="") -> None:
    """记录助手回复，route="chat"|"workflow" """

def save_session(workspace, session) -> dict:
    """保存 session.json，同时生成 SESSION_SUMMARY.md"""

def build_session_context(workspace, session=None) -> str:
    """构建 session 上下文字符串，供 intent_router 和 chat_responder 使用：
    - 包含 session_id, turn_index
    - workspace 文件清单（最近 30 个文件）
    - 最近 10 轮对话的摘要
    - 总长度不超过 MAX_SESSION_CONTEXT
    """

session.json 结构：
{
  "session_id": "uuid",
  "turn_index": 3,
  "recent_turns": [
    {"turn": 1, "role": "user", "content": "...", "timestamp": "..."},
    {"turn": 2, "role": "assistant", "route": "workflow", "content": "...", "summary": "..."},
    {"turn": 3, "role": "user", "content": "...", "timestamp": "..."}
  ],
  "created_at": "...",
  "updated_at": "..."
}
```

**Step 3: 实现多轮对话流**

```
在 src/nexusagent/core/agent.py 中新增 stream_session_events：

def stream_session_events(task, *, session_workspace=None, ...):
    """
    支持多轮对话的事件流：

    1. 加载或创建 Session
    2. 记录用户 turn
    3. 构建 session_context
    4. 运行入口图（intent_router → chat/workflow）
       - 如果 chat：直接回复，记录 assistant turn
       - 如果 workflow：运行 build_complex_workflow()，记录 assistant turn
    5. 每次 session turn 的 session_context 都包含：
       - 当前 workspace 文件清单
       - 最近对话摘要
       - 这样 Agent 就知道"之前聊过什么"
    """
```

**Step 4: 实现 TUI 界面**

```
在 src/nexusagent/cli/tui/app.py 中用 textual 实现 TUI：

class NexusAgentTuiApp(App[None]):
    """
    界面布局：
    ┌─────────────────────────────────────────────┐
    │ 🐾 NexusAgent                   session: xxx │  ← Header + 状态栏
    ├─────────────────────────────────────────────┤
    │                                             │
    │  [Plan] todo-1 ✅ todo-2 🔄 todo-3 ⬜      │  ← Plan 面板
    │                                             │
    │  🔧 FileWriteTool → app.py                  │  ← 事件流（可滚动）
    │  📝 NotepadAppend: "Created Flask app"      │
    │  🔄 Handoff: planner → codeAgent            │
    │  🔍 WebSearchTool: "Flask tutorial"         │
    │                                             │
    ├─────────────────────────────────────────────┤
    │  💬 Input: _________________________        │  ← 输入框
    └─────────────────────────────────────────────┘

    核心机制：
    1. 后台线程运行 stream_session_events
    2. 事件通过 AgentEventMessage(Message) 发送到 UI 线程
    3. UI 根据事件类型更新不同区域：
       - plan_snapshot → 更新 Plan 面板
       - tool_call / tool_result → 添加到事件流
       - handoff → 显示交接信息
       - checkpoint_saved → 显示检查点状态
       - approval_requested → 弹出审批弹窗
       - final_answer → 显示最终结果
    4. 输入框支持多轮对话，每次提交就是一个新的 session turn
    """

审批弹窗实现（src/nexusagent/cli/tui/approval.py）：

class ApprovalModal(ModalScreen[bool]):
    """
    弹窗显示：
    - 工具名：BashTool
    - 风险原因：Python package installation
    - 工作区路径
    - 完整命令
    - [Y] Approve / [N] Deny 按钮
    - 键盘快捷键：Y/Enter 批准，N/Escape 拒绝
    """

class ApprovalGate:
    """
    线程同步机制：
    - approval_handler 在 BashTool 线程中创建 ApprovalGate
    - 发送 ApprovalRequestedMessage 到 TUI
    - TUI 弹出 ApprovalModal
    - 用户点击后调用 gate.resolve(approved=True/False)
    - BashTool 线程通过 gate.wait() 阻塞等待决策
    """
```

**Step 5: Logo 和品牌**

```
在 src/nexusagent/cli/tui/logo.py 中实现 ASCII Art Logo：

 🐾 NexusAgent
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 6 · MultiAgent + Context/Harness
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 启动时用 rich 渲染，带颜色和动画效果。
```

**Step 6: 测试运行**

```bash
# 启动 TUI 模式
nexusagent

# 多轮对话测试：
> 你好                              # 走 chat 分支
> 帮我写一个 Flask 应用              # 走 workflow 分支
> 加一个用户登录功能                  # 继续在同一 workspace，session 感知上下文

# 审批测试：当 codeAgent 执行 pip install 时，TUI 会弹出审批弹窗
# Checkpoint 测试：Ctrl+C 中断后，用 --resume 恢复
# Trace 测试：查看 .nexusagent/traces/*/timeline.md
```

---

## 总结：六个阶段的进化路径

```
Stage 1          Stage 2           Stage 3           Stage 4           Stage 5           Stage 6
ReAct            LangGraph         MultiAgent        Context Eng       Harness Eng       Claw TUI
─────────────────────────────────────────────────────────────────────────────────────────────────
Actor            Planner           Planner           Planner           Planner           Intent Router
   │                │               │  │               │  │               │  │               │  │
   │             Actor          searchAgent       searchAgent       searchAgent       Chat Responder
   │                │               │               │  │               │  │               │  │
   │             Verifier       codeAgent         codeAgent         codeAgent          Planner
   │                │               │               │  │               │  │               │  │
   │                │            Verifier    Context Monitor     Context Monitor    searchAgent
   │                │               │               │               │  │               │  │
   │                │               │          Compressor         Compressor        codeAgent
   │                │               │               │               │  │               │  │
   │                │               │               │            Verifier           Verifier
   │                │               │               │               │               │  │
   │                │               │               │               │            Session Mgr
   │                │               │               │               │               │  │
   │                │               │               │               │            Checkpoint
   │                │               │               │               │               │  │
   │                │               │               │               │            Trace
   │                │               │               │               │               │  │
   │                │               │               │               │            Approval
```

**每个阶段增加的核心能力：**

| 阶段 | 核心增量 | 解决的问题 |
|------|---------|-----------|
| 1. ReAct | 基础工具调用 | Agent 能干活 |
| 2. LangGraph | Plan → Execute → Verify | 干活有规划、有验证 |
| 3. MultiAgent | 专家分工 + Handoff | 复杂任务需要不同能力 |
| 4. Context Eng | 压缩 + Notepad + 分层 Memory | 长程任务不丢上下文 |
| 5. Harness Eng | 审批 + 检查点 + Trace | 安全可控、可恢复、可观测 |
| 6. Claw TUI | 交互界面 + 多轮对话 | 用户体验、过程可视化 |

### 核心代码速查表

| 文件 | 核心函数 | 阶段 | 一句话 |
|------|---------|------|--------|
| [registry.py:13](src/nexusagent/tools/registry.py#L13) | `build_tools` / `build_read_only_tools` | 1-6 | 工具注册：全量 vs 只读 |
| [workflow.py:24](src/nexusagent/graph/workflow.py#L24) | `build_complex_workflow` | 2-6 | 图结构定义：节点 + 边 + 条件路由 |
| [workflow.py:49](src/nexusagent/graph/workflow.py#L49) | `build_entry_workflow` | 6 | 入口图：intent_router → chat/workflow |
| [state.py:60](src/nexusagent/graph/state.py#L60) | `NexusGraphState` | 2-6 | 图共享状态，`messages` 用 `add_messages` |
| [nodes.py:85](src/nexusagent/graph/nodes.py#L85) | `intent_router_node` | 6 | LLM 判断意图：chat 还是 workflow |
| [nodes.py:152](src/nexusagent/graph/nodes.py#L152) | `planner_node` | 2-6 | Supervisor：调用 TodoWrite/Search/Code 工具 |
| [nodes.py:221](src/nexusagent/graph/nodes.py#L221) | `verifier_node` | 2-6 | 只读工具验证 + JSON 判断 pass/fail |
| [nodes.py:412](src/nexusagent/graph/nodes.py#L412) | `verifier_route` | 2-6 | passed→final / 达上限→final / 否则→planner |
| [nodes.py:473](src/nexusagent/graph/nodes.py#L473) | `_build_planner_tools` | 3 | Planner 的 3 个调度工具 |
| [nodes.py:536](src/nexusagent/graph/nodes.py#L536) | `_call_search_agent_tool` | 3 | Handoff: planner→searchAgent |
| [nodes.py:558](src/nexusagent/graph/nodes.py#L558) | `_call_code_agent_tool` | 3 | Handoff: planner→codeAgent |
| [search_agent.py:17](src/nexusagent/agents/search_agent.py#L17) | `run_search_agent` | 3 | 搜索专家：只绑 WebSearchTool |
| [code_agent.py:21](src/nexusagent/agents/code_agent.py#L21) | `run_code_agent` | 3 | 编码专家：全量工具 + TodoUpdate + Memory |
| [memory.py:39](src/nexusagent/graph/memory.py#L39) | `build_layered_memory` | 4 | 三层 Memory：Rules + Working + History |
| [nodes.py:327](src/nexusagent/graph/nodes.py#L327) | `context_monitor_node` | 4 | 估算 token 数，决定是否压缩 |
| [nodes.py:356](src/nexusagent/graph/nodes.py#L356) | `context_compressor_node` | 4 | LLM 摘要 + RemoveMessage 清空历史 |
| [nodes.py:616](src/nexusagent/graph/nodes.py#L616) | `_compress_context_with_model` | 4 | 压缩核心：给 LLM 全量消息，返回摘要 |
| [notepad_tool.py:21](src/nexusagent/tools/notepad_tool.py#L21) | `append_notepad` | 4 | 追加写入 NOTEPAD.md，压缩后不丢失 |
| [approval.py:44](src/nexusagent/core/approval.py#L44) | `classify_command_risk` | 5 | 正则匹配高风险命令 |
| [bash_tool.py:224](src/nexusagent/tools/bash_tool.py#L224) | `_resolve_approval` | 5 | 三模式分流：auto/deny/inline |
| [bash_tool.py:154](src/nexusagent/tools/bash_tool.py#L154) | `run_bash` | 5 | 完整流程：校验→危险拦截→审批→执行 |
| [checkpoint.py:47](src/nexusagent/core/checkpoint.py#L47) | `CheckpointManager.save` | 5 | light/strict/off 三级检查点 |
| [trace.py:29](src/nexusagent/core/trace.py#L29) | `TraceRecorder` | 5 | 逐事件记录 + 统计 + timeline.md |
| [agent.py:153](src/nexusagent/core/agent.py#L153) | `stream_session_events` | 6 | 多轮对话：session→router→workflow→session |
| [session.py:100](src/nexusagent/core/session.py#L100) | `build_session_context` | 6 | 从 session.json 构建上下文字符串 |
| [tui/approval.py:34](src/nexusagent/cli/tui/approval.py#L34) | `ApprovalGate` | 6 | 线程同步：等待人类审批决策 |

---

## 视频拍摄建议

### 每期视频结构

1. **开场**：这期我们要解决什么问题？（1 分钟）
2. **架构设计**：画图讲解，先有图再写代码（3-4 分钟）
3. **Prompt 编写**：展示核心 Prompt，解释设计意图（2-3 分钟）
4. **实机演示**：运行 nexusagent，展示效果（3-5 分钟）
5. **问题修复**：故意展示一个 vibe 中的问题，然后修复（2-3 分钟）
6. **总结**：这阶段学到了什么，和上一阶段有什么区别（1 分钟）

### 演示任务选择

| 阶段 | 推荐演示任务 | 为什么 |
|------|-------------|--------|
| 1 | 创建贪吃蛇游戏并运行 | 简单、可视化、容易出错（方便演示修复） |
| 2 | TDD 实现 Game of Life | 天然需要"规划→实现→验证"循环 |
| 3 | 查阅阿米娅资料写 HTML | 必须搜索 + 实现，体现 MultiAgent 价值 |
| 4 | 搭建 Flask 后台系统 | 文件多、步骤多，容易触发上下文压缩 |
| 5 | 搭建需安装依赖的系统 | pip install 触发审批，长程触发 checkpoint |
| 6 | 多轮对话 + 复杂任务 | 展示 chat/workflow 路由 + TUI 可视化 |

### Vibe 常见问题 & 修复演示

每个阶段可以故意展示一些 Agent 犯的错误，然后讲解如何修复：

1. **ReAct 循环死循环**：模型一直重复同一个工具调用 → 加 max_loops 限制
2. **Planner 输出不是合法 JSON**：Verifier 解析失败 → 加 JSON 解析容错 + fallback
3. **searchAgent 搜索不到结果**：查询词太模糊 → Prompt 加"使用多个搜索词"
4. **Context 被压缩后 Agent 失忆**：忘记之前做过什么 → Notepad 机制 + History Summary
5. **BashTool 执行危险命令**：直接 rm -rf → 风险分类 + 人类审批
6. **Intent Router 误判**：把"帮我看看这个 bug"判为 chat → 调整 confidence 阈值

---

## Vibe Coding 使用指南

> 每个阶段的 🤖 Vibe Coding Prompt 是你跟着视频写代码时输入给 AI 编程助手的 Prompt。
> 使用方式：打开一个全新的对话，把对应 Step 的 Prompt 复制粘贴进去，让 AI 帮你生成代码框架，
> 然后你根据视频讲解理解每一部分，微调细节，最终运行测试。

### 使用技巧

1. **按 Step 顺序执行**：每个 Step 都依赖前一步的代码，不要跳步
2. **每次 Vibe 前先说明上下文**：如果 AI 助手丢失了上下文，先贴上当前文件结构让它了解项目状态
3. **生成的代码要审查**：Vibe Coding 不是无脑复制，理解每一行再使用
4. **遇到错误先自己修**：视频里展示的"Vibe 问题修复"环节就是教你这个技能
5. **测试先行**：每个 Step 完成后先跑通测试再进下一步

### Vibe 常见问题速查

| 问题 | 原因 | 修复 Prompt |
|------|------|------------|
| 模型一直重复同一工具调用 | max_loops 不够或 Prompt 缺少"结束"指令 | "在 Prompt 末尾加上：'当你认为任务完成时，不要调用任何工具，直接给出总结。'" |
| Planner 输出不是合法 JSON | 模型没遵循格式要求 | "在 Planner Prompt 中强调：'你必须返回且仅返回合法的 JSON 对象，不要包含任何其他文本。'" |
| searchAgent 搜索不到结果 | 查询词太模糊 | "在 SearchAgent Prompt 中加上：'如果第一次搜索结果不理想，尝试用不同的关键词重新搜索。'" |
| Context 压缩后 Agent 失忆 | 压缩丢失了关键上下文 | "检查 build_layered_memory 是否正确包含了 Notepad 和 History Summary" |
| BashTool 执行了危险命令 | 缺少审批机制 | "检查 classify_command_risk 的正则是否覆盖了该命令模式" |
| Intent Router 误判 | confidence 阈值过高或过低 | "调整 INTENT_ROUTER_PROMPT 中的分类规则或 0.55 阈值" |

---

> 🐾 NexusAgent — 从一行代码到一个 Agent，六步走完。
