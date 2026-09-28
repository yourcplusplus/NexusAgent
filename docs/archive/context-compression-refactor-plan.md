# NexusAgent 上下文压缩重构计划

> 版本:v1 草案(待确认) · 日期:2026-09-26
> 范围:`src/nexusagent/graph/`(nodes / memory / workflow / state)及 `tools/`、`agents/` 相关接入点
> 状态:**仅分析与计划,未修改任何代码**

---

## 0. TL;DR

当前 `context_compressor` 在 token 超限时,把全部消息转录交给一次 LLM 摘要,然后用 `RemoveMessage(REMOVE_ALL_MESSAGES)` 一次性清空,只留一条摘要消息。这条链路上有**三层叠加的信息损耗**(进摘要前每条消息截 2000 字符 → 摘要输出再截 1600/2200 字符 → HISTORY_SUMMARY 覆盖式重写导致"摘要的摘要"迭代衰减),且 Git 状态、文件路径等关键信息**从未被结构化跟踪**,能否幸存完全取决于摘要模型这一次调用的发挥。

重构方案的核心原则是**"确定性优先于生成式"**:关键信息(TODO / 文件路径 / Git 状态 / 落盘产物指针)由代码渲染成固定结构块,压缩时原样保留、永不截断;LLM 只负责对被逐出的窗口片段做增量叙事摘要。四个机制:

1. **滑动窗口压缩器**:按"节点执行轮"分组逐出,保留最近 N 轮,杜绝 `REMOVE_ALL_MESSAGES`;
2. **关键信息白名单(critical_context)**:state 中的结构化注册表 + FileLedger,采集/渲染/压缩不变式三层保障;
3. **agent_status_bar 节点**:每次进入 LLM 节点前刷新 git/工作区/todo/token 预算快照并注入 prompt;
4. **工具输出统一落盘**:四个 ToolMessage 构造点统一接入 spill 机制,超长输出写盘、inline 留头尾摘要 + 文件指针。

实施分 6 个 Phase,每个机制独立可验收、可回退。

---

## 1. 现状梳理

### 1.1 图拓扑与消息流(workflow.py:24-46)

```
START → planner → context_monitor ─┬─(超限)→ context_compressor → 目标节点
                                   └─(未超限)→ 目标节点(context_next_node)
verifier → context_monitor(循环回环)
final → END
```

`NexusGraphState.messages` 是 `Annotated[list[BaseMessage], add_messages]`(state.py:63),由 reducer 累加合并。

### 1.2 一个关键事实:messages 转录是"只写"的

阅读 planner / verifier / code_agent 的实现可以发现:

- planner 每次执行都**新建** prompt(`nodes.py:169-172`:`[SystemMessage(PLANNER_PROMPT), HumanMessage(_planner_input(...))]`),不回放 `state["messages"]`;
- verifier 同理(`nodes.py:237-240`),codeAgent 同理(`code_agent.py:46-49`),searchAgent 同理;
- 节点把自己的 `produced_messages`(模型响应 + ToolMessage)**追加**进共享 `messages`;
- 因此 `state["messages"]` 的消费者只有两个:
  1. `estimate_context_tokens`(`nodes.py:460-470`)—— 做阈值判断;
  2. `_compress_context_with_model`(`nodes.py:616-634`)—— 作为摘要输入。

**结论:压缩清空 messages 并不会让下一个节点的 prompt"变瞎"(节点 prompt 由结构化 state 字段 + 分层记忆重建),真正被销毁的是转录中独有的原始细节(完整工具输出、精确报错栈、读过的文件内容片段),以及它们进入后续 prompt 的唯一通道——那一次摘要。** 这解释了为什么问题在长链路任务中才爆发:短任务里结构化字段(`research_notes`、`code_agent_summary`、`last_error`…)够用;长链路中大量细节只存在于转录里。

### 1.3 当前压缩链路(nodes.py:356-409)

```
context_monitor(估 token,>= limit?) 
  → context_compressor:
      1. build_layered_memory(第 1 次)
      2. _compress_context_with_model:
         - build_layered_memory(第 2 次,重复构建)
         - 把「context_summary + memory + 全部 messages(每条截 2000 字符)」JSON 化喂给 LLM
         - 失败 → _fallback_compression(直接丢弃转录,只拼接结构化字段)
      3. _format_compressed_context:模型摘要 + state 里的 task/todos/criteria 等拼成 JSON
      4. persist_history_summary:覆盖写 HISTORY_SUMMARY.md
      5. 返回 [RemoveMessage(REMOVE_ALL_MESSAGES), summary_message]  ← 全量清空
```

---

## 2. 问题分析

### P0 —— 直接导致长链路任务丢关键信息

**P0-1 全量清空 + 单次 LLM 摘要是唯一信息通道,无确定性保底。**
`nodes.py:393`。摘要模型能否保住文件路径、进行中的 TODO、分支名,完全取决于这一次调用;prompt(stage4.py)只是自然语言要求"Keep everything needed",无结构化校验。`_important_files_from_state`(nodes.py:700-713)能从 verification_commands / 摘要文本里正则抽路径,但**只在 fallback 路径使用**,模型路径成功返回时不与该结果合并——即模型漏了路径,没人兜底。

**P0-2 三层叠加截断,信息在到达下一个节点前已被物理删除。**
- 进摘要器前:`_message_snapshot` 每条消息截 2000 字符(nodes.py:689);
- 摘要输出后:`context_summary` 截 1600、`history_summary` 截 2200(memory.py:25-36);
- 截断从尾部切,而路径列表、待办、后续步骤恰恰通常排在文本末尾,最先被切掉。

**P0-3 "摘要的摘要"迭代衰减。**
`persist_history_summary`(memory.py:123-130)**覆盖式**重写 HISTORY_SUMMARY.md;下一轮压缩的输入又包含旧摘要(`context_summary` / `history_summary` 都在 payload 里),LLM 对旧摘要再摘要 → 细节逐轮漂移、丢失,不可逆。

**P0-4 Git 状态从未被跟踪。**
没有任何 state 字段、任何工具钩子记录分支 / 脏文件 / ahead-behind。压缩只是"没有丢失它"的问题,而是系统**从来不知道它**。文件路径跟踪同样零散:只靠对 commands/summary 文本的正则。

**P0-5 长工具输出:落盘雏形有,但指针随清空一起丢失,且仅 bash 一家。**
- `bash_tool._format_captured_output`(bash_tool.py:390-406)超限时会把完整 stdout/stderr 写到 `.nexusagent/bash-outputs/*.log` 并返回 `stdout_path`——但 inline 预览**只有头部**(`stdout[:max_output_chars]`),无尾部;报错的关键行(traceback 末尾)恰恰在尾部;
- file_tools 的 diff 截 4000(file_tools.py:129,173)、web_search content 截 1200(web_search_tool.py:50)、grep matches 截断(grep_tool.py:66)——**直接丢弃,不落盘**;
- 落盘路径只存在于 ToolMessage 里,`REMOVE_ALL_MESSAGES` 后指针丢失:文件还在盘上,但 Agent 不知道去哪找、也不知道存在过。

### P1 —— 机制性缺陷

**P1-1 压缩触发与真实上下文脱节。**
默认阈值 `DEFAULT_CONTEXT_TOKEN_LIMIT = 400000`(nodes.py:30),而每次真实 LLM 调用的 prompt 只是"SystemMessage + 单条 HumanMessage(结构化记忆)"——远小于该值。`estimate_context_tokens` 度量的是"把全部 messages + memory payload 塞进一个假想 prompt"的规模,**这个 prompt 从未被真正发出过**。结果是:(a) 默认配置下压缩几乎永不触发,messages 无界增长;(b) 若调低阈值,压缩在"真实调用还很安全"时就发生,白白销毁转录。

**P1-2 压缩器自身的上下文爆炸。**
触发压缩意味着转录很大,而 `_compress_context_with_model` 把**全部**消息(每条 2000 字符)一次性喂给摘要模型——压缩调用本身可能超限、超时或昂贵。失败后 fallback(P0-1)则是零审查全丢弃。

**P1-3 `REMOVE_ALL_MESSAGES` 销毁 checkpoint / 审计价值。**
LangGraph checkpointer 线程历史中的转录被清;事后 resume、回放、调试"当时 Agent 看到了什么"不可复现。会话连续性只能靠 session.json 的 recent_turns(那只有用户↔助手层面的内容)。

**P1-4 无梯度悬崖。**
阈值前毫无作为,阈值后一次性大动作;压缩前后 Agent 可见信息量断崖式变化,行为不可预期。

### P2 —— 小问题(顺手修)

- `context_compressor_node` 与 `_compress_context_with_model` 各构建一次 `build_layered_memory`(nodes.py:360 与 617),重复;
- `estimate_context_tokens` 每次调用 `create_model()` 做 tokenize,monitor 每圈都跑,浪费;
- 压缩时对结构化字段的静默剁短(`research_notes` 1200、handoffs 留 6 条,nodes.py:376-380)与摘要截断叠加,无事件告知。

### 2.1 值得保留的现有资产

- **分层记忆架构**(rules / working_memory / history_summary_store)是好底座,重构在其上扩展而非推翻;
- TODO.md / NOTEPAD.md / HISTORY_SUMMARY.md 的落盘职责划分清晰;
- bash_tool 已有落盘雏形与 `.nexusagent/` 目录约定,可推广为统一机制;
- `session.py` 的 `recent_turns` + `_compact_session`(session.py:229-243)已经是"滑动窗口 + 旧内容折叠进 summary"的参考实现——本轮重构就是把同样的模式搬到 graph messages 层。

---

## 3. 重构总体设计

### 3.1 设计原则

1. **确定性优先于生成式**:白名单内容(TODO/路径/Git/产物指针)由代码从结构化 state 渲染,不经过 LLM 转述,不受字符截断;LLM 只做叙事性增量摘要。
2. **增量优先于全量**:只摘要"本次被逐出的窗口片段",输入规模小、可预期;历史叙事滚动合并而非反复重述。
3. **状态外置**:关键信息存 state 结构化字段 + 落盘文件;messages 转录只是可再生的缓存,逐出时"指针化"而非"丢弃"。
4. **单一 choke point**:工具输出落盘、状态注入都收敛到统一入口,避免四个 ToolMessage 构造点各写一套。

### 3.2 目标架构(压缩路径视角)

```
                      ┌──────────────────────────────────────────────┐
                      │  state(压缩后必存活,结构化、不截断)          │
                      │  critical_context:                           │
                      │    goal / constraints / todos_digest         │
                      │    files(FileLedger) / git / artifacts       │
                      │  env_status(status_bar 快照)                  │
                      │  history_summary(滚动叙事,封顶折叠)          │
                      └──────────────────────────────────────────────┘

 messages: [ 保留窗口:最近 N 轮(组边界完整) ]  ←── 逐出时:
                    │                            被逐出组 → 增量 rollup(LLM,只喂这一段)
                    │                            落盘产物指针 → artifacts 登记
                    ▼
          planner / verifier / code_agent prompt
          = 白名单渲染块(确定性) + 状态栏块 + 滚动叙事 + 最近窗口
```

---

## 4. 四个机制的详细设计

## 4.1 机制一:滑动窗口压缩器(替换 REMOVE_ALL_MESSAGES)

### 4.1.1 消息分组

以"**一个节点的一次执行产出的消息序列**"为一组(round)。planner 一次工具循环、verifier 一次工具循环各为一组;组内天然保持 `AIMessage(tool_calls)` ↔ `ToolMessage` 配对完整。

实现:各节点构造消息时**显式赋 id**,前缀编码组标识:

```
planner-a1-0001, planner-a1-0002, ...   (planner 第 1 次 attempt)
verifier-a1-0001, ...                    (verifier 第 1 次 attempt)
codeagent-a2-0007, ...                   (第 2 次 attempt 内第 7 条)
```

同时在 state 新增 `message_groups: list[MessageGroup]`:

```python
class MessageGroup(TypedDict, total=False):
    group_id: str          # "planner-a1"
    node: str              # planner / verifier / ...
    attempt: int
    message_ids: list[str]
    token_estimate: int
    created_at: str
```

节点返回时同时返回增量(由于该字段无 reducer,节点读取最新值后返回"完整新列表")。

### 4.1.2 窗口策略

逐出条件(两者取更严):

- **轮数**:保留最近 `KEEP_ROUNDS`(默认 6)组;
- **token**:窗口 token 估算 ≤ `limit × MAX_WINDOW_RATIO`(默认 0.30)。

即:组数不多但单组巨大(如一次超长 codeAgent 循环)时由 token 约束兜底;反之由轮数约束兜底。

### 4.1.3 逐出实现

```python
def context_compressor_node(state):
    groups, keep = plan_window(state["message_groups"], state["messages"],
                               keep_rounds=cfg.keep_rounds,
                               max_window_tokens=cfg.limit * cfg.ratio)
    evicted_ids = flatten(g.message_ids for g in groups[: -keep if keep else 0])
    # 防御:若保留窗口内存在 ToolMessage,其 tool_call_id 指向被逐出的 AIMessage
    #       → 将对应组一并保留(组边界错位时兜底)
    evicted_messages = [m for m in state["messages"] if m.id in evicted_ids]

    narrative = rollup_evicted(state, evicted_messages)      # 机制一之增量摘要,见 4.1.4
    state_updates = merge_history_summary(state, narrative)  # 滚动合并,见 4.1.5

    return {
        "messages": [RemoveMessage(id=i) for i in evicted_ids],   # ← 逐条逐出,绝不用 REMOVE_ALL_MESSAGES
        "message_groups": groups_after_eviction,
        **state_updates,
    }
```

要点:

- **逐出后不注入"摘要消息"进窗口**(与现状不同)。滚动叙事存放在 `history_summary` 字段并进入各节点 prompt(见 4.1.5),messages 保持纯净的"最近事实"。这避免摘要消息在后续窗口计算中反复被逐出、再被摘要(衰减源头之一)。
- 逐出**幂等、可重入**:压缩失败(LLM rollup 异常)时,本轮只做逐出 + 用确定性 rollup(见 4.1.4 fallback)记录,绝不回退到全清空。

### 4.1.4 增量 rollup(取代全量重述)

- 输入:**仅被逐出的消息组** + 当前 `history_summary` 的叙事段(不是全部 messages);
- 输出:追加式叙事(narrative),新增信息在前、压缩旧事为单行;
- prompt 更名 `CONTEXT_ROLLUP_PROMPT`(stage4.py 重写):明确"输入是被逐出的旧窗口片段,产出面向未来执行的连续叙事,不要重复已知白名单内容";
- **fallback(确定性,不经 LLM)**:对被逐出组提取——每条 ToolMessage 的 `name + ok/exit_code + 产出的文件路径(正则) + artifact 指针`,每条 AIMessage 的首 200 字符,拼成结构化条目列表。保证 LLM 不可用时零丢失地保住骨架信息。

### 4.1.5 滚动合并与防衰减

```
history_summary = render_critical_context(state)      # 确定性白名单块,永不折叠截断
                 + narrative(滚动叙事段,封顶 2200 字符,超限折叠最旧)
```

- 白名单块由代码每次重新渲染(**新鲜覆盖**而非累积),天然不会衰减;
- 叙事段超限时折叠最旧的条目为单行("轮次 a1-a3:完成 X,失败原因 Y"),而非整段重述;
- `HISTORY_SUMMARY.md` 改为两段式写入:`## Critical Context(白名单渲染)+ ## Narrative(最新叙事)`。覆盖写不再有衰减问题,因为衰减源(对旧叙事再摘要)已消除。

### 4.1.6 触发机制改造(修 P1-1 / P1-4)

- token 估算改为**增量计数**:节点返回 `message_groups` 时带上该组 token_estimate(对 produced_messages 一次 tokenize),monitor 只做求和 + 常量(payload 开销),不再每次 `create_model()` 全量计数;
- 新增**软阈值事件**:达到 70% 时 writer 发 `context_pressure` 事件(TUI 可见),达到 100% 才压缩——从悬崖变成有预警的斜坡;
- `DEFAULT_CONTEXT_TOKEN_LIMIT` 建议默认值降为 `200000`(仍可用环境变量覆盖),并在 README/配置说明中解释口径变化:该值度量的是"窗口 + 记忆 payload"的目标规模。

---

## 4.2 机制二:关键信息白名单(critical_context)

### 4.2.1 数据结构(state.py 新增)

```python
class FileEntry(TypedDict, total=False):
    path: str            # 相对 workspace
    op: str              # read / write / edit / create / delete / artifact
    at: str              # ISO 时间
    via: str             # 产生来源:bash / file_tools / spill / status_bar

class CriticalContext(TypedDict, total=False):
    goal: str                     # 用户原始 task,不可变
    constraints: list[str]        # 验收标准等硬约束(来自 acceptance_criteria)
    todos_digest: list[dict]      # 当前 todos 的浅拷贝(渲染时取快照)
    files: list[FileEntry]        # FileLedger,LRU 上限 50
    git: dict                     # {branch, dirty_count, status_short, ahead_behind, is_repo}
    artifacts: list[dict]         # 落盘产物 {path, tool, producing_input, lines, bytes}
```

state 新增字段 `critical_context: CriticalContext`(与 `env_status`、`message_groups` 一起)。

### 4.2.2 采集点(谁写入)

| 板块 | 采集点 | 时机 |
|---|---|---|
| goal / constraints | planner 首次定稿计划时 | 一次 |
| todos_digest | `persist_todos` / `update_todo`(todo_tool.py)内同步刷新 | 每次 todo 变更 |
| files(FileLedger) | `RuntimeState` 新增 `touched_files: dict[Path, FileEntry]`;file_tools 读写、bash 落盘、grep 命中文件处调用 `runtime.record_touch(path, op, via)` | 每次工具触文件 |
| git | agent_status_bar 节点(机制三) | 每次节点转移 |
| artifacts | spill_tool_output(机制四) | 每次落盘 |

FileLedger 放在 `RuntimeState`(core/state.py)而非 graph state,因为工具层只有 runtime 句柄;`build_layered_memory` / status_bar 负责把它投影进 `critical_context.files`(LRU 截 50 条)。

### 4.2.3 渲染规则(谁消费)

新增 `render_critical_context(state) -> str`,输出固定结构块(**头尾都不截断,用数量上限控制体积**):

```
=== CRITICAL CONTEXT (auto-pinned, never dropped by compression) ===
[Goal] <task 原文>
[Constraints]
  - <acceptance_criteria 每条>
[TODOs]
  - todo-1 [in_progress] 实现 X —— note
[Files]
  - src/a.py (write, 09-26 14:02, via file_tools)
  - report.html (create, via bash)
[Git]
  branch: feat/context-ctx | dirty: 3 | ahead/behind: 2/0
  M src/graph/nodes.py | ?? out.log
[Artifacts on disk]
  - .nexusagent/tool-outputs/bash-...json (1,204 lines, from `pytest -x`)
=== END CRITICAL CONTEXT ===
```

注入位置(全部为代码拼接,非 LLM 生成):

- `_planner_input`(nodes.py:716)、`_verifier_input`(nodes.py:727)、`_code_agent_input`(code_agent.py:144)及 search_agent 等价物 —— **置于最前**,确保任何后续截断都先切叙事而非白名单;
- `context_compressor` 产出的 `history_summary` / `HISTORY_SUMMARY.md` 的固定头部。

### 4.2.4 压缩不变式(防丢保底)

压缩节点最后执行校验:最终 `history_summary` 必须包含白名单渲染块;若 LLM rollup 异常或输出被截导致缺失,则**用确定性渲染块直接兜底整体替换叙事段**。同时 `render_critical_context` 的输出不进入任何 `_short_text` 截断路径(`MAX_TEXT_CHARS` 表显式排除该块)。

---

## 4.3 机制三:agent_status_bar 节点

### 4.3.1 图拓扑(workflow.py 变更)

现状 `monitor → (compressor → target | target)` 改为 monitor 与 compressor 之后**统一经过状态栏**:

```
START → planner → context_monitor ─┬→ context_compressor → agent_status_bar → target
                                   └→ agent_status_bar → target
verifier → context_monitor(不变)
agent_status_bar --(context_next_node)--> planner | verifier | final
```

效果:每次进入 LLM 节点(verifier / 重试的 planner / final 前)恰好刷新一次环境快照;`context_compressor_route` 的职责并入 `agent_status_bar` 的路由。

### 4.3.2 采集内容与实现

新增 `src/nexusagent/graph/status_bar.py`:

```python
def refresh_env_status(runtime, state) -> dict:   # 写入 state.env_status
    git      = _git_snapshot(runtime)        # branch / porcelain / ahead-behind,非 git 仓库 → {"is_repo": False}
    files    = _workspace_delta(runtime)     # 复用 workspace_manifest,对比上次快照给出 新增/修改/删除
    todos    = 概要(id+status 计数)
    budget   = {token_estimate, limit, window_groups, pressure}   # 来自 monitor 的缓存计数
    jobs     = 后台任务(.nexusagent/background 下活跃 job)
    artifacts= 落盘产物计数
```

- **TTL 缓存**:git 子进程结果缓存 5 秒(`NEXUS_STATUS_BAR_TTL_SECONDS`),工具循环内高频刷新不产生重复开销;非 git 仓库优雅降级为一行 `(not a git repo)`;
- 预算块 ≤ 500 token,渲染模板为代码常量(不经 LLM);
- writer 发 `{"type": "status_bar", ...}` 事件,TUI 可展示(类似 IDE 状态栏)。

### 4.3.3 两级注入

1. **图级**(节点职责):`env_status` 渲染进 `_planner_input` / `_verifier_input` 等的"Environment status"小节;
2. **循环级**(长工具会话防陈旧):planner / verifier / code_agent 的工具循环内,每轮执行工具后检查 TTL,若距上次刷新超时则重新渲染并**追加一条轻量 SystemMessage**(content 前缀 `[STATUS ...]`)。该消息属于当前组,窗口逐出时自然带走,不污染白名单。

> 说明:循环级注入是对函数内部循环的小改动(非图节点);图级节点满足"每次调用 LLM 前(节点粒度)动态注入"的要求,循环级是补充,两者共用同一个 `status_bar.py` 渲染器。

---

## 4.4 机制四:工具输出统一落盘 + 头尾摘要

### 4.4.1 现状差距回顾

仅 bash 有落盘且 head-only;file diff / grep / web content 直接截断丢弃;落盘指针不登记。

### 4.4.2 统一入口

新增 `src/nexusagent/tools/output_sink.py`:

```python
def spill_tool_output(runtime, tool_name, result: dict, *,
                      inline_chars=2000, head_lines=30, tail_lines=20) -> dict:
    """result 序列化后 <= inline_chars → 原样返回;
       超限 → 全量写 .nexusagent/tool-outputs/{tool}-{time_ns}.json,
       inline 替换为:标量字段(ok/command/exit_code/error...)
         + 主文本字段(stdout/content/diff/matches 按优先级探测)的头 N 行 + 尾 M 行
         + '[... X lines omitted; full output: {path}]'
       同时 runtime.record_touch(path, op="artifact") 登记到 FileLedger/critical_context.artifacts"""
```

### 4.4.3 接入点(全部 ToolMessage 构造处收敛为一个 helper)

| 位置 | 现状 | 改造 |
|---|---|---|
| nodes.py:588(`_execute_planner_tool`) | `json.dumps(result)` 直塞 | 改用 `make_tool_message(call, result, runtime)` |
| nodes.py:609(`_execute_read_only_tool`) | 同上 | 同上 |
| code_agent.py:125(`execute_code_agent_tool`) | 同上 | 同上 |
| search_agent.py:111 | 同上 | 同上 |

`make_tool_message` = spill_tool_output + ToolMessage 构造 + ToolMessage 元数据带 `artifact_path`(顺带修 P2:四个点共用一套逻辑)。`_tool_result_event` 同步带上 artifact 路径,TUI 可展示"完整输出已存盘"。

### 4.4.4 bash_tool 对齐

`_format_captured_output` 的头预览改为头+尾(复用 output_sink 的格式化函数),保留现有 `stdout_path` 语义;`.nexusagent/bash-outputs/` 与新的 `.nexusagent/tool-outputs/` 目录可并存(bash 原逻辑少动),但都要登记 artifacts。

### 4.4.5 回读保障

- 头尾摘要中的路径是 workspace 相对路径,现有 grep / tail(`_handle_tail_command`)工具即可回读,无需新工具;
- 白名单 `[Artifacts on disk]` 段保证压缩后指针不丢(P0-5 修复);
- 清理策略:artifacts 登记上限 100 条 LRU;文件保留整个 session,workspace 生命周期文档已有说明。

---

## 5. 实施阶段划分

> 每个 Phase 独立可合并、可回退;Phase 1 收益立竿见影且零图结构变更。

| Phase | 内容 | 主要改动文件 | 验收标准 |
|---|---|---|---|
| **1. 工具落盘** | output_sink + 4 个接入点 + bash 头尾对齐 | tools/output_sink.py(新)、tools/bash_tool.py、graph/nodes.py、agents/code_agent.py、agents/search_agent.py、core/state.py(record_touch) | 构造 100KB 假输出的工具调用,ToolMessage inline ≤ 2KB 且含头尾与路径;盘上文件可被 tail 回读 |
| **2. 白名单注册表** | CriticalContext / FileLedger / render_critical_context / 三个 *_input 注入 | graph/state.py、core/state.py、tools/todo_tool.py、tools/file_tools.py、graph/memory.py | 断言 planner/verifier/code_agent 的输入包含 Git(或 not-a-repo)、Files、TODOs 块;todos 变更后 1 次节点内同步 |
| **3. 滑动窗口压缩器** | message_groups + 显式 id + plan_window + 逐条 RemoveMessage + 增量 rollup + 滚动合并 + monitor 增量计数 + 软阈值事件 | graph/nodes.py(压缩/monitor 重写)、graph/state.py、prompts/stage4.py、graph/memory.py、agents/*.py(赋 id) | 强制压缩后:保留组数/轮数符合配置;无孤儿 ToolMessage;`REMOVE_ALL_MESSAGES` 全仓 grep 为 0;连续 3 次压缩后白名单信息逐字存活(黄金用例) |
| **4. agent_status_bar** | 新节点 + 拓扑改造 + 两级注入 + TTL 缓存 | graph/status_bar.py(新)、graph/workflow.py、graph/nodes.py | 每次进入 verifier/planner 前 env_status 时效 < TTL;非 git 工作区全程无异常;状态块 ≤ 500 token |
| **5. 记忆与 prompt 清理** | HISTORY_SUMMARY 两段式、MAX_TEXT_CHARS 排除白名单、rollup prompt、P2 小项(重复 build_layered_memory 等) | graph/memory.py、prompts/stage4.py、graph/nodes.py | HISTORY_SUMMARY.md 前后两段;连续压缩 3 次后叙事段无重复膨胀(长度单调不增) |
| **6. 配置与文档** | 环境变量、默认值下调、README/workspace-lifecycle.md 说明 | nodes.py 常量、README.md、workspace-lifecycle.md | 所有新配置有默认值且文档化;旧行为关闭开关(`NEXUS_CONTEXT_STRATEGY=legacy`?)——**不建议保留 legacy 分支,直接切换,见决策点 5** |

依赖关系:Phase 2 不依赖 1;Phase 3 依赖 2(逐出时需要白名单兜底);Phase 4 依赖 2(git 板块);1 与 2 可并行。

---

## 6. 测试计划

- **单元测试**(tests/ 下新增,沿用现有命名风格):
  - `plan_window`:多组、单组超大、恰好边界;组内配对完整性;防御性保留逻辑;
  - `render_critical_context`:各板块渲染、数量上限、不截断不变式;
  - `spill_tool_output`:阈值两侧、头尾行数、主文本字段探测、路径相对化;
  - FileLedger LRU、`record_touch` 去重合并;
  - status_bar:非 git 仓库降级、TTL 缓存命中(mock 子进程);
  - 滚动合并:叙事段折叠只作用于最旧条目、白名单段逐字不变。
- **集成测试**(fake model,可控输出):
  - 注入超长转录强制触发压缩,断言:压缩后 `history_summary` 含白名单渲染块;`message_groups` 与 `messages` 一致(无悬空 id);
  - **黄金用例**:含特定文件路径 / 分支名 / in_progress todo 的状态,经过 3 次"增长→压缩"循环后,三项信息均可从 `history_summary` 与 planner 输入中恢复;
  - LLM rollup 抛异常 → fallback 路径产出确定性骨架,无信息为空。
- **兼容性**:旧 checkpoint(缺新字段)载入 → 默认值填充,不抛错。
- **性能**:monitor 不再每次 tokenize(增量计数);status_bar 子进程调用次数在 TTL 内 ≤ 1(计数断言)。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `add_messages` 对无 id 消息的自动赋 id 时机与 `RemoveMessage` 不匹配 | 所有 produced_messages 在节点内**显式赋 id**;单测覆盖"逐出→再合并"循环 |
| 窗口逐出产生孤儿 ToolMessage(未来某节点真回放历史时炸 API) | 组边界逐出 + 逐出前防御扫描(4.1.3);单测构造跨组 tool_call_id 的对抗用例 |
| 增量 rollup 仍失败(模型不可用) | 确定性 fallback(4.1.4);白名单渲染块独立于叙事段,永远存在 |
| status_bar 拖慢节点转移(git 子进程) | TTL 缓存 + 非 git 降级 + 仅在节点转移与超时的循环轮次刷新 |
| 白名单块本身膨胀(50 文件 + 100 artifacts) | 数量上限 LRU;artifacts 只留一行摘要;预算断言 ≤ 500 token(状态块)/白名单整体 ≤ 1500 token |
| 旧会话 / checkpoint 兼容 | schema 只增不改;新字段全默认值;HISTORY_SUMMARY.md 读侧兼容旧格式(无两段式结构时整文件视为叙事) |
| 压缩语义变化引发的调参困惑 | Phase 6 文档明确新口径;`compression_events` 扩展字段(kept/evicted/whitelist_digest)便于观测 |

---

## 8. 待确认决策点

1. **窗口参数默认值**:`KEEP_ROUNDS=6`、`MAX_WINDOW_RATIO=0.30`、`NEXUS_CONTEXT_TOKEN_LIMIT` 默认 400000→200000 —— 是否认可?或先保持 400000 观测一轮?
2. **"轮"的定义**:本计划按"节点执行组"计轮(planner 一组 + verifier 一组 = 一个 attempt 的两组)。若你希望按"用户会话轮"计(session 层),窗口机制应放在 session.py 侧扩展而非 graph 层——我判断你的痛点在 graph 长链路,故选前者,请确认。
3. **循环级状态注入频率**:每轮刷新 vs 仅超 TTL 刷新(默认后者)。影响 codeAgent 长循环中的git 状态新鲜度。
4. **摘要消息是否保留在 messages 里**:本计划选择"不注入,叙事只放 history_summary"(理由见 4.1.3)。若你希望兼容现有"压缩后 messages[0] 是摘要"的行为(有测试依赖它),可加开关。
5. **是否保留 legacy 全清空分支**:我建议不保留(双路径维护成本高、且全清空正是问题根源),直接切换。
6. **PinFact 工具**(允许 planner 主动钉住任意事实到白名单):有价值但可后置为 Phase 7,不在本次范围。

---

## 附录 A:问题 → 机制映射

| 问题 | 修复机制 |
|---|---|
| P0-1 全清空无保底 | 机制一(窗口)+ 机制二(白名单不变式) |
| P0-2 三层截断 | 机制二(白名单不截断)+ 机制一(叙事滚动折叠替代重述截断) |
| P0-3 摘要迭代衰减 | 机制一 4.1.5(增量 rollup + 两段式 HISTORY_SUMMARY) |
| P0-4 Git/路径无跟踪 | 机制二(FileLedger)+ 机制三(git 快照) |
| P0-5 落盘指针丢失 / 仅 bash / head-only | 机制四(统一 spill + artifacts 白名单) |
| P1-1 触发脱节 | 机制一 4.1.6(增量计数 + 软阈值 + 口径文档化) |
| P1-2 压缩器爆炸 | 机制一 4.1.4(只喂被逐出片段) |
| P1-3 checkpoint 审计 | 机制一(逐条 RemoveMessage,历史线程可回放逐出前状态) |
| P1-4 悬崖式压缩 | 机制一 4.1.6(70% 预警事件) |
| P2 重复构建 / 重复 tokenize | Phase 5 / 4.1.6 顺带修复 |

## 附录 B:涉及的文件总览

```
新增:
  src/nexusagent/graph/status_bar.py
  src/nexusagent/tools/output_sink.py
  tests/(window / whitelist / spill / status_bar / golden-compression 等)

修改:
  src/nexusagent/graph/state.py        # +env_status +critical_context +message_groups +MessageGroup/FileEntry
  src/nexusagent/graph/nodes.py        # 压缩器/monitor 重写、三个 *_input 注入、choke point 接入
  src/nexusagent/graph/workflow.py     # +agent_status_bar 节点与边
  src/nexusagent/graph/memory.py       # 两段式 history、白名单投影、MAX_TEXT_CHARS 排除
  src/nexusagent/core/state.py         # RuntimeState.touched_files + record_touch
  src/nexusagent/agents/code_agent.py  # 消息赋 id、choke point、循环级状态注入
  src/nexusagent/agents/search_agent.py# 同上
  src/nexusagent/tools/bash_tool.py    # 头尾预览对齐
  src/nexusagent/tools/file_tools.py   # record_touch 钩子
  src/nexusagent/tools/todo_tool.py    # todos_digest 同步刷新
  src/nexusagent/prompts/stage4.py     # CONTEXT_ROLLUP_PROMPT
  README.md / workspace-lifecycle.md  # 配置与口径说明
```
