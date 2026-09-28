# Phase 4:agent_status_bar(环境状态栏)

## 核心问题

`[Git]` 板块从 Phase 2 起一直渲染 `(git: pending)` 占位:模型不知道自己在哪条分支上,不知道工作区刚发生了什么,也不知道上下文预算还剩多少。这些事实只散落在转录的工具输出里,一压缩连痕迹都没了。

## 机制

做法是加一个图级节点。monitor 和 compressor 两条路先在 `agent_status_bar` 汇合,再由它按 `context_next_node` 决定去 planner、verifier 还是 final——旧的 `context_compressor_route` 并进来后删掉了。这样每次进 LLM 节点前恰好刷一次快照。有个拓扑盲点要单独补:首个 planner 在 START 之后直接运行,还没路过状态栏,所以 planner/verifier/codeAgent 在节点入口各自补采一次——TTL 内是空操作,不会重复起子进程。

git 段只报仓库和分支,不报 dirty。这是个刻意决策:默认 workspace 在 `.nexusagent/workspaces/` 下且被 .gitignore 忽略,`git status` 会向上找到项目仓库,它的 dirty 统计里装的是开发者自己的未提交改动,渲染给模型,它会把别人的改动当成自己的产出。附带用 `git check-ignore` 探测 workspace 是否被跟踪,被忽略时明确标注 `workspace not tracked by this repo`。

成本闸门是 TTL 缓存,默认 5 秒(`NEXUS_STATUS_BAR_TTL_SECONDS` 可配)。git 子进程和工作区扫描共用一道闸:节点转移很勤,但 TTL 内只真正采集一次。缓存挂在 `RuntimeState.status_cache` 上——序列化 checkpoint 时整体跳过 runtime,不会污染载荷。

工作区 delta 没有复用 checkpoint 的 `workspace_manifest`:那个实现先 `rglob` 枚举全树再截断,放在每个节点转移的路径上太贵。自写了一个剪枝遍历,遍历时剪掉 `.venv`、`node_modules`、`.nexusagent` 整棵子树,用(字节数, mtime_ns)对比上次清单,给出增删改和最多 5 条样本。

注入分两级。图级是白名单 `[Git]` 板块加 prompt 里的 Environment status 小节;循环级在 planner/verifier/codeAgent 的工具循环内,TTL 过期才追加一条 `[STATUS ...]`。两者刻意分工:白名单承载 repo/branch/delta(要跨压缩存活),小节只放白名单没有的预算、产物计数、后台任务——同一事实不在一个 prompt 里出现两次。

循环级注入有个隐蔽坑:状态消息不能放进节点的 produced_messages——verifier 会拿最后一条非工具消息去解析 JSON,拿到状态栏文本就误判失败。所以只进本地循环 messages,新鲜度由事件流留痕。计划原文写的是「该消息属于当前组」,按字面实现会踩坑。

还有一处结构调整:env getter 抽成了 `graph/config.py`——status_bar 和 code_agent 都要读配置,留在 nodes.py 里会跟两者循环导入。

## 涉及文件

新增 `graph/status_bar.py`、`graph/config.py`、`tests/test_status_bar.py`;`core/state.py` 加 status_cache,`graph/state.py` 加 EnvStatus,`graph/memory.py` 的 `[Git]` 板块改从快照渲染,`graph/nodes.py` 加节点与路由、删 `context_compressor_route`,`graph/workflow.py` 改拓扑,`graph/context_window.py` 加两个估算函数,`agents/code_agent.py`、`cli/formatter.py`、`cli/event_summary.py`、`cli/tui/app.py` 各接一处,README 与 `.env.example` 补文档。测试侧同步更新 `test_graph.py`、`test_context_window.py`、`test_critical_context.py` 的既有断言。

## 测试

244 个测试通过,新增 32 个。覆盖:git 两态与四种降级(非仓库、无 git 可执行文件、游离 HEAD、被忽略的 workspace)、TTL 缓存命中与过期(带子进程调用计数断言)、delta 的基线/增删改/同尺寸改写/截断标记、状态块 ≤500 token 的渲染上界、拓扑边断言、环内注入的注入与不注入两态、CLI 与 TUI 事件分支,外加一个端到端整图跑通(monitor → compressor → status_bar → verifier → final,含一次真实逐出和 fallback)。

## 自查发现

五条,按踩到的顺序记。①getter 挪去 config.py 后忘了删 nodes.py 里的同名 def——import 和 def 同名不报错,静默取后者,7 个测试 NameError;影响面 grep 得同时覆盖「谁调用」和「旧定义还在不在」。②monkeypatch 的 patch 目标随实现搬走而失效,patch 目标也是契约。③同尺寸改写测试在 tmpfs 上失败,查下来是真边界不是测试瑕疵:tmpfs 的时间戳粒度让紧邻两次写入的 mtime_ns 完全相同,(size, mtime) 判定看不见变化——写进了 docstring,测试改成显式 os.utime 错开。④端到端测试抓出三处我自己写错的断言:断言了不存在的事件类型;以为清缓存等于 TTL 过期(清缓存连基线一起丢);以为 delta 该有变化(TTL 内只有一次采集,停在 baseline 才对)。⑤清单 400 条上限被触发时,计数只覆盖扫到的子集——加了 partial 标记,渲染层写明 (partial scan)。

## 已知边界

不报 dirty 是刻意的,模型真需要时再升级成带标注的全仓计数;delta 会漏检同长度、同时间戳刻度的改写;后台任务只报输出文件、不声称活跃(pid 没持久化,从盘上判断不了);TODO.md 和 HISTORY_SUMMARY.md 的写入会出现在 delta 里——它们确实变了,不做豁免。

## Phase 5 接管

原计划还有个 Phase 5(记忆与 prompt 清理:HISTORY_SUMMARY 两段式、`MAX_TEXT_CHARS` 排除白名单等),没有作为独立阶段执行;其中 rollup prompt 的逐字保留规则在评估集阶段落地了,其余未做。
