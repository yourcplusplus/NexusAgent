# Phase 2:关键信息白名单(critical_context)

## 核心问题

Phase 1 把产物落了盘,但压缩的账还没有算完:Git 状态、文件路径、TODO、落盘产物指针——这些要么从未被结构化跟踪,要么只存在于转录里,清空即不可逆。当时的压缩是整表清空加一句摘要,以上信息全押在摘要模型那一次发挥上。

## 机制

第一个要解决的是「登记」。所有关键信息里,文件足迹最琐碎:读过的、写过的、落盘的产物,性质不同,保留策略也不同。所以 `record_touch` 按 `op` 分流——落盘产物进 `artifacts`(上限 100),普通读写进 `touched_files`(上限 50),两个池独立淘汰。这样读几百个文件也不会把产物指针挤出去。

登记齐了,接下来是「渲染」。白名单块由 `render_critical_context` 从实时状态重建,六个板块(目标/约束/待办/文件/产物/分支),代码拼装、绝不截断,体积靠数量上限控制。它不经过 LLM,所以不存在被摘要吃掉的问题。写完放在四个 prompt(planner/verifier/codeAgent/searchAgent)的最前面——位置本身就是保险,真发生截断,先切的也是后面的叙事。

有个细节差点埋雷:产物的体积信息记了两组数,`source_*` 是原始输出的行数与字节,`json_*` 是落盘文件的。渲染优先用前者;只有半组时整组退回另一组。不然模型会看到「3000 行 / 189KB」这种混搭,把两个对象的数字读成一个的。

最后一个防呆:这块白名单不能在同一个 prompt 里出现两遍。分层记忆序列化给 prompt 时会剔除 critical_context 层;压缩器和 token 估算器不走这个序列化,直接拿 dict,该看的还是能看到。

## 涉及文件

改动铺得比较开:`core/state.py`(双池登记)、`graph/state.py`、`tools/file_tools.py` 与 `tools/output_sink.py`(登记接入)、`graph/memory.py`(渲染)、`graph/nodes.py` 加 `agents/code_agent.py`、`agents/search_agent.py`(注入)。测试在 `tests/test_critical_context.py`。

## 测试

187 个测试通过,新增 20 个。方法上有个讲究:patch 掉 `create_model` 换成假模型,捕获模型真实收到的 messages,再把序列化后的 JSON 载荷解析出来做断言——「块在最前且只出现一次」「压缩器仍保有这一层」。断言的是实际发出去的内容,不是内存里 dict 的长相。

## 自查发现的测试问题

两个都是断言自身的毛病,产品行为没错。一个空洞断言:写过 `"meta" not in touched_files.get("missing", {})`——查一个根本不存在的键,恒为真,零验证,跑之前自查换掉了。一个写错的断言:过滤时保留了 `  - ` 前缀,断言里忘了带上,首跑失败;改的是断言,不是代码。

## Phase 3 接管

用滑动窗口替换整表清空,被逐出的片段做增量 rollup。白名单从这里开始起保底作用——它由代码重建、不参与摘要,转录清空后六板块仍逐字存活。

## 遗留

`[Git]` 板块这一阶段还渲染 `(git: pending)` 占位,等 Phase 4 的 `agent_status_bar` 来填:git 快照、TTL 缓存、工作区 delta。
