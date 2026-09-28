# 文档索引

这套文档一共 7 份,外加 evals/ 的独立 README。想了解重构做了什么,按顺序读 phase-1 → phase-4;准备面试的话直接看 interview-cards.md;想知道评估怎么跑,去 evals/README.md——主 README 的「评估与验证」章节是整套工作最短的摘要。

逐份说明:

- [phase-1-output-sink.md](phase-1-output-sink.md) — 工具输出落盘。源头是"截断会砍掉 traceback"这个实测发现;超长输出写文件、Prompt 留头尾,单轮压到 ≤2KB。
- [phase-2-critical-context.md](phase-2-critical-context.md) — 关键信息白名单。六板块由代码重建、不经模型转述,压缩怎么都吃不掉它;产物体积的整组退让是差点埋雷的细节。
- [phase-3-sliding-window.md](phase-3-sliding-window.md) — 滑动窗口压缩。替换整表清空:逐条逐出、增量 rollup、叙事折叠,"摘要的摘要"式衰减从机制上消掉。
- [phase-4-status-bar.md](phase-4-status-bar.md) — 环境状态栏。git 与工作区快照加 TTL 缓存,`[Git]` 板块从占位变成事实;git 不报 dirty 是刻意决策,理由在文内。
- [interview-cards.md](interview-cards.md) — 8 张面试卡片,每张 150 字内,问题→方案→数据。
- [archive/context-compression-refactor-plan.md](archive/context-compression-refactor-plan.md) — 重构前的总计划,留着对照"当初打算做什么、实际做成了什么"。
- [archive/workspace-lifecycle.md](archive/workspace-lifecycle.md) — 早期的工作区生命周期教程,部分内容已过时,归档备查。

评估部分不在这份索引里展开:12 个任务怎么定义、dual-gate 怎么判、报告怎么读、已知局限有哪些,都在 [evals/README.md](../evals/README.md);主 [README](../README.md) 的「评估与验证」章节是压缩到两段的版本。
