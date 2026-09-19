# 文档导航

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

对外文档以英文为默认版本，主要入口同时提供日文和简体中文版本。更详细的开发记录和研究笔记中仍保留一些最初用日文编写的材料，它们用于保存当时的判断过程和研究来源，不应与当前对外发布的结论混为一谈。

## 建议先阅读

- [`../README.zh-CN.md`](../README.zh-CN.md) — 项目概览、基准结果、安装与复现方法
- [`results.zh-CN.md`](results.zh-CN.md) — 基准结果与历史开发结果的区分，以及公开表述范围
- [`architecture.zh-CN.md`](architecture.zh-CN.md) — 当前实际实现的系统结构
- [`code-structure.md`](code-structure.md) — 当前各模块的职责划分（英文）
- [`../canonical/canonical-v1/README.md`](../canonical/canonical-v1/README.md) — canonical v1 实验包
- [`release-and-zenodo.md`](release-and-zenodo.md) — GitHub 发布版本与 Zenodo 的发布流程（英文）

## 当前有效的科学与开发文档

以下文档说明目前仍然有效的实现原则和科学约束，其中部分只有日文版本。

- `scientific-model.md` — 神经系统与可塑性模型的原则，以及可以主张到什么程度
- `requirements.md` — 项目要求与禁止采用的捷径
- `embodiment-v0.md` — 身体、环境与神经系统之间的闭环连接
- `wing-motor-boundary.md` — 运动神经元到身体之间的边界
- `live-observers.md` — 保证可视化不干扰学习过程的规定
- `data-and-reproducibility.md` — 数据来源与大型生成物的管理原则
- `experiments.md` — 实验设计以及判断“发生学习”的标准
- `references.md` — 一手资料与外部资源
- `flyppy/README.md` — 当前 Flyppy 实现与开发细节

如果不同文档对当前代码结构的描述存在差异，请以 `architecture.zh-CN.md` 和 `code-structure.md` 为准。

## 复现性与来源记录

- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — 对历史来源和运行定义所做的审计与修复
- [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json) — canonical v1 的完整来源记录
- [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md) — canonical v1 的结果摘要
- [`data-and-reproducibility.md`](data-and-reproducibility.md) — 通用数据管理原则

## 历史资料

按日期保存的评审记录、交接资料、旧设计和一次性问题调查位于 `archive/YYYY-MM-DD/` 或 `reports/` 中。

这些资料被有意保留，用于追踪研究和开发过程。除非当前文档明确说明，否则不要把它们理解成在 canonical v1 相同条件和相同运行定义下得到的结果。

## 后续计划

`roadmap.md` 记录的是后续开发计划，并不意味着其中列出的全部生物学机制现在都已经实现。对外发布时的结论应优先依据 README、`results.zh-CN.md`、canonical v1 实验包以及当前代码。
