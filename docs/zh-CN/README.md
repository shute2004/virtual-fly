# 文档 — 简体中文

[English](../en/README.md) · [日本語](../ja/README.md) · [简体中文](README.md)

本目录保存用于向第三方说明研究内容的简体中文公开文档。英文版和日文版使用相同文件名和相同文档结构。

## 建议先阅读

- [`architecture.md`](architecture.md) — 当前实际实现的架构与科学边界
- [`results.md`](results.md) — canonical v1 结果，以及与历史开发结果之间的界限
- [`scientific-model.md`](scientific-model.md) — 神经系统、可塑性、感觉和运动输出的模型方针
- [`requirements.md`](requirements.md) — 项目需求、非目标和科学不变量

## 可重现性与来源

- [`data-and-reproducibility.md`](data-and-reproducibility.md) — 数据分层、来源类别、大型成果物策略和可重现性等级
- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — 研究开源发布前的可重现性与规则契约审计、修复
- [`code-structure.md`](code-structure.md) — 当前模块职责以及 Python / Rust 边界

## 实验与生物学边界

- [`experiments.md`](experiments.md) — 分阶段实验、对照条件、行为/神经指标和学习判定标准
- [`wing-motor-boundary.md`](wing-motor-boundary.md) — 单个翼运动神经元到外周肌肉的边界
- [`references.md`](references.md) — 一次文献、官方数据和主要外部软件

## canonical 实验

对外发布的基准实验作为自包含的可重现性包放在 `docs/` 之外：

- [`../../canonical/canonical-v1/README.md`](../../canonical/canonical-v1/README.md)
- [`../../canonical/canonical-v1/reference-report.md`](../../canonical/canonical-v1/reference-report.md)
- [`../../canonical/canonical-v1/reference-manifest.json`](../../canonical/canonical-v1/reference-manifest.json)

## 非公开文档类别

开发内部文档放在 [`../internal/`](../internal/README.md)，历史设计、调查和交接记录放在 [`../archive/`](../archive/README.md)。这些内容并不是翻译缺失，因此不会强制复制成三种语言。
