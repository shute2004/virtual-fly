# 文档导航

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

对外公开文档以英文为 default，主要公开入口同时提供日文和简体中文版本。更深层的 developer/scientific note 中仍保留一些最初作为内部研究记录编写的日文文档。它们按照用途分类，不能被默认为对外 scientific claim。

## 建议先阅读

- [`../README.zh-CN.md`](../README.zh-CN.md) — 项目概览、canonical 结果、installation、reproduction
- [`results.zh-CN.md`](results.zh-CN.md) — canonical / historical result 边界与 claim 规则
- [`architecture.zh-CN.md`](architecture.zh-CN.md) — 当前实际实现的 architecture
- [`code-structure.md`](code-structure.md) — 当前 module ownership（英文）
- [`../canonical/canonical-v1/README.md`](../canonical/canonical-v1/README.md) — canonical v1 package
- [`release-and-zenodo.md`](release-and-zenodo.md) — GitHub Release / Zenodo 发布计划（英文）

## 当前 scientific / developer contract

以下文档定义或说明当前有效的项目规则，其中一部分仍只有日文版本。

- `scientific-model.md` — 神经/可塑性模型原则与 claim 边界
- `requirements.md` — 项目需求与禁止的 shortcut
- `embodiment-v0.md` — embodiment / closed-loop integration
- `wing-motor-boundary.md` — motor neuron→body boundary
- `live-observers.md` — viewer non-interference contract
- `data-and-reproducibility.md` — data provenance / large artifact policy
- `experiments.md` — experiment design / 学习判定标准
- `references.md` — 一次资料 / upstream asset
- `flyppy/README.md` — 当前 Flyppy development/runtime note

如果旧设计文字与现行代码 ownership 有差异，请优先参考 `architecture.zh-CN.md` + `code-structure.md`。

## Reproducibility / provenance

- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — historical provenance / semantics 审计与修复
- [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json) — canonical provenance
- [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md) — canonical 结果简报
- [`data-and-reproducibility.md`](data-and-reproducibility.md) — 通用 data policy

## Historical material

按日期保存的 review、handoff、旧设计与 one-off investigation 位于 `archive/YYYY-MM-DD/` 或 `reports/`。

它们为了 provenance 被刻意保留。除非当前文档明确说明，否则不能把它们解释为由 current canonical semantics 生成的结果。

## Roadmap

`roadmap.md` 是 development planning document，并不表示其中列出的所有 biological mechanism 已经实现。对外 claim 应优先依据 README、`results.zh-CN.md`、canonical package 与当前代码。
