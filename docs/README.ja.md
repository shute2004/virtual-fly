# ドキュメント案内

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

公開向けドキュメントは英語をdefaultとし、主要な公開入口には日本語・簡体字中国語版も用意します。より深いdeveloper/scientific noteには、内部研究記録として日本語で作成されたものが残っています。それらを公開claimと混同せず、役割で分けます。

## 最初に読むもの

- [`../README.ja.md`](../README.ja.md) — project概要、canonical結果、installation、reproduction
- [`results.ja.md`](results.ja.md) — canonical / historical resultの境界とclaimルール
- [`architecture.ja.md`](architecture.ja.md) — 現在実装されているarchitecture
- [`code-structure.md`](code-structure.md) — 現行module ownership（英語）
- [`../canonical/canonical-v1/README.md`](../canonical/canonical-v1/README.md) — canonical v1 package
- [`release-and-zenodo.md`](release-and-zenodo.md) — GitHub Release / Zenodo公開手順（英語）

## 現行scientific / developer contract

以下は現在有効な契約・実装方針を説明します。一部は日本語のみです。

- `scientific-model.md` — 神経・可塑性model方針とclaim境界
- `requirements.md` — project要件と禁止するshortcut
- `embodiment-v0.md` — embodiment / closed-loop integration
- `wing-motor-boundary.md` — motor neuron→body boundary
- `live-observers.md` — viewer非干渉contract
- `data-and-reproducibility.md` — data provenance / large artifact policy
- `experiments.md` — experiment design / 学習判定基準
- `references.md` — 一次資料 / upstream asset
- `flyppy/README.md` — 現行Flyppy development/runtime note

実装構造について文書間で差がある場合、現在のproduction ownershipは`architecture.ja.md` + `code-structure.md`を優先してください。

## Reproducibility / provenance

- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — historical provenance / semantics監査と修正
- [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json) — canonical provenance
- [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md) — canonical結果の短い要約
- [`data-and-reproducibility.md`](data-and-reproducibility.md) — 一般data policy

## Historical material

日付付きreview、handoff、旧design、one-off investigationは`archive/YYYY-MM-DD/`や`reports/`に残します。

これらはprovenanceのため意図的に保持されています。現行文書が明示しない限り、current canonical semanticsで生成された結果として解釈してはいけません。

## Roadmap

`roadmap.md`はdevelopment planning documentであり、記載された全biological mechanismが既に実装済みであることを意味しません。公開claimではREADME、`results.ja.md`、canonical package、現行codeを優先してください。
