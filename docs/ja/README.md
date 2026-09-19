# ドキュメント — 日本語

[English](../en/README.md) · [日本語](README.md) · [简体中文](../zh-CN/README.md)

このディレクトリには、研究内容を第三者へ説明するための日本語公開文書を置きます。英語版・簡体字中国語版も同じファイル名・同じ文書構成で揃えます。

## まず読むもの

- [`architecture.md`](architecture.md) — 現在実装されている構成と科学的な境界
- [`results.md`](results.md) — canonical v1の結果と、過去の開発結果との区別
- [`scientific-model.md`](scientific-model.md) — 神経系、可塑性、感覚、運動出力のモデル方針
- [`requirements.md`](requirements.md) — 要件、非目的、科学的な不変条件

## 再現性と来歴

- [`data-and-reproducibility.md`](data-and-reproducibility.md) — データ層、来歴区分、大容量成果物の扱い、再現性の段階
- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — 研究OSS公開前の再現性・仕様契約監査と修正
- [`code-structure.md`](code-structure.md) — 現在のモジュール責務とPython / Rustの境界

## 実験と生物学的な境界

- [`experiments.md`](experiments.md) — 段階的実験、対照条件、行動・神経指標、学習判定基準
- [`wing-motor-boundary.md`](wing-motor-boundary.md) — 個別翼運動ニューロンから末梢筋までの境界
- [`references.md`](references.md) — 科学的参考文献と、一次論文を実装判断へ対応付けた根拠表

## canonical実験

公開時の基準実験は、自己完結した再現性パッケージとして`docs/`の外に置いています。

- [`../../canonical/canonical-v1/README.md`](../../canonical/canonical-v1/README.md)
- [`../../canonical/canonical-v1/reference-report.md`](../../canonical/canonical-v1/reference-report.md)
- [`../../canonical/canonical-v1/reference-manifest.json`](../../canonical/canonical-v1/reference-manifest.json)

## 公開文書以外

開発内部向けの文書は[`../internal/`](../internal/README.md)、過去の設計・調査記録は[`../archive/`](../archive/README.md)に分けています。これらは意図的に3言語へ複製していません。
