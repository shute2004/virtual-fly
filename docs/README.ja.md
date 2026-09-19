# ドキュメント案内

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

公開向けの文書は英語を既定とし、主要な入口には日本語版と簡体字中国語版も用意しています。より詳細な開発記録や研究メモには、日本語で作成された資料も残っています。これらは当時の判断や来歴を保存するためのもので、現在の公開上の主張とは分けて扱います。

## 最初に読む文書

- [`../README.ja.md`](../README.ja.md) — プロジェクト概要、基準結果、導入、再現方法
- [`results.ja.md`](results.ja.md) — 基準結果と過去結果の区別、公開時の主張範囲
- [`architecture.ja.md`](architecture.ja.md) — 現在実装されている構成
- [`code-structure.md`](code-structure.md) — 現在のモジュールごとの責務（英語）
- [`../canonical/canonical-v1/README.md`](../canonical/canonical-v1/README.md) — canonical v1の実験パッケージ
- [`release-and-zenodo.md`](release-and-zenodo.md) — GitHub ReleaseとZenodoへの公開手順（英語）

## 現在有効な科学・開発上の文書

次の文書は、現在の実装方針や科学上の約束事を説明します。一部は日本語のみです。

- `scientific-model.md` — 神経系・可塑性モデルの方針と、主張できる範囲
- `requirements.md` — プロジェクト要件と禁止している近道
- `embodiment-v0.md` — 身体・環境との閉ループ接続
- `wing-motor-boundary.md` — 運動ニューロンから身体までの境界
- `live-observers.md` — 可視化を学習から分離するための仕様
- `data-and-reproducibility.md` — データの来歴と大容量生成物の扱い
- `experiments.md` — 実験設計と、学習と判断するための基準
- `references.md` — 一次資料と外部資産
- `flyppy/README.md` — 現在のFlyppy実装・開発に関する詳細

文書間で実装構造の説明が食い違う場合は、現在の構成については`architecture.ja.md`と`code-structure.md`を優先してください。

## 再現性と来歴

- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — 過去の来歴・実行上の意味づけに関する監査と修正
- [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json) — canonical v1の完全な来歴記録
- [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md) — canonical v1の結果要約
- [`data-and-reproducibility.md`](data-and-reproducibility.md) — データ全般の扱い

## 過去資料

日付付きのレビュー、作業引き継ぎ、旧設計、単発の原因調査などは`archive/YYYY-MM-DD/`や`reports/`に残しています。

これらは研究・開発の来歴を保存するために意図的に保持しています。現在の文書で明示されていない限り、現在のcanonical v1と同じ条件・意味づけで得られた結果として解釈しないでください。

## 今後の計画

`roadmap.md`は今後の開発計画を記した文書です。そこに書かれている生物学的機構が、すべて現在すでに実装されているという意味ではありません。公開上の主張については、README、`results.ja.md`、canonical v1の実験パッケージ、現在のコードを優先してください。
