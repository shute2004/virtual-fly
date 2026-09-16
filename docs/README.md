# virtual-fly ドキュメント案内

このディレクトリでは、現在有効な仕様書と履歴資料を分離する。

## 現行仕様

- `scientific-model.md` — 神経系・可塑性・検証方針のモデル契約
- `architecture.md` — システム全体のアーキテクチャ
- `requirements.md` — 実装要件
- `embodiment-v0.md` — 身体・環境接続
- `wing-motor-boundary.md` — wing motor neuron から身体側への境界
- `live-observers.md` — viewer / observer の非侵襲接続
- `data-and-reproducibility.md` — データと再現性
- `experiments.md` — 実験運用
- `references.md` — 出典
- `roadmap.md` — 開発ロードマップ
- `flyppy/README.md` — 現行 Flyppy v3 の開発・実行入口

## 履歴資料

日付付きレビュー、Part間の引継ぎ、原因調査、過去の設計案は `archive/YYYY-MM-DD/` に置く。
これらは当時の判断根拠を残すための資料であり、現在の実行仕様は上記の現行仕様と実コードを優先する。
