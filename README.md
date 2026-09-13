# virtual-fly

成体オスのショウジョウバエ（*Drosophila melanogaster*）の中枢神経系コネクトームを初期状態として、PC上で神経活動・神経修飾・シナプス可塑性を時間発展させ、仮想身体と閉ループ接続する研究・実装プロジェクトです。

最初の目標は、仮想ショウジョウバエを Flappy Bird 型の環境に置き、感覚入力を神経刺激へ、行動結果を報酬系・嫌悪系への神経刺激へ変換し、外部の学習アルゴリズムを使わずに神経系自身の可塑性だけで行動が変化するかを検証することです。

## 重要な設計原則

- コネクトームは固定ネットワークではなく、`t = 0` の初期状態として扱う。
- シナプスは固定しない。強化・弱化・形成・消失を扱える設計にする。
- 既存のニューラルネットワーク、Transformer、誤差逆伝播、勾配降下、Q学習などを学習機構として導入しない。
- 外部から与えるのは、可能な限り生物が受け取る形に対応した神経刺激と神経修飾刺激に限定する。
- 行動は外部プログラムが決定せず、神経活動から運動系を経由して仮想身体に生じさせる。
- 実測情報、文献から採用したモデル、便宜的な仮定を明確に区別する。
- 生物学的忠実度は段階的に上げる。最初から「完全再現」を前提にしない。

## 最初の実験

`Flyppy`（仮称）という Flappy Bird 型環境を用います。

```text
3D environment
      ↓
compound-eye / sensory transduction
      ↓
adult male fly CNS
      ↓
neural activity + plasticity
      ↓
motor system
      ↓
virtual fly body
      ↓
3D environment
```

成功時には報酬に関与するドーパミン作動性回路へ、失敗時には嫌悪学習に関与する回路へ刺激を与えます。外部プログラムは「どのシナプスをどう更新するか」を指定しません。

## 想定構成

- **神経系コア**: Rust を中心に設計。大規模疎グラフ、神経状態、可塑性、チェックポイントを担当する。
- **科学実験・統合層**: Python。データ前処理、MuJoCo / FlyGym / FlyBody との接続、実験設定、解析を担当する。
- **身体・物理**: FlyBody / FlyGym / MuJoCo を第一候補とする。
- **将来のWeb実行**: Rust コアを WASM / WebGPU へ展開し、明示的に参加した閲覧者のPCで仮想ハエを動かして実験データを収集する。

## ドキュメント

- [`docs/requirements.md`](docs/requirements.md) — 要件定義
- [`docs/architecture.md`](docs/architecture.md) — システム設計
- [`docs/scientific-model.md`](docs/scientific-model.md) — 神経系・可塑性モデルの考え方
- [`docs/experiments.md`](docs/experiments.md) — 実験設計
- [`docs/data-and-reproducibility.md`](docs/data-and-reproducibility.md) — データ来歴・再現性
- [`docs/roadmap.md`](docs/roadmap.md) — 開発ロードマップ
- [`docs/references.md`](docs/references.md) — 基礎資料・外部資産
- [`AGENTS.md`](AGENTS.md) — 開発エージェント向けプロジェクト規約

## 現在地

現在は **設計・実現可能性検証段階** です。まずは、成体オス中枢神経系データの取得形式、利用条件、運動系・感覚系の対応情報、FlyBody/FlyGymとの接続点を確定し、最小の閉ループ実験を成立させます。

## ライセンス

現時点ではプロジェクト本体のライセンスを確定していません。外部データセット・モデル・ソフトウェアはそれぞれのライセンス・利用条件に従います。
