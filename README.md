# virtual-fly

成体オスのショウジョウバエ（*Drosophila melanogaster*）の中枢神経系コネクトームを初期状態として、PC上で神経活動・神経修飾・シナプス可塑性を時間発展させ、仮想身体と閉ループ接続する研究・実装プロジェクトです。

最終目標は、仮想ショウジョウバエを Flappy Bird 型の環境に置き、感覚入力を神経刺激へ、行動結果を報酬系・嫌悪系への神経刺激へ変換し、外部の学習アルゴリズムを使わずに神経系自身の可塑性だけで行動が変化するかを検証することです。

## 重要な設計原則

- コネクトームは固定ネットワークではなく、`t = 0` の初期状態として扱う。
- シナプスは固定しない。強化・弱化・形成・消失を扱える設計にする。
- 既存のニューラルネットワーク、Transformer、誤差逆伝播、勾配降下、Q学習などを学習機構として導入しない。
- 外部から与えるのは、可能な限り生物が受け取る形に対応した神経刺激と神経修飾刺激に限定する。
- 行動は外部プログラムが決定せず、神経活動から運動系を経由して仮想身体に生じさせる。
- 実測情報、文献から採用したモデル、便宜的な仮定を明確に区別する。
- 生物学的忠実度は段階的に上げる。最初から「完全再現」を前提にしない。

## 現在の実装

現在の開発ブランチ `feat/bootstrap-neural-runtime` では、MaleCNS v1.0の166,700ニューロンと約2,558万のニューロン間接続を読み込み、CPU並列またはGPU computeで時間発展させる神経ランタイムを実装しています。

加えて、最初の実データ学習実験として嗅覚連合学習を実装しています。

- 手掛かりA: `DA1_lPN`
- 手掛かりB: `DL3_lPN`
- 報酬側DAN候補: `PAM08`
- 嫌悪側DAN候補: `PPL1` 系
- 読み出し候補: `MBON` 群

これらはMaleCNSの公開注釈からbody IDを抽出して使用します。外部プログラムが個々のシナプス重みを指定することはなく、cueとDAN活動の時間的組合せに応じて局所可塑性則がシナプス状態を更新します。学習後の重みは `learned_weights.f32le` として保存します。

この嗅覚条件付けは、**実MaleCNSグラフ上で可塑性が経験依存に動くことを検証するための最初の学習実験**です。Flyppyの身体・視覚閉ループは次の実装段階です。

## 実行

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
git switch feat/bootstrap-neural-runtime
bash scripts/dev/bootstrap.sh
```

MaleCNSの元データがすでに存在する場合は再ダウンロードしません。

主な生成物:

```text
artifacts/malecns-v1.0/
  manifest.json
  conditioning-v0.json

artifacts/experiments/conditioning-v0/
  learned_weights.f32le
  result.json
```

学習回数は環境変数で変更できます。

```bash
VF_TRAIN_CYCLES=100 bash scripts/dev/bootstrap.sh
```

## 最終的なFlyppy実験

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

成功時には報酬に関与する神経回路へ、失敗時には嫌悪学習に関与する回路へ刺激を与えます。外部プログラムは「どのシナプスをどう更新するか」を指定しません。

## 想定構成

- **神経系コア**: Rust。大規模疎グラフ、神経状態、可塑性、チェックポイントを担当する。
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

## ライセンス

現時点ではプロジェクト本体のライセンスを確定していません。外部データセット・モデル・ソフトウェアはそれぞれのライセンス・利用条件に従います。
