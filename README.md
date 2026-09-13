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

`feat/bootstrap-neural-runtime` では、MaleCNS v1.0 の166,700ニューロンと約2,558万のニューロン間接続を読み込み、CPU並列またはGPU computeで時間発展させる神経ランタイムを実装しています。最初の実データ学習として、`DA1_lPN` / `DL3_lPN` と `PAM08` / `PPL1` を用いた嗅覚連合学習を実装しています。

`feat/flybody-flyppy-loop` では、その神経ランタイムを FlyBody / FlyGym / MuJoCo と接続し、最初の閉ループ Flyppy 実験まで進めています。

```text
Flyppy physical world
      ↓
FlyGym compound-eye ommatidia
      ↓
T4/T5 vertical-motion input approximation
      ↓
MaleCNS runtime + local plasticity
      ↓
DNg02 bilateral readout
      ↓
FlyBody wing actuation
      ↓
Flyppy physical world
```

ゲート通過時は `PAM08` 候補群、衝突時は `PPL1` 候補群を刺激します。ゲームのゲート座標をCNSへ直接入力したり、外部プログラムから個々のシナプス重みを指定したりはしません。

現在の主な近似は、FlyGymの複眼各ommatidiumとMaleCNS各視覚ニューロンの個別対応がまだ確定していないため、縦方向の視覚運動をT4c/T4d・T5c/T5d集団へ与える人口レベルの暫定入力層を置いている点です。

## 初回セットアップ

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
git switch feat/bootstrap-neural-runtime
bash scripts/dev/bootstrap.sh
```

MaleCNSの元データがすでに存在する場合は再ダウンロードしません。

## Flyppy閉ループ学習

閉ループ実装ブランチへ切り替え、学習を実行します。

```bash
git switch feat/flybody-flyppy-loop
git pull
bash scripts/dev/train_flyppy.sh
```

デフォルトは表示なしです。GPU backend、複眼入力、FlyBody物理、局所可塑性、報酬・嫌悪刺激を使って学習を進めます。

主な生成物:

```text
artifacts/experiments/flyppy-v0/
├── trajectory.jsonl
├── summary.json
└── checkpoint/
    ├── manifest.json
    ├── membrane.f32le
    ├── spikes.u32le
    ├── refractory.u32le
    ├── activity-trace.f32le
    ├── modulation.f32le
    ├── weights.f32le
    └── eligibility.f32le
```

checkpointはシナプス重みだけでなく、膜電位・spike・refractory・activity trace・neuromodulation・eligibilityまで含みます。既定では8 episodeごとと最終episodeに保存します。

長く回す例:

```bash
bash scripts/dev/train_flyppy.sh --episodes 100
```

保存済みCNSからさらに続ける場合:

```bash
bash scripts/dev/train_flyppy.sh \
  --resume-checkpoint artifacts/experiments/flyppy-v0/checkpoint \
  --episodes 100
```

同じ実験ディレクトリへresumeするとtrajectoryは追記されます。身体・コースはepisode境界から再開し、CNS内部状態はcheckpointから継続します。

### 3Dで身体を見る

学習中のFlyBodyを3D表示したい場合だけ `--render` を付けます。

```bash
bash scripts/dev/train_flyppy.sh --render
```

動画として保存する場合:

```bash
bash scripts/dev/train_flyppy.sh --record-video artifacts/videos/flyppy
```

通常の学習速度を優先するときは、どちらも付けません。

### シナプス変化を記録する

神経可視化用に、エピソード単位でシナプス変化を抽出したい場合だけ `--synapse-trace` を付けます。全約2,558万重みを毎step読み戻すことはせず、低頻度で重みを取得して変化量の大きい接続だけ残します。

```bash
bash scripts/dev/train_flyppy.sh --synapse-trace --synapse-top-n 128
```

追加生成物:

```text
artifacts/experiments/flyppy-v0/synapse-snapshots.jsonl
```

### 神経活動・シナプス変化を3Dで見る

別ターミナルで以下を実行します。

```bash
bash scripts/dev/view_neural.sh
```

localhost上のブラウザビューアが開き、`trajectory.jsonl` と `synapse-snapshots.jsonl` を1秒ごとに再取得して学習中でも追従します。表示するのはT4/T5入力、左右DNg02発火率、PAM08/PPL1刺激、強化・弱化した上位シナプスです。

シナプスの3Dノード位置は現時点ではbody IDから決定論的に生成した模式配置であり、実際の解剖学的位置ではありません。MaleCNSの実形態・実シナプス座標を接続できた段階で表示層を置換します。

## 想定構成

- **神経系コア**: Rust。大規模疎グラフ、神経状態、可塑性、チェックポイントを担当する。
- **科学実験・統合層**: Python。データ前処理、MuJoCo / FlyGym / FlyBody との接続、実験設定、解析を担当する。
- **身体・物理**: FlyBody / FlyGym / MuJoCo。
- **可視化**: 通常はheadless。必要時だけ身体3D、動画、神経活動・シナプス変化の表示を有効化する。
- **将来のWeb実行**: Rust コアを WASM / WebGPU へ展開し、明示的に参加した閲覧者のPCで仮想ハエを動かして実験データを収集する。

## ドキュメント

- [`docs/requirements.md`](docs/requirements.md) — 要件定義
- [`docs/architecture.md`](docs/architecture.md) — システム設計
- [`docs/scientific-model.md`](docs/scientific-model.md) — 神経系・可塑性モデルの考え方
- [`docs/experiments.md`](docs/experiments.md) — 実験設計
- [`docs/data-and-reproducibility.md`](docs/data-and-reproducibility.md) — データ来歴・再現性
- [`docs/roadmap.md`](docs/roadmap.md) — 開発ロードマップ
- [`docs/references.md`](docs/references.md) — 基礎資料・外部資産
- [`docs/embodiment-v0.md`](docs/embodiment-v0.md) — 現在の身体・閉ループ・可視化・checkpoint実装
- [`AGENTS.md`](AGENTS.md) — 開発エージェント向けプロジェクト規約

## ライセンス

現時点ではプロジェクト本体のライセンスを確定していません。外部データセット・モデル・ソフトウェアはそれぞれのライセンス・利用条件に従います。
