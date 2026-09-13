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
- 既知の生物学的・物理的な局所過程を、同じ結果を返す外部の特徴抽出・集約アルゴリズムで置換しない。
- 生物学的忠実度は段階的に上げる。最初から「完全再現」を前提にしない。

## 現在の実装

`feat/bootstrap-neural-runtime` では、MaleCNS v1.0 の166,700ニューロンと約2,558万のニューロン間接続を読み込み、CPU並列またはGPU computeで時間発展させる神経ランタイムを実装しています。最初の実データ学習として、`DA1_lPN` / `DL3_lPN` と `PAM08` / `PPL1` を用いた嗅覚連合学習を実装しています。

`feat/flybody-flyppy-loop` では、その神経ランタイムを FlyBody / FlyGym / MuJoCo と接続し、最初の閉ループ Flyppy 実験まで進めています。

```text
Flyppy physical world
      ↓
FlyBody raw eye cameras
      ↓
MaleCNS optic-lobe hex columns: local light sampling
      ↓
current into corresponding R1-R6 photoreceptor body IDs
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

視覚入力では、MaleCNS公式annotationの `assignedOlHex1` / `assignedOlHex2` をretinotopic座標として使い、対応する実際の `R1-R6` body IDへ局所受光量に応じた電流を流します。R1-R6自身にcolumn座標がない場合は、column座標を持つL1への実際のMaleCNS接続からR1-R6を逆引きします。外部でT4/T5運動応答、edge、障害物位置、gap位置などを計算してCNSへ与える処理はありません。

この境界で `observed` なのはMaleCNSのbody ID、接続、optic-lobe hex座標です。FlyBody眼カメラ上へhex latticeを投影する幾何変換と、局所受光量から外部電流へのscaleは現時点では `calibrated` な感覚変換境界です。空間情報を平均・poolingして意味情報へ変換する処理は行いません。

### FlyBody飛翔物理

FlyGym 2.1.0の実験的FlyBody統合では、元FlyBodyに含まれる左右wingのMuJoCo fluid geometryが変換時に省略されています。Flyppyではこれを復元し、元FlyBody飛翔タスクに合わせてwing gain、stiffness / damping、50 µs physics timestep、空気密度・粘性の単位系も補正しています。

Flyppyの各episodeはFlyBody公式飛翔条件を参考に47.5度のbody pitchから開始し、+X方向へ既定 `300 mm/s` の初速度を一度だけ与えます。その後の前進速度を外部から維持・補正する処理はありません。CNSから得たDNg02活動によるwing制御とMuJoCo物理だけで運動を継続します。

`bash scripts/dev/embodiment.sh` では、retinotopic R1-R6視覚入力、復元したflight geometry・parameterがcompile後のMuJoCoモデルへ実際に入っていることに加え、同一初速度でwing fluidあり/なしを比較するflight envelopeも検証します。

## 初回セットアップ

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
git switch feat/bootstrap-neural-runtime
bash scripts/dev/bootstrap.sh
```

MaleCNSの元データがすでに存在する場合は再ダウンロードしません。

## Flyppy閉ループ学習

閉ループ実装ブランチへ切り替え、まず身体・神経接続の検証を実行します。

```bash
git switch feat/flybody-flyppy-loop
git pull
bash scripts/dev/embodiment.sh
```

これが通った後に学習を実行します。

```bash
bash scripts/dev/train_flyppy.sh
```

デフォルトは表示なしです。GPU backend、retinotopic R1-R6入力、FlyBody物理、局所可塑性、報酬・嫌悪刺激を使って学習を進めます。

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

加えて、retinotopic入力の解決結果を以下へ保存します。

```text
artifacts/malecns-v1.0/retinotopic-vision-v1.json
```

checkpointはシナプス重みだけでなく、膜電位・spike・refractory・activity trace・neuromodulation・eligibilityまで含みます。既定では8 episodeごとと最終episodeに保存します。

長く回す例:

```bash
bash scripts/dev/train_flyppy.sh --episodes 100
```

飛翔開始速度を変更する場合:

```bash
bash scripts/dev/train_flyppy.sh --initial-forward-speed-mm-s 250
```

これはepisode開始時の初期条件だけを変更し、継続的な前進制御は追加しません。

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

localhost上のブラウザビューアはtrajectoryとsynapse snapshotを表示する開発用表示層です。表示用の集約値はCNSへの入力には使いません。

シナプスの3Dノード位置は現時点ではbody IDから決定論的に生成した模式配置であり、実際の解剖学的位置ではありません。MaleCNSの実形態・実シナプス座標を接続できた段階で表示層を置換します。

## 想定構成

- **神経系コア**: Rust。大規模疎グラフ、神経状態、可塑性、チェックポイントを担当する。
- **科学実験・統合層**: Python。データ前処理、MuJoCo / FlyGym / FlyBody との接続、実験設定、解析を担当する。
- **身体・物理**: FlyBody / FlyGym / MuJoCo。FlyGym 2.1.0で省略されたflight-only physicsは互換層で補う。
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
