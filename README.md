# virtual-fly

成体オスのショウジョウバエ（*Drosophila melanogaster*）のMaleCNSコネクトームを初期状態として、神経活動・神経修飾・局所シナプス可塑性を時間発展させ、FlyBody / MuJoCoの身体と閉ループ接続するプロジェクトです。

目標は、外部のANN・Transformer・Q学習・policy gradient・backpropなどで行動を学習させるのではなく、**MaleCNS自身の局所可塑性によって仮想ハエの行動が変化するか**を実装として確かめることです。

## 現在の正規経路

現在のFlyppy学習経路は次です。

```text
Flyppy physical world
      ↓
FlyBody eye sensors
      ↓
released MaleCNS R1-R6 body-ID current
      ↓
whole MaleCNS neural dynamics + local plasticity
      ↓
individual released wing motor-neuron spikes
      ↓
WingMusclePeriphery
      ↓
individual motor-unit / muscle activation state
      ↓
FlyBodyMuscleAdapter physical wing torque
      ↓
FlyBody / MuJoCo
      ↓
Flyppy physical world
```

ゲート通過時は実際のreward DAN候補群へ電流を与え、衝突時はaversive DAN候補群へ電流を与えます。外部からscalar reward、Q値、target action、policy lossなどは与えません。

旧試作の **DNg02 population average → wing amplitude** 経路は現在の学習・評価には使いません。`scripts/embodiment/flybody_adapter.py` は旧診断用のlegacy adapterであり、現行 `FlyBodyMuscleAdapter` の親クラスではありません。共通のFlyBody/MuJoCo物理層は `scripts/embodiment/flybody_runtime.py` に分離されています。

## 設計原則

- MaleCNS snapshotは初期状態として扱い、学習中の可塑状態は別に保持する。
- 外部ANN、Transformer、policy、action decoderを神経系へ付加しない。
- 視覚入力はreleased R1-R6 body IDへ局所電流として与える。
- 運動境界は個別wing motor neuron body IDを維持し、population averageで行動へ変換しない。
- 報酬・嫌悪はDANへの神経刺激としてのみ与える。
- episode resetでは学習済みweightを残し、膜電位・spike・refractory・trace・modulation・eligibilityなど短期状態を消す。
- 未知の生物学を「飛ばすため」に推測で埋めない。
- 実測・文献・推定・仮定・calibrationを区別する。
- 3D viewerは観察専用で、学習プロセスへ状態を返さない。

詳細な開発規約は [`AGENTS.md`](AGENTS.md) を参照してください。

## 初回セットアップ

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync
bash scripts/dev/bootstrap.sh
```

MaleCNSデータや生成済みartifactが存在する場合は可能な範囲で再利用します。

## 通常のFlyppy v3学習

現行production launcherは、shared-weight population、個別MN→筋肉境界、direct-ray視覚を使うFlyppy v3経路です。

```bash
bash scripts/dev/train_flyppy_v3_population.sh
```

既定では次を使います。

- population: 4
- episodes: 24
- curriculum: `boundary-band`
- launch mode: `async`
- vision: `direct-ray`, 13 rays/ommatidium
- body runtime: packed/process-isolated

実験ディレクトリの既定値は次です。

```text
artifacts/experiments/flyppy-v3/
├── checkpoint/
├── curriculum-state.json
├── population-state.json
├── trajectory.jsonl
├── commit-log.jsonl
├── summary.json
└── live/
```

`artifacts/` は巨大・高頻度データ用でGitには含めません。小さい結果は `reports/flyppy/` へ出力します。

```text
reports/flyppy/latest.md
reports/flyppy/latest.csv
reports/flyppy/population_latest.md
reports/flyppy/history.csv
```

checkpointが存在する場合は自動で継続します。中断時はpersist済みglobal weight versionを確定点として、未保存commit/trajectoryをresume時に整合させます。

### curriculum / scheduler

`boundary-band` は24 attemptを固定したまま評価し、batch境界でのみ難易度を更新します。非同期slotの終了順がcurriculum自体を動かさないようにするためです。

production既定は `async` です。`wave` は同一launch roundを同一CNS weight versionから開始する比較・診断用schedulerで、既存experimentの途中で `async ↔ wave` を切り替えることはできません。比較するときは同じcheckpointから別experimentへforkします。

```bash
bash scripts/dev/run_flyppy_scheduler_ab.sh
```

現在の詳細な実行条件とA/B結果は [`docs/flyppy/README.md`](docs/flyppy/README.md) を参照してください。

## 3D学習viewer

viewerは学習とは別プロセスで、production trainingへ状態を返さないobserverです。telemetryはviewer接続時だけ生成します。

```bash
bash scripts/dev/view_flyppy_v3.sh
```

別experimentへ接続する場合:

```bash
VF_EXPERIMENT_DIR=artifacts/experiments/<name> bash scripts/dev/view_flyppy_v3.sh
```

- FlyBody: メイン表示。ドラッグで回転、ホイールでズーム。
- MaleCNS: neural activity / propagationをobserverとして表示。
- viewer接続・切断によって学習意味論は変えません。

## 固定評価

学習済みCNSを固定条件で比較する評価器も、productionと同じ個別MN→筋肉経路を使います。評価では各neural stepを `plasticity=false` にし、weightを変更しません。

```bash
bash scripts/dev/evaluate_flyppy_v3_fixed.sh
```

現行結果は `reports/flyppy/fixed_evaluation_latest.md`、過去checkpointの固定評価は `reports/flyppy/evaluations/history/` に保存します。

## Pythonコード構成

再利用する純粋ロジックは段階的に `src/virtual_fly/` packageへ移しています。現在、curriculum policyは次にあります。

```text
src/virtual_fly/training/curriculum.py
```

`scripts/` は最終的にCLI・データ生成・開発launcher中心へ薄くしていきます。現時点ではFlyGym/MuJoCo統合コードの一部がまだ `scripts/embodiment/` に残っています。

## Rust neural core

神経系コアはRustです。大規模疎グラフ、CPU/GPU time evolution、局所可塑性、checkpointを担当します。GPUは計算基盤として使用しますが、GPU上に別の学習policyを置くことはありません。

現行GPU synapse scaleはwhole-CNS stability calibrationから取得し、launcherがartifactから読み込みます。

## 主なドキュメント

- [`docs/README.md`](docs/README.md) — ドキュメント全体の案内
- [`docs/flyppy/README.md`](docs/flyppy/README.md) — 現行Flyppy v3の実行・検証状態
- [`docs/requirements.md`](docs/requirements.md) — 要件定義
- [`docs/architecture.md`](docs/architecture.md) — システム設計
- [`docs/scientific-model.md`](docs/scientific-model.md) — 神経系・可塑性モデル
- [`docs/experiments.md`](docs/experiments.md) — 実験設計
- [`docs/data-and-reproducibility.md`](docs/data-and-reproducibility.md) — データ来歴・再現性
- [`docs/roadmap.md`](docs/roadmap.md) — 開発ロードマップ
- [`docs/references.md`](docs/references.md) — 基礎資料・外部資産
- [`AGENTS.md`](AGENTS.md) — 開発エージェント向け規約

## ライセンス

現時点ではプロジェクト本体のライセンスを確定していません。外部データセット・モデル・ソフトウェアはそれぞれのライセンス・利用条件に従います。
