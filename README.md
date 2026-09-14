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

## 通常のFlyppy学習

標準launcherはpersistent runtimeです。1つのPythonプロセス、1つのMuJoCo simulation、1つのMaleCNS GPU runtimeを複数episodeで使い続けます。

```bash
bash scripts/dev/train_flyppy.sh \
  --episodes 24 \
  --trajectory-stride 10
```

現在の実験ディレクトリは次です。

```text
artifacts/experiments/flyppy-v1/
├── curriculum-state.json
├── trajectory.jsonl
├── summary.json
├── live/
└── checkpoint/
```

`artifacts/` は巨大・高頻度データ用でGitには含めません。GitHubで結果をレビューするため、学習終了時に自動で以下へ小さいレポートを書き出します。

```text
reports/flyppy/latest.md
reports/flyppy/latest.csv
```

checkpointが存在する場合は自動で継続します。通常の継続学習で `--fresh` は使わないでください。

## 現在の第2gate境界帯カリキュラム

固定条件テストで次の鋭い境界が確認されています。

```text
難しい端: x=10.500, z=5.2900, vx=350.00  -> 24/24失敗
易しい端: x=10.625, z=5.3525, vx=356.25  -> 24/24成功
```

次の学習では、この2点の間を5段階に分け、24episodeのbatch内で順序を決定論的にshuffleして提示します。

```text
ease level 0.00  難しい端  4回
ease level 0.25             5回
ease level 0.50             6回
ease level 0.75             5回
ease level 1.00  易しい端  4回
```

1episodeの成功/失敗ごとに条件を往復させず、24episode単位で成功率を見ます。

- 成功率80%以上: 境界帯全体を少し難しくする。
- 成功率40%以上80%未満: 同じ境界帯を継続する。
- 成功率40%未満: 以前に学習済みの易しい側へ少し戻す。

難化時は最終的な `x=9.0, z=5.0, vx=300` へ向かって帯域全体を移動します。

実行:

```bash
bash scripts/dev/train_flyppy_gate2_band.sh \
  --episodes 24 \
  --trajectory-stride 10
```

これは第1gateを省略して第2gateだけを部分練習する段階です。第2gateが安定した後に、全gateを戻した連続飛行へ移行します。

## 3D学習viewer

viewerは学習とは別プロセスです。

```bash
bash scripts/dev/view_learning.sh
```

FlyBody姿勢は学習側から毎control step配信し、observer側でMuJoCoのgeneralized positionとして補間して約30fpsで描画します。MaleCNS活動telemetryは低頻度のままなので、viewerのために大規模neural readbackを毎step行いません。

- FlyBody: メイン表示。ドラッグで回転、ホイールでズーム。
- MaleCNS: 右下の模式3D小窓。`B` キーまたはcheckboxで非表示可能。
- viewer操作・補間結果は学習へ返りません。

## 凍結評価

学習済みCNSとbaselineを比較する評価器も、学習と同じ個別MN→筋肉経路を使います。評価中はplasticityとreinforcementを無効にします。

```bash
uv run python scripts/embodiment/evaluate_flyppy.py \
  --learned-checkpoint artifacts/experiments/flyppy-v1/checkpoint \
  --episodes 4
```

baseline checkpointを指定しない場合は初期MaleCNS stateと比較します。

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
