# Flyppy v3 開発入口

Flyppy は最終目的ではない。現実のショウジョウバエの神経回路・身体・感覚系などから再構築した仮想ショウジョウバエを、コンピュータ上で飼育可能にする開発の初期インパクト検証として使う。

Flyppy 側の役割は、物理環境から感覚刺激を作り、行動結果を報酬系・嫌悪系への神経刺激として返すことに限定する。外部の学習アルゴリズムや行動方策を導入しない。

## 現行 production 経路

```text
Flyppy physical world
  -> FlyBody eye sensors
  -> MaleCNS R1-R6 current
  -> whole MaleCNS neural dynamics + local plasticity
  -> identified motor-neuron spikes
  -> peripheral muscle state
  -> FlyBody physical torque
  -> FlyBody / MuJoCo
```

現在の標準条件:

- environment: Flyppy v3
- population: 4
- shared weight: yes
- weight averaging: no
- curriculum: boundary-band
- launch mode: `async`
- production vision: direct-ray, 13 rays/ommatidium
- body runtime: packed/process-isolated
- live viewer: observer-only, demand-driven telemetry

## 日常の入口

### preflight

```bash
bash scripts/dev/preflight_flyppy_v3.sh
```

### 学習

```bash
bash scripts/dev/train_flyppy_population.sh
```

主な環境変数:

```text
VF_EXPERIMENT_DIR
VF_FLYPPY_EPISODES
VF_FLYPPY_POPULATION
VF_FLYPPY_LAUNCH_MODE=async|wave
VF_FLYPPY_CURRICULUM_MODE=boundary-band|adaptive
VF_FLYPPY_VISION_MODE=direct-ray|raster
```

### viewer

```bash
bash scripts/dev/view_flyppy_v3.sh
```

別 experiment を見る場合:

```bash
VF_EXPERIMENT_DIR=artifacts/experiments/<name> bash scripts/dev/view_flyppy_v3.sh
```

### 固定評価

```bash
bash scripts/dev/evaluate_flyppy_v3_fixed.sh
```

### scheduler A/B

```bash
bash scripts/dev/run_flyppy_scheduler_ab.sh
```

同じpersist済み checkpoint から `async` と `wave` をforkし、各24 episode、固定評価、比較レポートまで実行する。

## 2026-09-16 scheduler A/B の確定結果

episode255 / global weight version184を共通親として24 episodeずつ比較した。

- training first gate: async 12/24, wave 12/24
- training second gate: async 5/24, wave 5/24
- frozen fixed evaluation:両者同一
  - frontier first 2/4, second 0/4
  - midpoint first 2/4, second 0/4
  - target first 1/4, second 0/4
- boundary attempt round の source-version spread
  - async: mean 9.5, max 13, zero-spread 1/6
  - wave: mean 0, max 0, zero-spread 6/6

したがって completion-order / source-generation bias は実在するが、それを除去しただけでは現在の固定能力停滞は解消しない。

確定レポートは `reports/flyppy/experiments/scheduler-ab/episode255-v184/comparison.md` に保存する。

## 履歴

2026-09-15 の設計レビュー、Part引継ぎ、viewer調査、非同期population設計の詳細は `docs/archive/2026-09-15/` を参照する。
