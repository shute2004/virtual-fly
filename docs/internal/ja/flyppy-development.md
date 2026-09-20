# Flyppy v3 開発入口

Flyppyそのものが最終目的ではない。現実のショウジョウバエの神経回路・身体・感覚系などから再構築した仮想ショウジョウバエを、計算機上で飼育できるところまで進めるための初期検証課題として使う。

Flyppy側の役割は、物理環境から感覚刺激を作り、行動の結果を報酬系・嫌悪系への神経刺激へ変換することに限る。外部の学習アルゴリズムや行動方策は導入しない。

## 現在の実行経路

```text
Flyppyの物理環境
  -> FlyBodyの眼センサー
  -> MaleCNS R1-R6への電流
  -> MaleCNS全体の神経活動 + 局所可塑性
  -> 同定済み運動ニューロンの発火
  -> 末梢筋状態
  -> FlyBodyへ加える物理トルク
  -> FlyBody / MuJoCo
```

現在の標準条件:

- 環境: Flyppy v3
- 同時個体数: 4
- 共有重み: あり
- 重み平均: なし
- 経験条件調整: `boundary-band`
- 起動方式: `async`
- 視覚: `direct-ray`、個眼あたり13本
- 身体実行方式: 複数個体をまとめて起動しつつ、各個体のMuJoCo状態は分離
- ライブ表示: 観察専用、接続中だけ表示用状態を生成

## 日常的に使う入口

### 事前確認

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

### 可視化

```bash
bash scripts/dev/view_flyppy_v3.sh
```

別の実験を見る場合:

```bash
VF_EXPERIMENT_DIR=artifacts/experiments/<name> bash scripts/dev/view_flyppy_v3.sh
```

### 固定評価

```bash
bash scripts/dev/evaluate_flyppy_v3_fixed.sh
```

### 実行順序のA/B比較

```bash
bash scripts/dev/run_flyppy_scheduler_ab.sh
```

同じ保存済みチェックポイントから`async`と`wave`へ分岐し、それぞれ24エピソードの学習、固定評価、比較レポート生成までを実行する。

## 2026年9月16日の実行順序A/B比較

エピソード255 / 共有重み版184を共通の親状態として、それぞれ24エピソードずつ比較した。

- 学習中の第1ゲート通過: `async` 12/24、`wave` 12/24
- 学習中の第2ゲート通過: `async` 5/24、`wave` 5/24
- 固定評価: 両者同一
  - 到達境界条件: 第1ゲート 2/4、第2ゲート 0/4
  - 中間条件: 第1ゲート 2/4、第2ゲート 0/4
  - 目標条件: 第1ゲート 1/4、第2ゲート 0/4
- 各試行まとまり内で、開始時の共有重み版がどれだけばらついたか
  - `async`: 平均9.5、最大13、ばらつき0だったまとまり1/6
  - `wave`: 平均0、最大0、ばらつき0だったまとまり6/6

したがって、終了順序や開始時の重み世代による偏りは実際に存在する。ただし、それを取り除いただけでは、現在の固定評価で見られる能力停滞は解消しなかった。

確定レポートは`reports/flyppy/experiments/scheduler-ab/episode255-v184/comparison.md`に保存する。

## 過去資料

2026年9月15日の設計レビュー、作業引き継ぎ、可視化調査、非同期並列学習の設計詳細は`docs/archive/2026-09-15/`を参照する。
