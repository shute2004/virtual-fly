# embodiment code map

このディレクトリにはFlyBody/MaleCNS統合用の実行scriptと互換層があります。

## 現行Flyppy経路

学習・評価で参照してよい主要ファイル:

- `train_flyppy_curriculum.py` — persistent Flyppy trainer。
- `flybody_runtime.py` — CNS motor decoderを持たない共通FlyBody/MuJoCo runtime。
- `flybody_muscle_adapter.py` — 現行の個別wing MN由来muscle state → physical torque境界。
- `wing_muscle_periphery.py` — individual MN spike → motor-unit/muscle state。
- `malecns_retina.py` — FlyBody eye → released MaleCNS R1-R6 current。
- `flyppy_course.py` / `flyppy_world.py` — 物理course。
- `evaluate_flyppy.py` — plasticity/reinforcementを止めた現行経路の評価。
- `live_telemetry.py` / `live_body_viewer.py` / `live_viewer_server.py` — 読み取り専用viewer。

カリキュラムpolicyの再利用ロジックは `src/virtual_fly/training/curriculum.py` にあります。

## Legacy / historical diagnostics

`flybody_adapter.py` の `FlyBodyWingAdapter` / `WingDrive` は旧 **DNg02 population activity → wing amplitude** 試作用です。現行学習・評価では使用しません。現行 `FlyBodyMuscleAdapter` はこのclassを継承していません。

DNg02前提の古いflight smoke/calibration scriptは歴史的診断として残っているものがあります。新しいFlyppy学習コードを書く際、それらをmotor pathの設計根拠として再利用しないでください。

以下の旧世代実装は現行トップ階層から削除済みで、必要ならGit履歴から参照できます。

- old `flyppy_closed_loop.py`
- old external `visual_motion_encoder.py`
- old in-process `training_visualizer.py`
- old runner-only `synapse_monitor.py`

## 不変条件

- 外部ANN/policy/action decoderを追加しない。
- DNg02 population averageへ現行motor pathを戻さない。
- reward/aversiveはDANへの神経刺激として扱う。
- viewerは学習状態を変更しない。
- body calibrationとCNS学習結果を混同しない。
