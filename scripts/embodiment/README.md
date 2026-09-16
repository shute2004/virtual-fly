# embodiment code map

このディレクトリにはFlyBody/MaleCNS統合用の実行scriptと互換層があります。

## 現行Flyppy v3経路

production学習で参照する主要ファイル:

- `train_flyppy_population_packed.py` — production CLI。packed body-process構成を選びprocess trainerへ接続する。
- `train_flyppy_population_process.py` — production shared-weight population trainer。
- `train_flyppy_population.py` — 同じ学習意味論を確認するreference trainer。
- `flyppy_body_worker.py` / `flyppy_packed_body_worker.py` — MuJoCo/FlyBody body worker。
- `population_neural_bridge_client.py` — Rust shared-weight neural runtimeへのbridge client。
- `flybody_runtime.py` — CNS motor decoderを持たない共通FlyBody/MuJoCo runtime。
- `flybody_v3_adapter.py` / `flybody_muscle_adapter.py` — v3個別MN由来muscle state → physical torque境界。
- `wing_muscle_periphery.py` / `whole_body_periphery.py` — individual MN spike → motor-unit/muscle state。
- `malecns_retina.py` — FlyBody eye → released MaleCNS R1-R6 current。
- `direct_ommatidia_sensor_bodyexclude.py` — production image-free ommatidial sensing。
- `flyppy_course.py` / `flyppy_world.py` — 物理course。
- `live_telemetry.py` / `live_body_viewer.py` / `live_viewer_server.py` — observer-only viewer。

固定評価は `scripts/analysis/evaluate_flyppy_fixed.py` を使う。

再利用するtrainingロジックは `src/virtual_fly/training/` に置く。現在、curriculum、scheduler、checkpoint/resume、fixed evaluation、experiment forkをpackage側へ分離している。

## Legacy / historical diagnostics

`train_flyppy_curriculum.py` と `evaluate_flyppy.py` はpopulation production化前の経路を再現・比較するために残している。新規production開発の入口にはしない。

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
