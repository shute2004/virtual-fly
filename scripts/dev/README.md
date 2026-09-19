# scripts/dev

開発・検証用のlauncherを置く。現行production入口と履歴用launcherを混在させない。

## Flyppy v3 現行入口

- `preflight_flyppy_v3.sh` — FlyBody/Flyppy v3物理preflight
- `train_flyppy_v3_population.sh` — production shared-weight population学習
- `view_flyppy_v3.sh` — observer-only live viewer
- `evaluate_flyppy_v3_fixed.sh` — 重みを固定した条件評価（課題イベント由来のDAN刺激は有効）
- `run_flyppy_scheduler_ab.sh` — async / wave A/B
- `fork_flyppy_experiment.py` — persist済み状態からexperiment fork
- `prepare_flyppy_population_inputs.py` — population入力準備

## 診断・性能測定

- `benchmark_flyppy_v3_population.sh`
- `smoke_flyppy_v3_population.sh`
- `profile_flyppy_v3_p0.sh`
- `profile_flyppy_v3_p0_ab.sh`
- `diagnose_flybody_reference_lift.sh`

## 共通helper

- `lib/runtime.sh` — `uv` / `.venv` と `cargo` / `~/.cargo/bin/cargo` のruntime探索

## legacy

`legacy/` は過去のFlyppy v1/v2、旧single-v3、gate2部分練習の再現用launcherを保存する。新規開発では使用しない。
