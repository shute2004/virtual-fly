# scripts/analysis

解析・benchmark・verify用CLIを置く。再利用する純粋ロジックは可能な限り `src/virtual_fly/` へ置き、このディレクトリはI/Oと実験組み立てを中心にする。

## 現行Flyppy評価

- `evaluate_flyppy_fixed.py` — plasticityを止めた固定suite
- `export_flyppy_report.py` — 最新run report
- `export_flyppy_population_report.py` — population report
- `append_flyppy_run_history.py` — run履歴
- `compare_flyppy_scheduler_runs.py` — async / wave比較

## causal / learning診断

- `analyze_flyppy_async_bias.py`
- `analyze_flyppy_motor_weight_reach.py`
- `analyze_flyppy_reinforcement_reach.py`
- `compare_flyppy_fixed_motor.py`
- `profile_flyppy_plasticity_live.py`
- `profile_flyppy_transaction_dirty.py`

one-off出力は `reports/flyppy/diagnostics/` へ置く。

## vision / body / runtime診断

`probe_*`, `verify_*`, `benchmark_*`, `profile_*` の各scriptは、production意味論を変更せず物理・視覚・runtime性能を測るためのものとする。

新しい解析scriptで共通処理が増えた場合、script間importを増やすより `src/virtual_fly/` packageへ切り出す。
