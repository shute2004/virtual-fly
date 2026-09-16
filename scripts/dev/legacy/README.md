# legacy Flyppy launchers

過去のFlyppy v1/v2および旧single-v3経路を再現するためのlauncherを保存する。
現行production開発には使用しない。

- `train_flyppy_v2.sh` — 既知の非飛行v2基準。明示overrideなしでは学習を拒否する。
- `view_flyppy_v2.sh` — v2 viewer
- `train_flyppy_v3.sh` — shared-weight population化前のsingle-v3入口
- `train_flyppy_gate2.sh` — v1 gate2部分練習
- `train_flyppy_gate2_band.sh` — v1 gate2 boundary-band試作
- `train_flyppy_gate2_boundary.sh` — v1 gate2境界固定試作
- `train_flyppy_gate2_hard.sh` — v1 gate2 hard条件固定試作

現行入口は一階層上の `scripts/dev/README.md` を参照する。
