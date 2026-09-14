# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- curriculum target gate: 1
- episode: 0〜11（12 episode）
- elapsed: 180.020 s（平均 15.002 s/episode）
- checkpoint neural step: 1252
- target gate成功episode: 12/12（100.0%）
- target gate後にさらに1 gate以上通過: 5 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 2
- run内最大x: 33.338 mm
- 最も小さいspawn xでの成功: episode 11 / x=5.643 / z=9.834 / vx=300.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 5.346
- spawn_z_mm: 9.685
- initial_speed_mm_s: 300.000
- successful_first_gates: 12
- consecutive_failures: 0
- curriculum_episodes: 12
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 9.834 | 300.0 | 5.643 | 11 | 1 |
| 9.982 | 300.0 | 5.940 | 10 | 1 |
| 10.131 | 300.0 | 6.237 | 9 | 1 |
| 10.279 | 300.0 | 6.534 | 8 | 1 |
| 10.428 | 312.5 | 6.831 | 7 | 1 |
| 10.576 | 325.0 | 7.128 | 6 | 1 |
| 10.725 | 337.5 | 7.425 | 5 | 1 |
| 10.873 | 350.0 | 7.722 | 4 | 2 |
| 11.022 | 362.5 | 8.019 | 3 | 2 |
| 11.170 | 375.0 | 8.316 | 2 | 2 |
| 11.319 | 387.5 | 8.613 | 1 | 2 |
| 11.467 | 400.0 | 8.910 | 0 | 2 |

## Collision

- gate: 12

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 8.910 | 11.467 | 400.0 | 115 | 2 | gate | 32.759 | 279.168 |
| 1 | 8.613 | 11.319 | 387.5 | 123 | 2 | gate | 33.338 | -184.002 |
| 2 | 8.316 | 11.170 | 375.0 | 83 | 2 | gate | 24.560 | 456.548 |
| 3 | 8.019 | 11.022 | 362.5 | 84 | 2 | gate | 24.266 | 440.678 |
| 4 | 7.722 | 10.873 | 350.0 | 86 | 2 | gate | 24.114 | 395.469 |
| 5 | 7.425 | 10.725 | 337.5 | 87 | 1 | gate | 23.403 | 235.724 |
| 6 | 7.128 | 10.576 | 325.0 | 89 | 1 | gate | 23.002 | -28.485 |
| 7 | 6.831 | 10.428 | 312.5 | 91 | 1 | gate | 22.613 | 453.356 |
| 8 | 6.534 | 10.279 | 300.0 | 90 | 1 | gate | 21.590 | 335.720 |
| 9 | 6.237 | 10.131 | 300.0 | 96 | 1 | gate | 22.468 | -347.342 |
| 10 | 5.940 | 9.982 | 300.0 | 96 | 1 | gate | 22.086 | -534.832 |
| 11 | 5.643 | 9.834 | 300.0 | 96 | 1 | gate | 22.304 | 151.679 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
