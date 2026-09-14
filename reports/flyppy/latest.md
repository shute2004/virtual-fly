# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum target gate: 2
- episode: 96〜119（24 episode）
- elapsed: 949.973 s（平均 39.582 s/episode）
- checkpoint neural step: 35221
- target gate成功episode: 24/24（100.0%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 20.589 mm
- 最も小さいspawn xでの成功: episode 96 / x=10.625 / z=5.353 / vx=356.2 / passed=1

## 最終カリキュラム状態

- training_gate_index: 1
- spawn_x_mm: 10.625
- spawn_z_mm: 5.353
- initial_speed_mm_s: 356.250
- successful_first_gates: 50
- consecutive_failures: 0
- curriculum_episodes: 120
- target_x_mm: 10.625
- target_z_mm: 5.353
- target_speed_mm_s: 356.250
- curriculum_complete: true

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 5.353 | 356.2 | 10.625 | 96 | 1 |

## Collision

- floor: 22

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 96 | 10.625 | 5.353 | 356.2 | 168 | 1 | floor | 19.306 | -13.580 |
| 97 | 10.625 | 5.353 | 356.2 | 307 | 1 | floor | 18.590 | 92.721 |
| 98 | 10.625 | 5.353 | 356.2 | 1800 | 1 | - | 18.667 | -1.126 |
| 99 | 10.625 | 5.353 | 356.2 | 242 | 1 | floor | 19.676 | 24.604 |
| 100 | 10.625 | 5.353 | 356.2 | 201 | 1 | floor | 19.374 | -2.318 |
| 101 | 10.625 | 5.353 | 356.2 | 473 | 1 | floor | 20.564 | -23.839 |
| 102 | 10.625 | 5.353 | 356.2 | 257 | 1 | floor | 19.649 | -7.568 |
| 103 | 10.625 | 5.353 | 356.2 | 795 | 1 | floor | 17.531 | 10.823 |
| 104 | 10.625 | 5.353 | 356.2 | 269 | 1 | floor | 19.746 | -11.074 |
| 105 | 10.625 | 5.353 | 356.2 | 179 | 1 | floor | 18.543 | -31.643 |
| 106 | 10.625 | 5.353 | 356.2 | 717 | 1 | floor | 18.089 | 42.953 |
| 107 | 10.625 | 5.353 | 356.2 | 1800 | 1 | - | 20.282 | -10.088 |
| 108 | 10.625 | 5.353 | 356.2 | 311 | 1 | floor | 19.357 | -0.394 |
| 109 | 10.625 | 5.353 | 356.2 | 201 | 1 | floor | 19.489 | 62.017 |
| 110 | 10.625 | 5.353 | 356.2 | 172 | 1 | floor | 19.650 | -24.407 |
| 111 | 10.625 | 5.353 | 356.2 | 770 | 1 | floor | 18.717 | 38.049 |
| 112 | 10.625 | 5.353 | 356.2 | 228 | 1 | floor | 18.456 | -9.150 |
| 113 | 10.625 | 5.353 | 356.2 | 576 | 1 | floor | 17.921 | 21.838 |
| 114 | 10.625 | 5.353 | 356.2 | 273 | 1 | floor | 19.414 | 19.774 |
| 115 | 10.625 | 5.353 | 356.2 | 261 | 1 | floor | 20.549 | 24.179 |
| 116 | 10.625 | 5.353 | 356.2 | 269 | 1 | floor | 19.125 | 78.086 |
| 117 | 10.625 | 5.353 | 356.2 | 350 | 1 | floor | 20.589 | -27.512 |
| 118 | 10.625 | 5.353 | 356.2 | 235 | 1 | floor | 18.499 | 48.924 |
| 119 | 10.625 | 5.353 | 356.2 | 260 | 1 | floor | 17.921 | 0.155 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
