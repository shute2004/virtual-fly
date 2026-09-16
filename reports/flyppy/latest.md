# Flyppy latest run

## 集計

- backend: `gpu-population:Apple M1 (Metal)`
- curriculum mode: `boundary-band`
- launch mode: `async`
- adaptive training criterion: first gate pass (`passed_gates > 0`)
- episode: 232〜255（24 episode）
- elapsed: 50.371 s（平均 2.099 s/episode）
- checkpoint neural step: 15781
- first gate通過episode: 12/24（50.0%）
- first gate後にさらに1 gate以上通過: 4 episode
- first gate通過後に衝突: 12 episode
- 無衝突episode: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 2
- run内最大x: 24.467 mm
- spawn高度を一度でも上回ったepisode: 0/24
- run内最大高度gain: -0.005 mm
- run内最大高度loss: 2.981 mm
- episodeごとの最大高度gain平均: -0.005 mm
- 最も小さいspawn xでのfirst gate通過: episode 234 / x=4.752 / z=11.022 / vx=412.5 / passed=1

## 最終カリキュラム状態

- curriculum_mode: boundary-band
- spawn_x_mm: 4.975
- spawn_z_mm: 11.170
- initial_speed_mm_s: 425.000
- successful_first_gates: 96
- consecutive_failures: 0
- curriculum_episodes: 256
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 境界帯カリキュラム

- batch_number: 2
- attempts_in_batch: 0
- successes_in_batch: 0
- last_batch_success_rate: 0.500
- last_batch_raw_success_rate: 0.500
- last_batch_aggregation: equal_group_mean
- last_batch_group_success_rates: {'0': 1.0, '1': 0.0, '2': 1.0, '3': 0.0}
- last_adjustment: hold
- harder_shifts: 0
- easier_shifts: 0
- hard endpoint: x=4.752, z=11.022, vx=412.500
- easy endpoint: x=5.198, z=11.319, vx=437.500

| ease level | episode数 | first gate通過 | 通過率 |
|---:|---:|---:|---:|
| 0.00 | 4 | 3 | 75.0% |
| 0.25 | 5 | 1 | 20.0% |
| 0.50 | 6 | 4 | 66.7% |
| 0.75 | 5 | 3 | 60.0% |
| 1.00 | 4 | 1 | 25.0% |

## First-gate成功境界

同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 11.022 | 412.5 | 4.752 | 234 | 1 |
| 11.096 | 418.8 | 4.863 | 253 | 2 |
| 11.170 | 425.0 | 4.975 | 232 | 2 |
| 11.245 | 431.2 | 5.086 | 242 | 1 |
| 11.319 | 437.5 | 5.198 | 249 | 2 |

## Collision

- gate: 24

## Episode一覧

| ep | ease | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 232 | 0.50 | 4.975 | 11.170 | 425.0 | 89 | 2 | gate | 8.461 | 11.165 | 24.172 | 379.624 |
| 233 | 0.75 | 5.086 | 11.245 | 431.2 | 23 | 0 | gate | 11.102 | 11.239 | 10.130 | 88.030 |
| 234 | 0.00 | 4.752 | 11.022 | 412.5 | 86 | 1 | gate | 8.402 | 11.016 | 22.672 | 194.728 |
| 235 | 0.25 | 4.863 | 11.096 | 418.8 | 27 | 0 | gate | 10.900 | 11.091 | 10.595 | -10.812 |
| 236 | 0.50 | 4.975 | 11.170 | 425.0 | 24 | 0 | gate | 11.013 | 11.165 | 10.141 | 73.464 |
| 237 | 0.25 | 4.863 | 11.096 | 418.8 | 27 | 0 | gate | 10.900 | 11.091 | 10.595 | -10.812 |
| 238 | 1.00 | 5.198 | 11.319 | 437.5 | 22 | 0 | gate | 11.189 | 11.313 | 10.107 | 136.603 |
| 239 | 0.00 | 4.752 | 11.022 | 412.5 | 28 | 0 | gate | 10.826 | 11.016 | 10.587 | -13.743 |
| 240 | 0.50 | 4.975 | 11.170 | 425.0 | 24 | 0 | gate | 11.013 | 11.165 | 10.141 | 73.464 |
| 241 | 0.25 | 4.863 | 11.096 | 418.8 | 27 | 0 | gate | 10.900 | 11.091 | 10.595 | -10.812 |
| 242 | 0.75 | 5.086 | 11.245 | 431.2 | 82 | 1 | gate | 9.117 | 11.239 | 22.711 | 122.786 |
| 243 | 0.00 | 4.752 | 11.022 | 412.5 | 88 | 1 | gate | 8.088 | 11.016 | 23.116 | 223.996 |
| 244 | 1.00 | 5.198 | 11.319 | 437.5 | 22 | 0 | gate | 11.189 | 11.313 | 10.107 | 136.603 |
| 245 | 1.00 | 5.198 | 11.319 | 437.5 | 22 | 0 | gate | 11.189 | 11.313 | 10.107 | 136.603 |
| 246 | 0.75 | 5.086 | 11.245 | 431.2 | 23 | 0 | gate | 11.102 | 11.239 | 10.130 | 88.030 |
| 247 | 0.25 | 4.863 | 11.096 | 418.8 | 27 | 0 | gate | 10.900 | 11.091 | 10.595 | -10.812 |
| 248 | 0.50 | 4.975 | 11.170 | 425.0 | 83 | 1 | gate | 8.877 | 11.165 | 22.665 | 170.121 |
| 249 | 1.00 | 5.198 | 11.319 | 437.5 | 86 | 2 | gate | 8.995 | 11.313 | 24.012 | 431.064 |
| 250 | 0.75 | 5.086 | 11.245 | 431.2 | 82 | 1 | gate | 9.136 | 11.239 | 22.724 | 71.030 |
| 251 | 0.00 | 4.752 | 11.022 | 412.5 | 88 | 1 | gate | 8.041 | 11.016 | 23.095 | 207.568 |
| 252 | 0.50 | 4.975 | 11.170 | 425.0 | 83 | 1 | gate | 8.939 | 11.165 | 22.659 | 173.139 |
| 253 | 0.25 | 4.863 | 11.096 | 418.8 | 92 | 2 | gate | 8.449 | 11.091 | 24.467 | 506.108 |
| 254 | 0.75 | 5.086 | 11.245 | 431.2 | 82 | 1 | gate | 9.055 | 11.239 | 22.712 | 170.951 |
| 255 | 0.50 | 4.975 | 11.170 | 425.0 | 89 | 2 | gate | 8.474 | 11.165 | 24.230 | 383.553 |

## 判定用メモ

- adaptive curriculumの現在の成功条件は `passed_gates > 0`、すなわちfirst gate通過です。episode全体の無衝突成功を意味しません。
- `training_gate_index` がstateに存在する場合のみ、その値を明示します。0始まりです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
