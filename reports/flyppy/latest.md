# Flyppy latest run

## 集計

- backend: `gpu-population:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- adaptive training criterion: first gate pass (`passed_gates > 0`)
- episode: 184〜207（24 episode）
- elapsed: 31.016 s（平均 1.292 s/episode）
- checkpoint neural step: 13213
- first gate通過episode: 6/24（25.0%）
- first gate後にさらに1 gate以上通過: 1 episode
- first gate通過後に衝突: 6 episode
- 無衝突episode: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 2
- run内最大x: 24.757 mm
- spawn高度を一度でも上回ったepisode: 0/24
- run内最大高度gain: -0.005 mm
- run内最大高度loss: 2.964 mm
- episodeごとの最大高度gain平均: -0.006 mm
- 最も小さいspawn xでのfirst gate通過: episode 184 / x=4.158 / z=11.170 / vx=425.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 5.049
- spawn_z_mm: 11.170
- initial_speed_mm_s: 425.000
- successful_first_gates: 78
- consecutive_failures: 0
- curriculum_episodes: 208
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## First-gate成功境界

同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 11.170 | 425.0 | 4.158 | 184 | 1 |
| 11.319 | 437.5 | 4.752 | 194 | 1 |

## Collision

- gate: 24

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 184 | 4.158 | 11.170 | 425.0 | 92 | 1 | gate | 8.206 | 11.165 | 23.703 | 358.382 |
| 185 | 4.158 | 11.170 | 425.0 | 28 | 0 | gate | 10.974 | 11.165 | 10.168 | -61.836 |
| 186 | 4.158 | 11.170 | 425.0 | 87 | 1 | gate | 9.338 | 11.165 | 22.780 | 112.155 |
| 187 | 4.158 | 11.170 | 425.0 | 28 | 0 | gate | 10.974 | 11.165 | 10.236 | 285.105 |
| 188 | 4.306 | 11.319 | 437.5 | 26 | 0 | gate | 11.131 | 11.313 | 10.123 | 163.165 |
| 189 | 4.455 | 11.467 | 450.0 | 25 | 0 | gate | 11.295 | 11.462 | 10.136 | 98.516 |
| 190 | 4.603 | 11.467 | 450.0 | 24 | 0 | gate | 11.309 | 11.462 | 10.114 | 146.555 |
| 191 | 4.752 | 11.467 | 450.0 | 23 | 0 | gate | 11.311 | 11.462 | 10.075 | 292.575 |
| 192 | 4.900 | 11.467 | 450.0 | 23 | 0 | gate | 11.324 | 11.462 | 10.124 | 96.888 |
| 193 | 5.049 | 11.467 | 450.0 | 22 | 0 | gate | 11.337 | 11.462 | 10.100 | 167.199 |
| 194 | 4.752 | 11.319 | 437.5 | 83 | 1 | gate | 9.322 | 11.313 | 22.768 | 168.381 |
| 195 | 4.455 | 11.170 | 425.0 | 91 | 1 | gate | 8.283 | 11.165 | 23.950 | 445.190 |
| 196 | 4.603 | 11.319 | 437.5 | 27 | 0 | gate | 11.107 | 11.313 | 10.559 | 159.259 |
| 197 | 4.752 | 11.467 | 450.0 | 23 | 0 | gate | 11.311 | 11.462 | 10.075 | 292.575 |
| 198 | 4.900 | 11.467 | 450.0 | 23 | 0 | gate | 11.324 | 11.462 | 10.124 | 96.888 |
| 199 | 5.049 | 11.467 | 450.0 | 22 | 0 | gate | 11.337 | 11.462 | 10.100 | 167.199 |
| 200 | 5.197 | 11.467 | 450.0 | 20 | 0 | gate | 11.349 | 11.462 | 9.763 | -77.715 |
| 201 | 5.346 | 11.467 | 450.0 | 20 | 0 | gate | 11.349 | 11.462 | 9.798 | -177.979 |
| 202 | 5.494 | 11.467 | 450.0 | 20 | 0 | gate | 11.349 | 11.462 | 9.899 | -227.291 |
| 203 | 5.643 | 11.467 | 450.0 | 19 | 0 | gate | 11.349 | 11.462 | 10.034 | -30.619 |
| 204 | 5.346 | 11.319 | 437.5 | 80 | 1 | gate | 9.287 | 11.313 | 22.753 | 164.377 |
| 205 | 5.049 | 11.170 | 425.0 | 92 | 2 | gate | 8.564 | 11.165 | 24.757 | 432.399 |
| 206 | 5.198 | 11.319 | 437.5 | 22 | 0 | gate | 11.189 | 11.313 | 10.107 | 136.603 |
| 207 | 5.346 | 11.467 | 450.0 | 20 | 0 | gate | 11.349 | 11.462 | 9.798 | -177.979 |

## 判定用メモ

- adaptive curriculumの現在の成功条件は `passed_gates > 0`、すなわちfirst gate通過です。episode全体の無衝突成功を意味しません。
- `training_gate_index` がstateに存在する場合のみ、その値を明示します。0始まりです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
