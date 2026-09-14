# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `boundary-band`
- curriculum target gate: 2
- episode: 168〜191（24 episode）
- elapsed: 515.579 s（平均 21.482 s/episode）
- checkpoint neural step: 49403
- target gate成功episode: 5/24（20.8%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 19.585 mm
- 最も小さいspawn xでの成功: episode 176 / x=10.562 / z=5.321 / vx=353.1 / passed=1

## 最終カリキュラム状態

- training_gate_index: 1
- training_phase: gate2_boundary_band
- curriculum_mode: boundary-band
- spawn_x_mm: 10.625
- spawn_z_mm: 5.353
- initial_speed_mm_s: 356.250
- successful_first_gates: 55
- consecutive_failures: 6
- curriculum_episodes: 192
- target_x_mm: 9.000
- target_z_mm: 5.000
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 境界帯カリキュラム

- batch_number: 1
- attempts_in_batch: 0
- successes_in_batch: 0
- last_batch_success_rate: 0.208
- last_adjustment: easier
- harder_shifts: 0
- easier_shifts: 1
- hard endpoint: x=10.562, z=5.321, vx=353.125
- easy endpoint: x=10.688, z=5.384, vx=359.375

| ease level | episode数 | 成功 | 成功率 |
|---:|---:|---:|---:|
| 0.00 | 4 | 0 | 0.0% |
| 0.25 | 5 | 0 | 0.0% |
| 0.50 | 6 | 2 | 33.3% |
| 0.75 | 5 | 1 | 20.0% |
| 1.00 | 4 | 2 | 50.0% |

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 5.321 | 353.1 | 10.562 | 176 | 1 |
| 5.337 | 354.7 | 10.594 | 181 | 1 |
| 5.353 | 356.2 | 10.625 | 174 | 1 |

## Collision

- gate: 15
- floor: 9

## Episode一覧

| ep | ease | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 168 | 0.50 | 10.562 | 5.321 | 353.1 | 139 | 0 | gate | 14.367 | 16.177 |
| 169 | 0.00 | 10.500 | 5.290 | 350.0 | 118 | 0 | gate | 14.338 | -13.349 |
| 170 | 0.75 | 10.594 | 5.337 | 354.7 | 84 | 0 | gate | 14.347 | -25.288 |
| 171 | 0.25 | 10.531 | 5.306 | 351.6 | 441 | 0 | floor | 14.319 | -28.479 |
| 172 | 0.25 | 10.531 | 5.306 | 351.6 | 220 | 0 | gate | 14.318 | 23.330 |
| 173 | 0.50 | 10.562 | 5.321 | 353.1 | 134 | 0 | gate | 14.363 | 31.236 |
| 174 | 1.00 | 10.625 | 5.353 | 356.2 | 223 | 1 | floor | 19.075 | -63.305 |
| 175 | 1.00 | 10.625 | 5.353 | 356.2 | 116 | 0 | gate | 14.860 | 56.570 |
| 176 | 0.50 | 10.562 | 5.321 | 353.1 | 355 | 1 | floor | 19.378 | 20.726 |
| 177 | 0.25 | 10.531 | 5.306 | 351.6 | 219 | 0 | floor | 14.322 | -47.704 |
| 178 | 0.00 | 10.500 | 5.290 | 350.0 | 155 | 0 | gate | 14.265 | 31.103 |
| 179 | 1.00 | 10.625 | 5.353 | 356.2 | 110 | 0 | gate | 14.739 | 26.300 |
| 180 | 0.00 | 10.500 | 5.290 | 350.0 | 125 | 0 | gate | 14.250 | 27.841 |
| 181 | 0.75 | 10.594 | 5.337 | 354.7 | 818 | 1 | floor | 19.585 | 78.611 |
| 182 | 0.25 | 10.531 | 5.306 | 351.6 | 200 | 0 | floor | 14.314 | -61.557 |
| 183 | 0.75 | 10.594 | 5.337 | 354.7 | 112 | 0 | gate | 14.518 | 4.299 |
| 184 | 0.50 | 10.562 | 5.321 | 353.1 | 329 | 1 | floor | 19.511 | 42.295 |
| 185 | 1.00 | 10.625 | 5.353 | 356.2 | 183 | 1 | floor | 19.365 | -11.919 |
| 186 | 0.75 | 10.594 | 5.337 | 354.7 | 98 | 0 | gate | 14.347 | 17.635 |
| 187 | 0.75 | 10.594 | 5.337 | 354.7 | 106 | 0 | gate | 14.347 | 66.874 |
| 188 | 0.25 | 10.531 | 5.306 | 351.6 | 248 | 0 | floor | 14.318 | -43.997 |
| 189 | 0.00 | 10.500 | 5.290 | 350.0 | 119 | 0 | gate | 14.336 | -1.259 |
| 190 | 0.50 | 10.562 | 5.321 | 353.1 | 184 | 0 | gate | 14.776 | 23.571 |
| 191 | 0.50 | 10.562 | 5.321 | 353.1 | 142 | 0 | gate | 14.363 | 37.083 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
