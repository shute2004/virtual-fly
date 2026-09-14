# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum target gate: 2
- episode: 72〜95（24 episode）
- elapsed: 554.023 s（平均 23.084 s/episode）
- checkpoint neural step: 23923
- target gate成功episode: 12/24（50.0%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 20.941 mm
- 最も小さいspawn xでの成功: episode 73 / x=10.750 / z=5.415 / vx=362.5 / passed=1

## 最終カリキュラム状態

- training_gate_index: 1
- spawn_x_mm: 10.500
- spawn_z_mm: 5.290
- initial_speed_mm_s: 350.000
- successful_first_gates: 26
- consecutive_failures: 0
- curriculum_episodes: 96
- target_x_mm: 9.000
- target_z_mm: 5.000
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 5.415 | 362.5 | 10.750 | 73 | 1 |

## Collision

- floor: 21
- gate: 3

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 72 | 10.500 | 5.290 | 350.0 | 263 | 0 | floor | 14.269 | -16.664 |
| 73 | 10.750 | 5.415 | 362.5 | 148 | 1 | floor | 19.663 | -38.325 |
| 74 | 10.500 | 5.290 | 350.0 | 220 | 0 | floor | 14.269 | -31.277 |
| 75 | 10.750 | 5.415 | 362.5 | 843 | 1 | floor | 19.123 | 61.230 |
| 76 | 10.500 | 5.290 | 350.0 | 222 | 0 | floor | 14.253 | -24.648 |
| 77 | 10.750 | 5.415 | 362.5 | 166 | 1 | floor | 18.942 | -2.813 |
| 78 | 10.500 | 5.290 | 350.0 | 165 | 0 | gate | 14.254 | 31.037 |
| 79 | 10.750 | 5.415 | 362.5 | 267 | 1 | floor | 18.671 | 38.982 |
| 80 | 10.500 | 5.290 | 350.0 | 255 | 0 | floor | 14.255 | -70.391 |
| 81 | 10.750 | 5.415 | 362.5 | 199 | 1 | floor | 18.958 | -49.708 |
| 82 | 10.500 | 5.290 | 350.0 | 161 | 0 | gate | 14.256 | 26.750 |
| 83 | 10.750 | 5.415 | 362.5 | 262 | 1 | floor | 20.902 | 14.724 |
| 84 | 10.500 | 5.290 | 350.0 | 174 | 0 | gate | 14.257 | 26.429 |
| 85 | 10.750 | 5.415 | 362.5 | 619 | 1 | floor | 20.941 | -6.305 |
| 86 | 10.500 | 5.290 | 350.0 | 175 | 0 | floor | 14.278 | 21.497 |
| 87 | 10.750 | 5.415 | 362.5 | 155 | 1 | floor | 18.947 | 12.666 |
| 88 | 10.500 | 5.290 | 350.0 | 267 | 0 | floor | 14.263 | -30.515 |
| 89 | 10.750 | 5.415 | 362.5 | 155 | 1 | floor | 19.535 | 23.744 |
| 90 | 10.500 | 5.290 | 350.0 | 222 | 0 | floor | 14.272 | -24.074 |
| 91 | 10.750 | 5.415 | 362.5 | 198 | 1 | floor | 19.663 | -19.031 |
| 92 | 10.500 | 5.290 | 350.0 | 230 | 0 | floor | 14.269 | -8.164 |
| 93 | 10.750 | 5.415 | 362.5 | 161 | 1 | floor | 19.521 | 20.905 |
| 94 | 10.500 | 5.290 | 350.0 | 196 | 0 | floor | 14.268 | -25.433 |
| 95 | 10.750 | 5.415 | 362.5 | 218 | 1 | floor | 19.597 | -16.442 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
