# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum target gate: 2
- episode: 144〜167（24 episode）
- elapsed: 435.982 s（平均 18.166 s/episode）
- checkpoint neural step: 44309
- target gate成功episode: 0/24（0.0%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 0
- run内最大x: 14.261 mm

## 最終カリキュラム状態

- training_gate_index: 1
- spawn_x_mm: 10.500
- spawn_z_mm: 5.290
- initial_speed_mm_s: 350.000
- successful_first_gates: 50
- consecutive_failures: 24
- curriculum_episodes: 168
- target_x_mm: 10.500
- target_z_mm: 5.290
- target_speed_mm_s: 350.000
- curriculum_complete: false

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| - | - | - | - | - |

## Collision

- gate: 13
- floor: 11

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 144 | 10.500 | 5.290 | 350.0 | 110 | 0 | gate | 14.246 | 5.091 |
| 145 | 10.500 | 5.290 | 350.0 | 179 | 0 | gate | 14.247 | 21.434 |
| 146 | 10.500 | 5.290 | 350.0 | 117 | 0 | gate | 14.247 | -22.256 |
| 147 | 10.500 | 5.290 | 350.0 | 255 | 0 | floor | 14.246 | -69.035 |
| 148 | 10.500 | 5.290 | 350.0 | 159 | 0 | gate | 14.244 | 17.066 |
| 149 | 10.500 | 5.290 | 350.0 | 168 | 0 | gate | 14.257 | 44.224 |
| 150 | 10.500 | 5.290 | 350.0 | 230 | 0 | floor | 14.255 | -1.290 |
| 151 | 10.500 | 5.290 | 350.0 | 162 | 0 | gate | 14.256 | 43.651 |
| 152 | 10.500 | 5.290 | 350.0 | 165 | 0 | gate | 14.256 | 16.650 |
| 153 | 10.500 | 5.290 | 350.0 | 164 | 0 | gate | 14.260 | 46.520 |
| 154 | 10.500 | 5.290 | 350.0 | 230 | 0 | floor | 14.255 | -35.590 |
| 155 | 10.500 | 5.290 | 350.0 | 141 | 0 | gate | 14.261 | 36.048 |
| 156 | 10.500 | 5.290 | 350.0 | 180 | 0 | floor | 14.261 | 7.395 |
| 157 | 10.500 | 5.290 | 350.0 | 225 | 0 | floor | 14.229 | 13.022 |
| 158 | 10.500 | 5.290 | 350.0 | 204 | 0 | floor | 14.229 | -41.276 |
| 159 | 10.500 | 5.290 | 350.0 | 213 | 0 | floor | 14.233 | -55.882 |
| 160 | 10.500 | 5.290 | 350.0 | 268 | 0 | floor | 14.233 | -26.045 |
| 161 | 10.500 | 5.290 | 350.0 | 157 | 0 | gate | 14.232 | 21.948 |
| 162 | 10.500 | 5.290 | 350.0 | 222 | 0 | floor | 14.232 | -36.015 |
| 163 | 10.500 | 5.290 | 350.0 | 164 | 0 | gate | 14.232 | 62.458 |
| 164 | 10.500 | 5.290 | 350.0 | 162 | 0 | gate | 14.232 | 10.105 |
| 165 | 10.500 | 5.290 | 350.0 | 172 | 0 | floor | 14.232 | -53.395 |
| 166 | 10.500 | 5.290 | 350.0 | 266 | 0 | floor | 14.232 | -40.501 |
| 167 | 10.500 | 5.290 | 350.0 | 157 | 0 | gate | 14.232 | 20.871 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
