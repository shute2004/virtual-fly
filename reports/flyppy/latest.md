# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- curriculum target gate: 1
- episode: 12〜19（8 episode）
- elapsed: 127.034 s（平均 15.879 s/episode）
- checkpoint neural step: 1175
- target gate成功episode: 4/8（50.0%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 18.346 mm
- 最も小さいspawn xでの成功: episode 18 / x=6.050 / z=9.793 / vx=325.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 5.912
- spawn_z_mm: 9.793
- initial_speed_mm_s: 325.000
- successful_first_gates: 11
- consecutive_failures: 1
- curriculum_episodes: 16
- target_x_mm: 0.000
- target_z_mm: 8.250
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 9.793 | 325.0 | 6.050 | 18 | 1 |

## Collision

- gate: 6
- floor: 2

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 12 | 6.462 | 9.793 | 325.0 | 42 | 1 | gate | 12.320 | 176.545 |
| 13 | 6.187 | 9.655 | 312.5 | 32 | 0 | gate | 10.639 | 172.900 |
| 14 | 6.325 | 9.793 | 325.0 | 110 | 1 | floor | 18.346 | 160.704 |
| 15 | 6.050 | 9.655 | 312.5 | 32 | 0 | gate | 10.502 | 172.900 |
| 16 | 6.187 | 9.793 | 325.0 | 109 | 1 | floor | 17.935 | 183.432 |
| 17 | 5.912 | 9.655 | 312.5 | 32 | 0 | gate | 10.364 | 172.899 |
| 18 | 6.050 | 9.793 | 325.0 | 38 | 1 | gate | 11.455 | 236.547 |
| 19 | 5.775 | 9.655 | 312.5 | 32 | 0 | gate | 10.233 | 189.538 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
