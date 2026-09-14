# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- curriculum target gate: 1
- episode: 4〜7（4 episode）
- elapsed: 159.410 s（平均 39.852 s/episode）
- checkpoint neural step: 700
- target gate成功episode: 3/4（75.0%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 19.390 mm
- 最も小さいspawn xでの成功: episode 6 / x=6.600 / z=9.793 / vx=325.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 6.462
- spawn_z_mm: 9.793
- initial_speed_mm_s: 325.000
- successful_first_gates: 7
- consecutive_failures: 1
- curriculum_episodes: 8
- target_x_mm: 0.000
- target_z_mm: 8.250
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 9.793 | 325.0 | 6.600 | 6 | 1 |
| 9.930 | 337.5 | 6.875 | 5 | 1 |
| 10.068 | 350.0 | 7.150 | 4 | 1 |

## Collision

- gate: 4

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 4 | 7.150 | 10.068 | 350.0 | 97 | 1 | gate | 19.390 | 20.324 |
| 5 | 6.875 | 9.930 | 337.5 | 106 | 1 | gate | 19.387 | 11.464 |
| 6 | 6.600 | 9.793 | 325.0 | 47 | 1 | gate | 13.075 | 175.905 |
| 7 | 6.325 | 9.655 | 312.5 | 32 | 0 | gate | 10.777 | 172.916 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
