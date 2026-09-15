# Flyppy latest run

## 集計

- backend: `gpu-population:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- adaptive training criterion: first gate pass (`passed_gates > 0`)
- episode: 152〜159（8 episode）
- elapsed: 12.209 s（平均 1.526 s/episode）
- checkpoint neural step: 10984
- first gate通過episode: 2/8（25.0%）
- first gate後にさらに1 gate以上通過: 0 episode
- first gate通過後に衝突: 2 episode
- 無衝突episode: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 23.152 mm
- spawn高度を一度でも上回ったepisode: 0/8
- run内最大高度gain: -0.006 mm
- run内最大高度loss: 3.349 mm
- episodeごとの最大高度gain平均: -0.006 mm
- 最も小さいspawn xでのfirst gate通過: episode 152 / x=2.970 / z=11.467 / vx=450.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 3.267
- spawn_z_mm: 11.467
- initial_speed_mm_s: 450.000
- successful_first_gates: 66
- consecutive_failures: 1
- curriculum_episodes: 160
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## First-gate成功境界

同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 11.467 | 450.0 | 2.970 | 152 | 1 |

## Collision

- gate: 8

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 152 | 2.970 | 11.467 | 450.0 | 93 | 1 | gate | 8.119 | 11.462 | 23.152 | 291.071 |
| 153 | 2.970 | 11.467 | 450.0 | 32 | 0 | gate | 11.233 | 11.462 | 10.223 | 183.936 |
| 154 | 2.970 | 11.467 | 450.0 | 89 | 1 | gate | 8.843 | 11.462 | 22.678 | 121.637 |
| 155 | 2.970 | 11.467 | 450.0 | 33 | 0 | gate | 11.194 | 11.462 | 10.474 | 335.714 |
| 156 | 3.118 | 11.467 | 450.0 | 31 | 0 | gate | 11.239 | 11.462 | 10.169 | 318.321 |
| 157 | 3.267 | 11.467 | 450.0 | 32 | 0 | gate | 11.215 | 11.462 | 10.476 | 317.623 |
| 158 | 3.415 | 11.467 | 450.0 | 29 | 0 | gate | 11.271 | 11.462 | 9.914 | -113.802 |
| 159 | 3.564 | 11.467 | 450.0 | 29 | 0 | gate | 11.271 | 11.462 | 9.979 | -127.466 |

## 判定用メモ

- adaptive curriculumの現在の成功条件は `passed_gates > 0`、すなわちfirst gate通過です。episode全体の無衝突成功を意味しません。
- `training_gate_index` がstateに存在する場合のみ、その値を明示します。0始まりです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
