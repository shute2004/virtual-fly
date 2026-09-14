# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- adaptive training criterion: first gate pass (`passed_gates > 0`)
- episode: 12〜23（12 episode）
- elapsed: 164.583 s（平均 13.715 s/episode）
- checkpoint neural step: 2254
- first gate通過episode: 7/12（58.3%）
- first gate後にさらに1 gate以上通過: 1 episode
- first gate通過後に衝突: 7 episode
- 無衝突episode: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 2
- run内最大x: 30.391 mm
- spawn高度を一度でも上回ったepisode: 4/12
- run内最大高度gain: +5.465 mm
- run内最大高度loss: 7.358 mm
- episodeごとの最大高度gain平均: +1.436 mm
- 最も小さいspawn xでのfirst gate通過: episode 22 / x=4.158 / z=9.388 / vx=312.5 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 4.010
- spawn_z_mm: 9.388
- initial_speed_mm_s: 312.500
- successful_first_gates: 19
- consecutive_failures: 1
- curriculum_episodes: 24
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## First-gate成功境界

同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 9.388 | 300.0 | 4.752 | 14 | 1 |
| 9.388 | 312.5 | 4.158 | 22 | 1 |
| 9.537 | 300.0 | 5.049 | 13 | 2 |
| 9.537 | 325.0 | 4.455 | 21 | 1 |
| 9.685 | 300.0 | 5.346 | 12 | 1 |

## Collision

- gate: 11
- floor: 1

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 12 | 5.346 | 9.685 | 300.0 | 96 | 1 | gate | 5.345 | 9.680 | 21.732 | -227.839 |
| 13 | 5.049 | 9.537 | 300.0 | 181 | 2 | floor | 2.179 | 11.671 | 30.391 | 178.598 |
| 14 | 4.752 | 9.388 | 300.0 | 84 | 1 | gate | 9.063 | 14.262 | 11.952 | 11.844 |
| 15 | 4.455 | 9.240 | 300.0 | 37 | 0 | gate | 8.914 | 9.234 | 10.410 | 278.696 |
| 16 | 4.604 | 9.388 | 312.5 | 92 | 1 | gate | 9.064 | 14.853 | 12.549 | 34.049 |
| 17 | 4.307 | 9.240 | 300.0 | 38 | 0 | gate | 8.914 | 9.234 | 10.266 | -128.006 |
| 18 | 4.455 | 9.388 | 312.5 | 38 | 0 | gate | 9.062 | 9.383 | 10.741 | 107.079 |
| 19 | 4.604 | 9.537 | 325.0 | 96 | 1 | gate | 5.489 | 9.531 | 21.874 | -197.904 |
| 20 | 4.307 | 9.388 | 312.5 | 38 | 0 | gate | 9.062 | 9.383 | 10.583 | 41.766 |
| 21 | 4.455 | 9.537 | 325.0 | 96 | 1 | gate | 5.526 | 9.531 | 21.569 | -56.782 |
| 22 | 4.158 | 9.388 | 312.5 | 88 | 1 | gate | 9.066 | 14.194 | 12.718 | 75.141 |
| 23 | 3.861 | 9.240 | 300.0 | 38 | 0 | gate | 8.914 | 9.234 | 9.902 | -50.013 |

## 判定用メモ

- adaptive curriculumの現在の成功条件は `passed_gates > 0`、すなわちfirst gate通過です。episode全体の無衝突成功を意味しません。
- `training_gate_index` がstateに存在する場合のみ、その値を明示します。0始まりです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
