# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- adaptive training criterion: first gate pass (`passed_gates > 0`)
- episode: 24〜47（24 episode）
- elapsed: 202.136 s（平均 8.422 s/episode）
- checkpoint neural step: 3815
- first gate通過episode: 11/24（45.8%）
- first gate後にさらに1 gate以上通過: 1 episode
- first gate通過後に衝突: 11 episode
- 無衝突episode: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 2
- run内最大x: 28.077 mm
- spawn高度を一度でも上回ったepisode: 1/24
- run内最大高度gain: +2.496 mm
- run内最大高度loss: 7.720 mm
- episodeごとの最大高度gain平均: +0.099 mm
- 最も小さいspawn xでのfirst gate通過: episode 47 / x=2.970 / z=9.834 / vx=350.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 2.673
- spawn_z_mm: 9.685
- initial_speed_mm_s: 337.500
- successful_first_gates: 30
- consecutive_failures: 0
- curriculum_episodes: 48
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## First-gate成功境界

同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 9.537 | 325.0 | 4.010 | 27 | 2 |
| 9.685 | 337.5 | 3.861 | 32 | 1 |
| 9.834 | 350.0 | 2.970 | 47 | 1 |

## Collision

- gate: 23
- floor: 1

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 24 | 4.010 | 9.388 | 312.5 | 38 | 0 | gate | 9.064 | 9.383 | 10.237 | -12.077 |
| 25 | 4.158 | 9.537 | 325.0 | 44 | 0 | gate | 9.052 | 9.531 | 11.707 | 284.272 |
| 26 | 4.307 | 9.685 | 337.5 | 96 | 1 | gate | 5.878 | 9.680 | 21.824 | -0.219 |
| 27 | 4.010 | 9.537 | 325.0 | 184 | 2 | floor | 1.817 | 12.033 | 28.077 | 110.404 |
| 28 | 3.713 | 9.388 | 312.5 | 42 | 0 | gate | 8.939 | 9.383 | 10.719 | 278.058 |
| 29 | 3.861 | 9.537 | 325.0 | 44 | 0 | gate | 9.052 | 9.531 | 11.409 | 283.896 |
| 30 | 4.010 | 9.685 | 337.5 | 105 | 1 | gate | 3.927 | 9.680 | 21.588 | 1.788 |
| 31 | 3.713 | 9.537 | 325.0 | 42 | 0 | gate | 9.185 | 9.531 | 10.900 | 268.188 |
| 32 | 3.861 | 9.685 | 337.5 | 51 | 1 | gate | 9.146 | 9.680 | 12.854 | 320.015 |
| 33 | 3.564 | 9.537 | 325.0 | 41 | 0 | gate | 9.145 | 9.531 | 10.631 | 299.718 |
| 34 | 3.713 | 9.685 | 337.5 | 45 | 0 | gate | 9.207 | 9.680 | 11.686 | 323.537 |
| 35 | 3.861 | 9.834 | 350.0 | 50 | 1 | gate | 9.262 | 9.828 | 12.984 | 359.434 |
| 36 | 3.564 | 9.685 | 337.5 | 45 | 0 | gate | 9.207 | 9.680 | 11.538 | 323.465 |
| 37 | 3.713 | 9.834 | 350.0 | 50 | 1 | gate | 9.249 | 9.828 | 12.841 | 370.843 |
| 38 | 3.416 | 9.685 | 337.5 | 47 | 0 | gate | 9.206 | 9.680 | 11.712 | 261.120 |
| 39 | 3.564 | 9.834 | 350.0 | 50 | 1 | gate | 9.261 | 9.828 | 12.686 | 378.923 |
| 40 | 3.267 | 9.685 | 337.5 | 47 | 0 | gate | 9.206 | 9.680 | 11.552 | 203.725 |
| 41 | 3.416 | 9.834 | 350.0 | 51 | 1 | gate | 9.220 | 9.828 | 12.733 | 365.752 |
| 42 | 3.119 | 9.685 | 337.5 | 46 | 0 | gate | 9.221 | 9.680 | 11.251 | 313.000 |
| 43 | 3.267 | 9.834 | 350.0 | 97 | 1 | gate | 5.456 | 9.828 | 20.830 | -125.350 |
| 44 | 2.970 | 9.685 | 337.5 | 46 | 0 | gate | 9.166 | 9.680 | 11.110 | 327.843 |
| 45 | 3.119 | 9.834 | 350.0 | 54 | 1 | gate | 9.139 | 9.828 | 12.992 | 341.763 |
| 46 | 2.822 | 9.685 | 337.5 | 47 | 0 | gate | 9.204 | 9.680 | 11.093 | 115.259 |
| 47 | 2.970 | 9.834 | 350.0 | 55 | 1 | gate | 9.138 | 9.828 | 13.031 | 356.363 |

## 判定用メモ

- adaptive curriculumの現在の成功条件は `passed_gates > 0`、すなわちfirst gate通過です。episode全体の無衝突成功を意味しません。
- `training_gate_index` がstateに存在する場合のみ、その値を明示します。0始まりです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
