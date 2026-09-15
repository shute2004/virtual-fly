# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- adaptive training criterion: first gate pass (`passed_gates > 0`)
- episode: 48〜71（24 episode）
- elapsed: 251.466 s（平均 10.478 s/episode）
- checkpoint neural step: 5757
- first gate通過episode: 11/24（45.8%）
- first gate後にさらに1 gate以上通過: 0 episode
- first gate通過後に衝突: 11 episode
- 無衝突episode: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 23.833 mm
- spawn高度を一度でも上回ったepisode: 1/24
- run内最大高度gain: +1.432 mm
- run内最大高度loss: 7.478 mm
- episodeごとの最大高度gain平均: +0.054 mm
- 最も小さいspawn xでのfirst gate通過: episode 65 / x=1.634 / z=9.834 / vx=350.0 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 1.337
- spawn_z_mm: 9.982
- initial_speed_mm_s: 362.500
- successful_first_gates: 41
- consecutive_failures: 0
- curriculum_episodes: 72
- target_x_mm: 0.000
- target_z_mm: 8.910
- target_speed_mm_s: 300.000
- curriculum_complete: false

## First-gate成功境界

同一 `spawn_z / initial_vx` 条件ごとに、first gateを通過した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 9.834 | 350.0 | 1.634 | 65 | 1 |
| 9.982 | 362.5 | 1.931 | 64 | 1 |
| 10.131 | 375.0 | 1.634 | 71 | 1 |

## Collision

- gate: 24

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | min z | max z | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 48 | 2.673 | 9.685 | 337.5 | 47 | 0 | gate | 9.204 | 9.680 | 10.948 | 131.795 |
| 49 | 2.822 | 9.834 | 350.0 | 46 | 0 | gate | 9.368 | 9.828 | 11.197 | 320.891 |
| 50 | 2.970 | 9.982 | 362.5 | 102 | 1 | gate | 5.825 | 9.977 | 22.710 | 54.239 |
| 51 | 2.673 | 9.834 | 350.0 | 47 | 0 | gate | 9.358 | 9.828 | 11.216 | 207.128 |
| 52 | 2.822 | 9.982 | 362.5 | 102 | 1 | gate | 6.068 | 9.977 | 22.581 | -68.403 |
| 53 | 2.525 | 9.834 | 350.0 | 47 | 0 | gate | 9.316 | 9.828 | 11.058 | 173.481 |
| 54 | 2.673 | 9.982 | 362.5 | 102 | 1 | gate | 5.960 | 9.977 | 22.735 | 190.502 |
| 55 | 2.376 | 9.834 | 350.0 | 47 | 0 | gate | 9.356 | 9.828 | 10.925 | 226.916 |
| 56 | 2.525 | 9.982 | 362.5 | 51 | 1 | gate | 9.305 | 9.977 | 12.098 | 370.493 |
| 57 | 2.228 | 9.834 | 350.0 | 47 | 0 | gate | 9.358 | 9.828 | 10.770 | 195.393 |
| 58 | 2.376 | 9.982 | 362.5 | 104 | 1 | gate | 5.523 | 9.977 | 22.898 | 213.769 |
| 59 | 2.079 | 9.834 | 350.0 | 47 | 0 | gate | 9.349 | 9.828 | 10.633 | 324.153 |
| 60 | 2.228 | 9.982 | 362.5 | 105 | 1 | gate | 5.153 | 9.977 | 22.658 | -191.791 |
| 61 | 1.931 | 9.834 | 350.0 | 47 | 0 | gate | 9.317 | 9.828 | 10.462 | 111.916 |
| 62 | 2.079 | 9.982 | 362.5 | 104 | 1 | gate | 5.451 | 9.977 | 22.599 | 327.882 |
| 63 | 1.782 | 9.834 | 350.0 | 47 | 0 | gate | 9.351 | 9.828 | 10.332 | 308.991 |
| 64 | 1.931 | 9.982 | 362.5 | 105 | 1 | gate | 5.238 | 9.977 | 22.425 | -253.713 |
| 65 | 1.634 | 9.834 | 350.0 | 179 | 1 | gate | 2.356 | 11.266 | 23.833 | 199.746 |
| 66 | 1.337 | 9.685 | 337.5 | 54 | 0 | gate | 9.003 | 9.680 | 10.925 | 311.153 |
| 67 | 1.485 | 9.834 | 350.0 | 55 | 0 | gate | 9.239 | 9.828 | 11.346 | 258.916 |
| 68 | 1.634 | 9.982 | 362.5 | 52 | 0 | gate | 9.355 | 9.977 | 11.393 | 331.941 |
| 69 | 1.782 | 10.131 | 375.0 | 105 | 1 | gate | 5.545 | 10.125 | 22.761 | -63.623 |
| 70 | 1.485 | 9.982 | 362.5 | 56 | 0 | gate | 9.286 | 9.977 | 11.996 | 337.693 |
| 71 | 1.634 | 10.131 | 375.0 | 104 | 1 | gate | 6.173 | 10.125 | 22.499 | 395.801 |

## 判定用メモ

- adaptive curriculumの現在の成功条件は `passed_gates > 0`、すなわちfirst gate通過です。episode全体の無衝突成功を意味しません。
- `training_gate_index` がstateに存在する場合のみ、その値を明示します。0始まりです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
