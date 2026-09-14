# Flyppy latest run

## 集計

- backend: `gpu:Apple M1 (Metal)`
- curriculum mode: `adaptive`
- curriculum target gate: 1
- episode: 0〜3（4 episode）
- elapsed: 126.023 s（平均 31.506 s/episode）
- checkpoint neural step: 390
- target gate成功episode: 4/4（100.0%）
- target gate後にさらに1 gate以上通過: 0 episode
- stage内course完走: 0 episode
- 1 episode最大通過gate数: 1
- run内最大x: 21.037 mm
- 最も小さいspawn xでの成功: episode 3 / x=7.425 / z=10.205 / vx=362.5 / passed=1

## 最終カリキュラム状態

- curriculum_mode: adaptive
- spawn_x_mm: 7.150
- spawn_z_mm: 10.068
- initial_speed_mm_s: 350.000
- successful_first_gates: 4
- consecutive_failures: 0
- curriculum_episodes: 4
- target_x_mm: 0.000
- target_z_mm: 8.250
- target_speed_mm_s: 300.000
- curriculum_complete: false

## 成功境界

同一 `spawn_z / initial_vx` 条件ごとに、成功した中で最小のspawn xを記録します。

| spawn z (mm) | initial vx (mm/s) | 最小成功spawn x (mm) | episode | passed gates |
|---:|---:|---:|---:|---:|
| 10.205 | 362.5 | 7.425 | 3 | 1 |
| 10.343 | 375.0 | 7.700 | 2 | 1 |
| 10.480 | 387.5 | 7.975 | 1 | 1 |
| 10.618 | 400.0 | 8.250 | 0 | 1 |

## Collision

- gate: 4

## Episode一覧

| ep | spawn x | spawn z | vx | steps | passed | collision | max x | final vx |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 8.250 | 10.618 | 400.0 | 86 | 1 | gate | 21.037 | 16.398 |
| 1 | 7.975 | 10.480 | 387.5 | 87 | 1 | gate | 20.414 | 7.303 |
| 2 | 7.700 | 10.343 | 375.0 | 88 | 1 | gate | 19.793 | -25.043 |
| 3 | 7.425 | 10.205 | 362.5 | 97 | 1 | gate | 19.996 | -5.011 |

## 判定用メモ

- `training_gate_index` は0始まりです。0=第1gate、1=第2gateです。
- `boundary_ease_level` は0.00=難しい端、1.00=易しい端です。
- `latest.csv` がepisode単位の機械可読データです。
- checkpoint、trajectory、live telemetryなどの巨大/高頻度データは `artifacts/` に残し、Gitへは含めません。
- このレポートは最新runで上書きします。過去runはGit履歴から比較できます。
