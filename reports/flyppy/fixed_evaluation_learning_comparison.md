# Flyppy固定評価: 初期MaleCNSと学習個体の比較

## 対象

固定評価suite `flyppy-v3-fixed-v1` を用いて、同じ3条件・course seed 0..3で比較した。

- 初期MaleCNS: checkpointをloadせず、MaleCNS snapshot由来の初期weightを使用
- Part3継続個体 before: training episode 183終了、neural step 12149、global weight version 112
- Part4追加学習 after: training episode 207終了、neural step 13213、global weight version 136

全評価で:

- direct-ray K=13
- plasticity=false
- gate pass時PAM刺激 / collision時PPL刺激は維持
- commit/saveなし
- global weight version不変
- transaction dirty edge = 0

## 固定評価結果

| subject | condition | first gate | second gate | collision | max gates |
|---|---|---:|---:|---:|---:|
| 初期MaleCNS | part3_frontier | 2/4 | 1/4 | 4/4 | 2 |
| episode 183 | part3_frontier | 2/4 | 1/4 | 4/4 | 2 |
| episode 207 | part3_frontier | 2/4 | 0/4 | 4/4 | 1 |
| 初期MaleCNS | midpoint | 1/4 | 0/4 | 4/4 | 1 |
| episode 183 | midpoint | 1/4 | 0/4 | 4/4 | 1 |
| episode 207 | midpoint | 2/4 | 0/4 | 4/4 | 1 |
| 初期MaleCNS | target | 1/4 | 0/4 | 4/4 | 1 |
| episode 183 | target | 1/4 | 0/4 | 4/4 | 1 |
| episode 207 | target | 1/4 | 0/4 | 4/4 | 1 |

## 重要な観測

### 1. Part3終了付近まで、集計上のtask能力は初期MaleCNSと同じ

初期MaleCNSとepisode 183 checkpointは、3条件すべてでfirst/second gateの通過数が一致した。

ただしepisode単位の軌跡・最終速度・高度低下などは一致しない。したがって「weight更新が行動へ影響していない」という意味ではない。

### 2. episode 184..207の追加学習も一方向の改善ではない

追加24 episode後は:

- midpoint first gate: 1/4 -> 2/4
- target first gate: 1/4 -> 1/4
- part3_frontier first gate: 2/4 -> 2/4
- part3_frontier second gate: 1/4 -> 0/4

となった。

小標本ではあるが、少なくともこの24 episodeだけから「固定task能力が明確に改善した」とは判定できない。

### 3. plasticityそのものは大規模に進んでいる

`inspect_flyppy_learning.py` をepisode 207 checkpointへ適用した結果:

```text
weights changed_gt_1.0e-07 = 6,057,081
max_abs_weight_delta = 0.233326107
```

よって現在の主要問題は「plasticityが動いていない」ではない。

観測されている状態は:

```text
大規模なweight変化
  -> 行動軌跡は変化
  -> しかし固定task能力へ安定した改善として現れていない
```

である。

## 現在の学習run

Part4で追加したepisode 184..207は:

- 24 episode
- first gate pass episode: 6/24
- total passed gates: 7
- max passed gates: 2
- collision: 24/24
- global weight version: 112 -> 136
- neural step: 12149 -> 13213

adaptive curriculumはrun終了時:

```text
spawn_x_mm = 5.049
spawn_z_mm = 11.170332...
initial_speed_mm_s = 425
```

で、target `x=0 / z=8.91 / vx=300` から遠ざかっている。

## 次に検証する設計上の問題

現行shared-weight population trainerは、slotがepisodeを終えるたびに `record_adaptive_result()` を呼び、その直後に次episodeのspawn条件を決める。

非同期populationでは、短時間で失敗したslotが先に複数回curriculum stateを更新し、長時間生存・成功するslotは後から更新する。そのためcurriculum変化がepisode集合だけでなく**slotの完了順**にも依存する。

Part3では旧wing-only系で、この1 episode単位の難化/易化ping-pongを避けるため `boundary-band` が実装されていた。現行population trainerにはまだ接続されていない。

したがって次の候補は、neural plasticity則を変更する前に、population trainingのcurriculum更新をbatch単位へ戻し、非同期完了順への依存を除くことである。
