# Flyppy v3 fixed learning evaluation

- suite: `flyppy-v3-fixed-v1`
- checkpoint: `artifacts/experiments/flyppy-v3/checkpoint`
- checkpoint neural step: 12149
- training episode end: 183
- global weight version: 112
- backend: `gpu-population:Apple M1 (Metal)`
- population / course seeds: 4 / 0..3
- vision: `direct-ray` / 13 rays/ommatidium
- plasticity: `false`
- global weight version unchanged: `true`
- transaction dirty edges after evaluation: {0: 0, 1: 0, 2: 0, 3: 0}
- elapsed: 39.621 s

## 固定条件別

| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 3.2670 | 11.4670 | 450.0 | 2/4 (50.0%) | 1/4 (25.0%) | 4/4 (100.0%) | 0.750 | 2 | -0.006 | 1.420 |
| midpoint | 1.6335 | 10.1885 | 375.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 0.657 |
| target | 0.0000 | 8.9100 | 300.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 3.073 |

## Episode一覧

| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| part3_frontier | 0 | 0 | 93 | 2 | gate | false | 24.076 | 8.519 | 11.461 | 418.477 |
| part3_frontier | 1 | 1 | 29 | 0 | gate | false | 9.844 | 11.270 | 11.461 | 77.733 |
| part3_frontier | 2 | 2 | 88 | 1 | gate | false | 22.759 | 9.190 | 11.461 | 68.963 |
| part3_frontier | 3 | 3 | 33 | 0 | gate | false | 10.572 | 11.207 | 11.461 | -18.945 |
| midpoint | 0 | 0 | 52 | 0 | gate | false | 11.662 | 9.491 | 10.183 | 358.854 |
| midpoint | 1 | 1 | 47 | 0 | gate | false | 10.632 | 9.710 | 10.183 | 37.608 |
| midpoint | 2 | 2 | 47 | 0 | gate | false | 10.567 | 9.681 | 10.183 | 38.183 |
| midpoint | 3 | 3 | 59 | 1 | gate | false | 13.034 | 9.244 | 10.183 | 424.534 |
| target | 0 | 0 | 65 | 0 | gate | false | 10.660 | 7.696 | 8.905 | 261.688 |
| target | 1 | 1 | 69 | 0 | gate | false | 11.388 | 7.286 | 8.905 | 328.778 |
| target | 2 | 2 | 66 | 0 | gate | false | 10.766 | 7.548 | 8.905 | 117.418 |
| target | 3 | 3 | 118 | 1 | floor | false | 19.801 | 0.820 | 8.905 | 214.891 |

## 解釈上の契約

この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。
gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。
各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。
curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。
