# Flyppy v3 fixed learning evaluation

- suite: `flyppy-v3-fixed-v1`
- subject: `trained_checkpoint`
- checkpoint: `artifacts/experiments/flyppy-v3/checkpoint`
- checkpoint neural step: 14295
- training episode end: 231
- global weight version: 160
- backend: `gpu-population:Apple M1 (Metal)`
- population / course seeds: 4 / 0..3
- vision: `direct-ray` / 13 rays/ommatidium
- plasticity: `false`
- global weight version unchanged: `true`
- transaction dirty edges after evaluation: {0: 0, 1: 0, 2: 0, 3: 0}
- elapsed: 44.406 s

## 固定条件別

| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 3.2670 | 11.4670 | 450.0 | 2/4 (50.0%) | 1/4 (25.0%) | 4/4 (100.0%) | 0.750 | 2 | -0.006 | 1.373 |
| midpoint | 1.6335 | 10.1885 | 375.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.005 | 1.816 |
| target | 0.0000 | 8.9100 | 300.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 3.077 |

## Episode一覧

| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| part3_frontier | 0 | 0 | 93 | 2 | gate | false | 24.058 | 8.763 | 11.461 | 428.490 |
| part3_frontier | 1 | 1 | 29 | 0 | gate | false | 9.844 | 11.270 | 11.461 | 77.733 |
| part3_frontier | 2 | 2 | 88 | 1 | gate | false | 22.756 | 9.135 | 11.461 | 40.317 |
| part3_frontier | 3 | 3 | 33 | 0 | gate | false | 10.572 | 11.207 | 11.461 | -18.945 |
| midpoint | 0 | 0 | 105 | 1 | gate | false | 22.248 | 4.813 | 10.183 | -376.836 |
| midpoint | 1 | 1 | 47 | 0 | gate | false | 10.629 | 9.707 | 10.183 | 43.315 |
| midpoint | 2 | 2 | 47 | 0 | gate | false | 10.560 | 9.680 | 10.183 | 12.381 |
| midpoint | 3 | 3 | 59 | 1 | gate | false | 12.960 | 9.289 | 10.183 | 484.299 |
| target | 0 | 0 | 66 | 0 | gate | false | 10.689 | 7.695 | 8.905 | 0.452 |
| target | 1 | 1 | 69 | 0 | gate | false | 11.399 | 7.285 | 8.905 | 336.063 |
| target | 2 | 2 | 66 | 0 | gate | false | 10.756 | 7.563 | 8.905 | 130.743 |
| target | 3 | 3 | 118 | 1 | floor | false | 19.699 | 0.790 | 8.905 | 236.632 |

## 解釈上の契約

この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。
gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。
各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。
curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。
