# Flyppy v3 fixed learning evaluation

- suite: `flyppy-v3-fixed-v1`
- subject: `trained_checkpoint`
- checkpoint: `artifacts/experiments/flyppy-v3-scheduler-ab3/async/checkpoint`
- checkpoint neural step: 17354
- training episode end: 279
- global weight version: 208
- backend: `gpu-population:Apple M1 (Metal)`
- population / course seeds: 4 / 0..3
- vision: `direct-ray` / 13 rays/ommatidium
- plasticity: `false`
- global weight version unchanged: `true`
- transaction dirty edges after evaluation: {0: 0, 1: 0, 2: 0, 3: 0}
- elapsed: 175.773 s

## 固定条件別

| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 3.2670 | 11.4670 | 450.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.006 | 1.357 |
| midpoint | 1.6335 | 10.1885 | 375.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.005 | 3.039 |
| target | 0.0000 | 8.9100 | 300.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 3.063 |

## Motor output

| condition | wing spikes/step | somatic spikes/step | active wing units | active somatic units | power activation | L/R power diff | steering channels | abs leg drive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 2.4340 | 3.2649 | 33.379 | 27.968 | 0.9274 | 0.0008 | 5.893 | 0.0413 |
| midpoint | 2.9149 | 4.3326 | 35.670 | 32.803 | 0.9268 | 0.0015 | 7.967 | 0.0913 |
| target | 3.0502 | 4.5006 | 35.852 | 32.830 | 0.9214 | 0.0020 | 7.927 | 0.1052 |

## Episode一覧

| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| part3_frontier | 0 | 0 | 92 | 1 | gate | false | 23.883 | 8.758 | 11.461 | 484.858 |
| part3_frontier | 1 | 1 | 29 | 0 | gate | false | 9.844 | 11.270 | 11.461 | 77.733 |
| part3_frontier | 2 | 2 | 88 | 1 | gate | false | 22.758 | 9.203 | 11.461 | 39.137 |
| part3_frontier | 3 | 3 | 33 | 0 | gate | false | 10.572 | 11.207 | 11.461 | -18.945 |
| midpoint | 0 | 0 | 98 | 1 | gate | false | 20.939 | 5.919 | 10.183 | 30.324 |
| midpoint | 1 | 1 | 47 | 0 | gate | false | 10.629 | 9.707 | 10.183 | 43.233 |
| midpoint | 2 | 2 | 47 | 0 | gate | false | 10.560 | 9.680 | 10.183 | 11.994 |
| midpoint | 3 | 3 | 100 | 1 | gate | false | 21.403 | 3.293 | 10.183 | 337.330 |
| target | 0 | 0 | 66 | 0 | gate | false | 10.688 | 7.696 | 8.905 | 0.404 |
| target | 1 | 1 | 69 | 0 | gate | false | 11.384 | 7.284 | 8.905 | 334.744 |
| target | 2 | 2 | 65 | 0 | gate | false | 10.696 | 7.618 | 8.905 | 255.422 |
| target | 3 | 3 | 118 | 1 | floor | false | 19.621 | 0.789 | 8.905 | 228.725 |

## 解釈上の契約

この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。
gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。
各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。
curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。
