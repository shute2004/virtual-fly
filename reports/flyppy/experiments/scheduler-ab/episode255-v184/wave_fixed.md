# Flyppy v3 fixed learning evaluation

- suite: `flyppy-v3-fixed-v1`
- subject: `trained_checkpoint`
- checkpoint: `artifacts/experiments/flyppy-v3-scheduler-ab3/wave/checkpoint`
- checkpoint neural step: 17354
- training episode end: 279
- global weight version: 208
- backend: `gpu-population:Apple M1 (Metal)`
- population / course seeds: 4 / 0..3
- vision: `direct-ray` / 13 rays/ommatidium
- plasticity: `false`
- global weight version unchanged: `true`
- transaction dirty edges after evaluation: {0: 0, 1: 0, 2: 0, 3: 0}
- elapsed: 145.164 s

## 固定条件別

| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 3.2670 | 11.4670 | 450.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.006 | 1.379 |
| midpoint | 1.6335 | 10.1885 | 375.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.005 | 1.549 |
| target | 0.0000 | 8.9100 | 300.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 3.061 |

## Motor output

| condition | wing spikes/step | somatic spikes/step | active wing units | active somatic units | power activation | L/R power diff | steering channels | abs leg drive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 2.3061 | 3.1286 | 33.372 | 28.031 | 0.9271 | 0.0008 | 5.878 | 0.0449 |
| midpoint | 2.4305 | 3.6625 | 32.789 | 25.448 | 0.9187 | 0.0014 | 6.270 | 0.0846 |
| target | 3.1385 | 4.4873 | 35.800 | 33.527 | 0.9220 | 0.0024 | 7.877 | 0.0992 |

## Episode一覧

| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| part3_frontier | 0 | 0 | 92 | 1 | gate | false | 23.763 | 8.735 | 11.461 | 469.554 |
| part3_frontier | 1 | 1 | 29 | 0 | gate | false | 9.844 | 11.270 | 11.461 | 77.733 |
| part3_frontier | 2 | 2 | 88 | 1 | gate | false | 22.749 | 9.137 | 11.461 | 29.924 |
| part3_frontier | 3 | 3 | 33 | 0 | gate | false | 10.572 | 11.207 | 11.461 | -18.945 |
| midpoint | 0 | 0 | 98 | 1 | gate | false | 21.002 | 5.944 | 10.183 | 24.756 |
| midpoint | 1 | 1 | 47 | 0 | gate | false | 10.629 | 9.707 | 10.183 | 43.233 |
| midpoint | 2 | 2 | 47 | 0 | gate | false | 10.560 | 9.680 | 10.183 | 11.994 |
| midpoint | 3 | 3 | 59 | 1 | gate | false | 13.002 | 9.228 | 10.183 | 474.288 |
| target | 0 | 0 | 66 | 0 | gate | false | 10.690 | 7.697 | 8.905 | -0.552 |
| target | 1 | 1 | 69 | 0 | gate | false | 11.386 | 7.283 | 8.905 | 329.434 |
| target | 2 | 2 | 65 | 0 | gate | false | 10.694 | 7.619 | 8.905 | 251.948 |
| target | 3 | 3 | 118 | 1 | floor | false | 19.718 | 0.796 | 8.905 | 227.486 |

## 解釈上の契約

この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。
gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。
各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。
curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。
