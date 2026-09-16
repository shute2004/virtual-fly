# Flyppy v3 fixed learning evaluation

- suite: `flyppy-v3-fixed-v1`
- subject: `trained_checkpoint`
- checkpoint: `artifacts/experiments/flyppy-v3/checkpoint`
- checkpoint neural step: 15781
- training episode end: 255
- global weight version: 184
- backend: `gpu-population:Apple M1 (Metal)`
- population / course seeds: 4 / 0..3
- vision: `direct-ray` / 13 rays/ommatidium
- plasticity: `false`
- global weight version unchanged: `true`
- transaction dirty edges after evaluation: {0: 0, 1: 0, 2: 0, 3: 0}
- elapsed: 129.095 s

## 固定条件別

| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 3.2670 | 11.4670 | 450.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.006 | 1.356 |
| midpoint | 1.6335 | 10.1885 | 375.0 | 2/4 (50.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.500 | 1 | -0.005 | 1.796 |
| target | 0.0000 | 8.9100 | 300.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 3.040 |

## Motor output

| condition | wing spikes/step | somatic spikes/step | active wing units | active somatic units | power activation | L/R power diff | steering channels | abs leg drive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 2.1806 | 3.2919 | 33.464 | 28.578 | 0.9261 | 0.0008 | 5.972 | 0.0478 |
| midpoint | 2.3460 | 3.7114 | 33.159 | 26.018 | 0.9191 | 0.0017 | 6.563 | 0.0985 |
| target | 3.0042 | 4.4616 | 35.735 | 33.629 | 0.9203 | 0.0034 | 7.806 | 0.1026 |

## Episode一覧

| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| part3_frontier | 0 | 0 | 92 | 1 | gate | false | 23.793 | 8.728 | 11.461 | 469.752 |
| part3_frontier | 1 | 1 | 29 | 0 | gate | false | 9.844 | 11.270 | 11.461 | 77.733 |
| part3_frontier | 2 | 2 | 88 | 1 | gate | false | 22.770 | 9.237 | 11.461 | 58.755 |
| part3_frontier | 3 | 3 | 33 | 0 | gate | false | 10.572 | 11.207 | 11.461 | -18.945 |
| midpoint | 0 | 0 | 105 | 1 | gate | false | 22.307 | 4.896 | 10.183 | -395.412 |
| midpoint | 1 | 1 | 47 | 0 | gate | false | 10.629 | 9.707 | 10.183 | 43.080 |
| midpoint | 2 | 2 | 47 | 0 | gate | false | 10.560 | 9.680 | 10.183 | 12.471 |
| midpoint | 3 | 3 | 59 | 1 | gate | false | 12.957 | 9.289 | 10.183 | 436.850 |
| target | 0 | 0 | 66 | 0 | gate | false | 10.689 | 7.692 | 8.905 | -3.861 |
| target | 1 | 1 | 69 | 0 | gate | false | 11.387 | 7.273 | 8.905 | 328.482 |
| target | 2 | 2 | 65 | 0 | gate | false | 10.684 | 7.619 | 8.905 | 258.490 |
| target | 3 | 3 | 118 | 1 | floor | false | 19.706 | 0.897 | 8.905 | 250.867 |

## 解釈上の契約

この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。
gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。
各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。
curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。
