# Flyppy v3 fixed learning evaluation

- suite: `flyppy-v3-fixed-v1`
- subject: `initial_malecns`
- checkpoint: `initial MaleCNS snapshot weights`
- checkpoint neural step: 0
- training episode end: None
- global weight version: 0
- backend: `gpu-population:Apple M1 (Metal)`
- population / course seeds: 4 / 0..3
- vision: `direct-ray` / 13 rays/ommatidium
- plasticity: `false`
- global weight version unchanged: `true`
- transaction dirty edges after evaluation: {0: 0, 1: 0, 2: 0, 3: 0}
- elapsed: 120.607 s

## 固定条件別

| condition | x mm | z mm | vx mm/s | first gate | second gate | collision | mean gates | max gates | mean altitude gain mm | mean altitude loss mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 3.2670 | 11.4670 | 450.0 | 2/4 (50.0%) | 1/4 (25.0%) | 4/4 (100.0%) | 0.750 | 2 | -0.006 | 1.452 |
| midpoint | 1.6335 | 10.1885 | 375.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 0.692 |
| target | 0.0000 | 8.9100 | 300.0 | 1/4 (25.0%) | 0/4 (0.0%) | 4/4 (100.0%) | 0.250 | 1 | -0.005 | 3.146 |

## Motor output

| condition | wing spikes/step | somatic spikes/step | active wing units | active somatic units | power activation | L/R power diff | steering channels | abs leg drive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| part3_frontier | 2.2401 | 3.2900 | 33.580 | 30.200 | 0.9263 | 0.0012 | 5.940 | 0.0452 |
| midpoint | 1.5951 | 3.1766 | 30.254 | 18.181 | 0.9078 | 0.0011 | 4.482 | 0.1028 |
| target | 2.8145 | 4.7720 | 36.310 | 36.332 | 0.9179 | 0.0015 | 8.205 | 0.1091 |

## Episode一覧

| condition | seed | slot | steps | passed | collision | finished | max x mm | min z mm | max z mm | final vx mm/s |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| part3_frontier | 0 | 0 | 93 | 2 | gate | false | 24.016 | 8.673 | 11.461 | 391.871 |
| part3_frontier | 1 | 1 | 29 | 0 | gate | false | 9.844 | 11.270 | 11.461 | 77.733 |
| part3_frontier | 2 | 2 | 88 | 1 | gate | false | 22.708 | 8.908 | 11.461 | 109.892 |
| part3_frontier | 3 | 3 | 33 | 0 | gate | false | 10.572 | 11.207 | 11.461 | -18.945 |
| midpoint | 0 | 0 | 54 | 0 | gate | false | 11.964 | 9.426 | 10.183 | 351.418 |
| midpoint | 1 | 1 | 48 | 0 | gate | false | 10.684 | 9.625 | 10.183 | 67.870 |
| midpoint | 2 | 2 | 47 | 0 | gate | false | 10.547 | 9.680 | 10.183 | -3.431 |
| midpoint | 3 | 3 | 59 | 1 | gate | false | 12.919 | 9.254 | 10.183 | 420.077 |
| target | 0 | 0 | 66 | 0 | gate | false | 10.685 | 7.568 | 8.905 | -15.312 |
| target | 1 | 1 | 70 | 0 | gate | false | 11.429 | 7.223 | 8.905 | 306.920 |
| target | 2 | 2 | 66 | 0 | gate | false | 10.719 | 7.426 | 8.905 | 124.336 |
| target | 3 | 3 | 117 | 1 | floor | false | 19.428 | 0.839 | 8.905 | 205.336 |

## 解釈上の契約

この評価はcheckpointを読み取り専用でロードし、全neural stepを `plasticity=false` で実行する。
gate pass時のPAM刺激とcollision時のPPL刺激は通常taskと同じく与えるが、weight更新は行わない。
各条件・各seedの開始前にslot neural stateをrestartし、body/periphery/retinal adaptationもresetする。
curriculum state、trajectory、checkpointは書き込まず、global weight versionも更新しない。
