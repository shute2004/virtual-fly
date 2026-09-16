# Flyppy motor-path weight reach

- checkpoint: `artifacts/experiments/flyppy-v3/checkpoint`
- checkpoint neural step: 15781
- weight epsilon: 1.0e-07
- synapse scale: 0.005
- max upstream hops: 4

## Direct actuated motor input

| target | neurons | neurons with plastic incoming | plastic incoming edges | incoming edges | changed | changed % | mean | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wing | 60 | 0 | 0 | 32830 | 0 | 0.000% | 0 | 0 |
| somatic | 208 | 0 | 0 | 47558 | 0 | 0.000% | 0 | 0 |
| all_actuated | 268 | 0 | 0 | 80388 | 0 | 0.000% | 0 | 0 |

## Released-connectome upstream distance

距離0はactuated motor neuron自身。各行のweight統計は、その距離にあるpostsynaptic neuronへ入るedgeを集計する。

| post distance to motor | neurons | incoming edges | changed | changed % | positive | negative | mean abs delta | max abs delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 268 | 80388 | 0 | 0.000% | 0 | 0 | 0 | 0 |
| 1 | 11807 | 2744393 | 248275 | 9.047% | 94913 | 153362 | 0.000313716 | 0.0122701 |
| 2 | 38148 | 9575682 | 3779855 | 39.473% | 1579109 | 2200746 | 0.000478598 | 0.278624 |
| 3 | 105401 | 12924189 | 2050609 | 15.866% | 867427 | 1183182 | 0.000154384 | 0.177435 |
| 4 | 9733 | 220278 | 24 | 0.011% | 5 | 19 | 3.84229e-07 | 9.12696e-07 |

## Global reference

- all edges: 25582938
- changed edges: 6078763 (23.761%)
- positive / negative changed edges: 2541454 / 3537309
- mean abs delta among changed: 0.000362491
- max abs delta: 0.278624

## Interpretation boundary

この解析はreleased MaleCNS connectivity上の経路距離とcheckpoint weight差だけを測る。
距離が近いことは、そのedgeが実際のepisodeでmotor activityを因果的に変えたことを意味しない。
逆にdirect motor incomingが変化していなくても、上流回路の可塑性がmotor出力を変えることは可能である。
