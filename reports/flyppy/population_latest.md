# Flyppy shared-weight population latest run

- backend: `gpu-population:Apple M1 (Metal)`
- population: 4
- body runtime: `packed-process`
- body processes: 4
- vision runtime: `direct-ray`
- rays/ommatidium: 13
- RGB framebuffer: false
- shared weight: true
- weight averaging: false
- episodes: 8
- elapsed: 12.209 s
- aggregate control steps: 368
- aggregate control steps/s: 30.141
- simulated biological seconds: 0.184000
- first gate pass: 2/8
- max passed gates: 1
- global weight version: 80 -> 88
- mean version staleness: 2.250
- max version staleness: 6
- commit semantics: episode-local additive+clamp transaction rebased onto latest global weight

| ep | slot | source v | commit from | commit v | stale | steps | passed | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 152 | 0 | 80 | 86 | 87 | 6 | 93 | 1 | true |
| 153 | 1 | 80 | 80 | 81 | 0 | 32 | 0 | true |
| 154 | 2 | 80 | 84 | 85 | 4 | 89 | 1 | true |
| 155 | 3 | 80 | 81 | 82 | 1 | 33 | 0 | true |
| 156 | 1 | 81 | 82 | 83 | 1 | 31 | 0 | true |
| 157 | 3 | 82 | 83 | 84 | 1 | 32 | 0 | true |
| 158 | 1 | 83 | 85 | 86 | 2 | 29 | 0 | true |
| 159 | 3 | 84 | 87 | 88 | 3 | 29 | 0 | true |
