# Flyppy shared-weight population latest run

- backend: `gpu-population:Apple M1 (Metal)`
- population: 2
- shared weight: true
- weight averaging: false
- episodes: 24
- elapsed: 200.746 s
- aggregate control steps: 1507
- aggregate control steps/s: 7.507
- simulated biological seconds: 0.753500
- first gate pass: 7/24
- max passed gates: 1
- global weight version: 0 -> 24
- mean version staleness: 0.958
- max version staleness: 3
- commit semantics: episode-local additive+clamp transaction rebased onto latest global weight

| ep | slot | source v | commit from | commit v | stale | steps | passed | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 72 | 0 | 0 | 0 | 1 | 0 | 52 | 0 | true |
| 73 | 1 | 0 | 1 | 2 | 1 | 59 | 1 | true |
| 74 | 0 | 1 | 3 | 4 | 2 | 105 | 1 | true |
| 75 | 1 | 2 | 2 | 3 | 0 | 60 | 1 | true |
| 76 | 1 | 3 | 5 | 6 | 2 | 119 | 1 | true |
| 77 | 0 | 4 | 4 | 5 | 0 | 60 | 0 | true |
| 78 | 0 | 5 | 6 | 7 | 1 | 56 | 0 | true |
| 79 | 1 | 6 | 7 | 8 | 1 | 62 | 0 | true |
| 80 | 0 | 7 | 8 | 9 | 1 | 56 | 0 | true |
| 81 | 1 | 8 | 9 | 10 | 1 | 60 | 0 | true |
| 82 | 0 | 9 | 10 | 11 | 1 | 52 | 0 | true |
| 83 | 1 | 10 | 11 | 12 | 1 | 49 | 0 | true |
| 84 | 0 | 11 | 14 | 15 | 3 | 157 | 1 | true |
| 85 | 1 | 12 | 12 | 13 | 0 | 45 | 0 | true |
| 86 | 1 | 13 | 13 | 14 | 0 | 43 | 0 | true |
| 87 | 1 | 14 | 15 | 16 | 1 | 41 | 0 | true |
| 88 | 0 | 15 | 18 | 19 | 3 | 100 | 1 | true |
| 89 | 1 | 16 | 16 | 17 | 0 | 42 | 0 | true |
| 90 | 1 | 17 | 17 | 18 | 0 | 38 | 0 | true |
| 91 | 1 | 18 | 19 | 20 | 1 | 38 | 0 | true |
| 92 | 0 | 19 | 22 | 23 | 3 | 100 | 1 | true |
| 93 | 1 | 20 | 20 | 21 | 0 | 38 | 0 | true |
| 94 | 1 | 21 | 21 | 22 | 0 | 38 | 0 | true |
| 95 | 1 | 22 | 23 | 24 | 1 | 37 | 0 | true |
