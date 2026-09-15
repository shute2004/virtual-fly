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
- episodes: 48
- elapsed: 77.510 s
- aggregate control steps: 2506
- aggregate control steps/s: 32.331
- simulated biological seconds: 1.253000
- first gate pass: 14/48
- max passed gates: 1
- global weight version: 24 -> 72
- mean version staleness: 2.875
- max version staleness: 7
- commit semantics: episode-local additive+clamp transaction rebased onto latest global weight

| ep | slot | source v | commit from | commit v | stale | steps | passed | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 96 | 0 | 24 | 29 | 30 | 5 | 96 | 1 | true |
| 97 | 1 | 24 | 24 | 25 | 0 | 38 | 0 | true |
| 98 | 2 | 24 | 28 | 29 | 4 | 95 | 1 | true |
| 99 | 3 | 24 | 25 | 26 | 1 | 40 | 0 | true |
| 100 | 1 | 25 | 26 | 27 | 1 | 37 | 0 | true |
| 101 | 3 | 26 | 27 | 28 | 1 | 37 | 0 | true |
| 102 | 1 | 27 | 30 | 31 | 3 | 36 | 0 | true |
| 103 | 3 | 28 | 31 | 32 | 3 | 37 | 0 | true |
| 104 | 2 | 29 | 36 | 37 | 7 | 95 | 1 | true |
| 105 | 0 | 30 | 37 | 38 | 7 | 99 | 1 | true |
| 106 | 1 | 31 | 32 | 33 | 1 | 38 | 0 | true |
| 107 | 3 | 32 | 33 | 34 | 1 | 37 | 0 | true |
| 108 | 1 | 33 | 34 | 35 | 1 | 36 | 0 | true |
| 109 | 3 | 34 | 35 | 36 | 1 | 37 | 0 | true |
| 110 | 1 | 35 | 38 | 39 | 3 | 34 | 0 | true |
| 111 | 3 | 36 | 39 | 40 | 3 | 35 | 0 | true |
| 112 | 2 | 37 | 42 | 43 | 5 | 94 | 1 | true |
| 113 | 0 | 38 | 44 | 45 | 6 | 98 | 1 | true |
| 114 | 1 | 39 | 40 | 41 | 1 | 37 | 0 | true |
| 115 | 3 | 40 | 41 | 42 | 1 | 37 | 0 | true |
| 116 | 1 | 41 | 43 | 44 | 2 | 34 | 0 | true |
| 117 | 3 | 42 | 45 | 46 | 3 | 35 | 0 | true |
| 118 | 2 | 43 | 50 | 51 | 7 | 93 | 1 | true |
| 119 | 1 | 44 | 46 | 47 | 2 | 34 | 0 | true |
| 120 | 0 | 45 | 51 | 52 | 6 | 96 | 1 | true |
| 121 | 3 | 46 | 47 | 48 | 1 | 37 | 0 | true |
| 122 | 1 | 47 | 48 | 49 | 1 | 34 | 0 | true |
| 123 | 3 | 48 | 49 | 50 | 1 | 35 | 0 | true |
| 124 | 1 | 49 | 52 | 53 | 3 | 33 | 0 | true |
| 125 | 3 | 50 | 53 | 54 | 3 | 33 | 0 | true |
| 126 | 2 | 51 | 57 | 58 | 6 | 91 | 1 | true |
| 127 | 0 | 52 | 59 | 60 | 7 | 97 | 1 | true |
| 128 | 1 | 53 | 54 | 55 | 1 | 35 | 0 | true |
| 129 | 3 | 54 | 55 | 56 | 1 | 35 | 0 | true |
| 130 | 1 | 55 | 56 | 57 | 1 | 33 | 0 | true |
| 131 | 3 | 56 | 58 | 59 | 2 | 33 | 0 | true |
| 132 | 1 | 57 | 60 | 61 | 3 | 31 | 0 | true |
| 133 | 2 | 58 | 65 | 66 | 7 | 91 | 1 | true |
| 134 | 3 | 59 | 61 | 62 | 2 | 33 | 0 | true |
| 135 | 0 | 60 | 67 | 68 | 7 | 97 | 1 | true |
| 136 | 1 | 61 | 62 | 63 | 1 | 33 | 0 | true |
| 137 | 3 | 62 | 63 | 64 | 1 | 33 | 0 | true |
| 138 | 1 | 63 | 64 | 65 | 1 | 31 | 0 | true |
| 139 | 3 | 64 | 66 | 67 | 2 | 32 | 0 | true |
| 140 | 1 | 65 | 68 | 69 | 3 | 29 | 0 | true |
| 141 | 2 | 66 | 70 | 71 | 4 | 90 | 1 | true |
| 142 | 3 | 67 | 69 | 70 | 2 | 32 | 0 | true |
| 143 | 0 | 68 | 71 | 72 | 3 | 93 | 1 | true |
