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
- episodes: 24
- elapsed: 31.016 s
- aggregate control steps: 940
- aggregate control steps/s: 30.307
- simulated biological seconds: 0.470000
- first gate pass: 6/24
- max passed gates: 2
- global weight version: 112 -> 136
- mean version staleness: 2.750
- max version staleness: 9
- commit semantics: episode-local additive+clamp transaction rebased onto latest global weight

| ep | slot | source v | commit from | commit v | stale | steps | passed | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 184 | 0 | 112 | 119 | 120 | 7 | 92 | 1 | true |
| 185 | 1 | 112 | 112 | 113 | 0 | 28 | 0 | true |
| 186 | 2 | 112 | 118 | 119 | 6 | 87 | 1 | true |
| 187 | 3 | 112 | 113 | 114 | 1 | 28 | 0 | true |
| 188 | 1 | 113 | 115 | 116 | 2 | 26 | 0 | true |
| 189 | 3 | 114 | 114 | 115 | 0 | 25 | 0 | true |
| 190 | 3 | 115 | 117 | 118 | 2 | 24 | 0 | true |
| 191 | 1 | 116 | 116 | 117 | 0 | 23 | 0 | true |
| 192 | 1 | 117 | 121 | 122 | 4 | 23 | 0 | true |
| 193 | 3 | 118 | 120 | 121 | 2 | 22 | 0 | true |
| 194 | 2 | 119 | 128 | 129 | 9 | 83 | 1 | true |
| 195 | 0 | 120 | 129 | 130 | 9 | 91 | 1 | true |
| 196 | 3 | 121 | 123 | 124 | 2 | 27 | 0 | true |
| 197 | 1 | 122 | 122 | 123 | 0 | 23 | 0 | true |
| 198 | 1 | 123 | 124 | 125 | 1 | 23 | 0 | true |
| 199 | 3 | 124 | 125 | 126 | 1 | 22 | 0 | true |
| 200 | 1 | 125 | 126 | 127 | 1 | 20 | 0 | true |
| 201 | 3 | 126 | 127 | 128 | 1 | 20 | 0 | true |
| 202 | 1 | 127 | 130 | 131 | 3 | 20 | 0 | true |
| 203 | 3 | 128 | 131 | 132 | 3 | 19 | 0 | true |
| 204 | 2 | 129 | 134 | 135 | 5 | 80 | 1 | true |
| 205 | 0 | 130 | 135 | 136 | 5 | 92 | 2 | true |
| 206 | 1 | 131 | 133 | 134 | 2 | 22 | 0 | true |
| 207 | 3 | 132 | 132 | 133 | 0 | 20 | 0 | true |
