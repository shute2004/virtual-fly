# Flyppy shared-weight population latest run

- backend: `gpu-population:Apple M1 (Metal)`
- population: 4
- body runtime: `packed-process`
- body processes: 4
- launch mode: `async`
- vision runtime: `direct-ray`
- rays/ommatidium: 13
- RGB framebuffer: false
- shared weight: true
- weight averaging: false
- episodes: 24
- elapsed: 50.371 s
- aggregate control steps: 1326
- aggregate control steps/s: 26.325
- simulated biological seconds: 0.663000
- first gate pass: 12/24
- max passed gates: 2
- global weight version: 160 -> 184
- mean version staleness: 2.083
- max version staleness: 7
- commit semantics: episode-local additive+clamp transaction rebased onto latest global weight

| ep | slot | source v | commit from | commit v | stale | steps | passed | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 232 | 0 | 160 | 167 | 168 | 7 | 89 | 2 | true |
| 233 | 1 | 160 | 160 | 161 | 0 | 23 | 0 | true |
| 234 | 2 | 160 | 166 | 167 | 6 | 86 | 1 | true |
| 235 | 3 | 160 | 161 | 162 | 1 | 27 | 0 | true |
| 236 | 1 | 161 | 162 | 163 | 1 | 24 | 0 | true |
| 237 | 3 | 162 | 163 | 164 | 1 | 27 | 0 | true |
| 238 | 1 | 163 | 164 | 165 | 1 | 22 | 0 | true |
| 239 | 3 | 164 | 165 | 166 | 1 | 28 | 0 | true |
| 240 | 1 | 165 | 168 | 169 | 3 | 24 | 0 | true |
| 241 | 3 | 166 | 169 | 170 | 3 | 27 | 0 | true |
| 242 | 2 | 167 | 174 | 175 | 7 | 82 | 1 | true |
| 243 | 0 | 168 | 175 | 176 | 7 | 88 | 1 | true |
| 244 | 1 | 169 | 170 | 171 | 1 | 22 | 0 | true |
| 245 | 3 | 170 | 171 | 172 | 1 | 22 | 0 | true |
| 246 | 1 | 171 | 172 | 173 | 1 | 23 | 0 | true |
| 247 | 3 | 172 | 173 | 174 | 1 | 27 | 0 | true |
| 248 | 2 | 175 | 176 | 177 | 1 | 83 | 1 | true |
| 249 | 0 | 176 | 177 | 178 | 1 | 86 | 2 | true |
| 250 | 2 | 177 | 178 | 179 | 1 | 82 | 1 | true |
| 251 | 0 | 178 | 179 | 180 | 1 | 88 | 1 | true |
| 252 | 2 | 179 | 180 | 181 | 1 | 83 | 1 | true |
| 253 | 0 | 180 | 181 | 182 | 1 | 92 | 2 | true |
| 254 | 2 | 181 | 182 | 183 | 1 | 82 | 1 | true |
| 255 | 0 | 182 | 183 | 184 | 1 | 89 | 2 | true |
