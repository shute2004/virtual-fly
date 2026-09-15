# Flyppy packed production-trainer benchmark

- generated_at_utc: 2026-09-15T08:20:28+00:00
- overall: PASS
- populations: 4, 8, 12, 16
- episodes per case: 32
- max control steps per episode: 96
- packed body-process rule: min(population, 4)
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

| population | body processes | local steps/s | packed steps/s | packed/local | local elapsed s | packed elapsed s | checkpoint equal | episode equal | version/step equal |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 4 | 4 | 19.186013 | 26.049007 | 1.358x | 94.235 | 69.408 | True | True | True |
| 8 | 4 | 18.882369 | 24.907330 | 1.319x | 99.299 | 75.279 | True | True | True |
| 12 | 4 | 19.073744 | 23.073710 | 1.210x | 97.202 | 80.351 | True | True | True |
| 16 | 4 | 19.994878 | 24.417437 | 1.221x | 83.621 | 68.476 | True | True | True |

## Scaling

- packed N=16 / N=4 aggregate throughput: 0.937x

Packed cutover is acceptable only when every same-N case preserves checkpoint bytes, episode/commit outcomes, global weight version, and neural step.
