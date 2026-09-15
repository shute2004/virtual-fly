# Flyppy packed production-trainer benchmark

- generated_at_utc: 2026-09-15T08:00:26+00:00
- overall: PASS
- populations: 4, 8, 12
- episodes per case: 12
- max control steps per episode: 32
- packed body-process rule: min(population, 4)
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

| population | body processes | local steps/s | packed steps/s | packed/local | local elapsed s | packed elapsed s | checkpoint equal | episode equal | version/step equal |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 4 | 4 | 22.511389 | 29.145111 | 1.295x | 17.058 | 13.175 | True | True | True |
| 8 | 4 | 21.568806 | 28.200176 | 1.307x | 17.803 | 13.617 | True | True | True |
| 12 | 4 | 22.408839 | 25.786484 | 1.151x | 17.136 | 14.892 | True | True | True |

## Scaling

- packed N=12 / N=4 aggregate throughput: 0.885x

Packed cutover is acceptable only when every same-N case preserves checkpoint bytes, episode/commit outcomes, global weight version, and neural step.
