# Flyppy packed body-worker probe

- generated_at_utc: 2026-09-15T07:38:01+00:00
- overall: PASS
- population: 8
- max control steps: 64
- CNS: real shared GPU MaleCNS runtime
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- best process count: 2
- best steady throughput: 24.341 steps/s

| body processes | flies/process | aggregate steps | steady steps/s | observe s | CNS s | act s | startup s | checkpoint equal/outcomes equal |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.0 | 414 | 20.076 | 12.051 | 5.804 | 2.719 | 17.067 | True |
| 2 | 4.0 | 414 | 24.341 | 9.250 | 5.771 | 1.939 | 17.719 | True |
| 4 | 2.0 | 414 | 24.293 | 9.607 | 5.948 | 1.431 | 21.951 | True |
| 8 | 1.0 | 414 | 23.152 | 10.809 | 5.872 | 1.143 | 26.982 | True |

Interpretation: population remains fixed at eight independent flies. Only the number of Python/MuJoCo owner processes changes. This directly tests whether one-process-per-fly resource duplication is responsible for the N=8 regression seen with the shared CNS present.
