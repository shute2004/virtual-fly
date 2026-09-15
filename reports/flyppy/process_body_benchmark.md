# Flyppy process-body production trainer benchmark

- generated_at_utc: 2026-09-15T07:02:26+00:00
- overall: PASS
- episodes per case: 8
- max control steps per episode: 32
- curriculum movement during benchmark: effectively frozen
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

| population | local steps/s | process steps/s | process/local | local elapsed s | process elapsed s | checkpoint equal | episode equal | version/step equal |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 18.408576 | 18.939651 | 1.029x | 13.907 | 13.517 | True | True | True |
| 2 | 20.958879 | 24.143688 | 1.152x | 12.214 | 10.603 | True | True | True |
| 4 | 21.207881 | 23.378501 | 1.102x | 12.071 | 10.950 | True | True | True |
| 8 | 20.624342 | 16.403842 | 0.795x | 12.413 | 15.606 | True | True | True |

## Scaling

- local N=8 / N=1 aggregate throughput: 1.120x
- process-body N=8 / N=1 aggregate throughput: 0.866x
- process-body N=8 vs local N=8: 0.795x

The process-body trainer is acceptable for production cutover only if every same-N local/process case preserves checkpoint bytes and episode/commit outcomes.
