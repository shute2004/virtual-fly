# Flyppy packed population scaling probe

- generated_at_utc: 2026-09-15T07:52:32+00:00
- overall: PASS
- populations: 4, 8, 12
- max control steps per slot: 48
- CNS: real shared GPU MaleCNS runtime
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

| population | body processes | flies/process | steady steps/s | observe s | CNS s | act s | startup s | parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1 | 4.00 | 18.614 | 6.407 | 2.077 | 0.844 | 8.960 | True |
| 4 | 2 | 2.00 | 21.381 | 5.705 | 1.824 | 0.589 | 10.465 | True |
| 4 | 4 | 1.00 | 23.819 | 5.107 | 1.786 | 0.391 | 13.071 | True |
| 8 | 1 | 8.00 | 21.979 | 10.723 | 3.544 | 1.616 | 16.143 | True |
| 8 | 2 | 4.00 | 27.181 | 8.234 | 3.494 | 1.103 | 17.573 | True |
| 8 | 4 | 2.00 | 28.389 | 7.998 | 3.550 | 0.736 | 21.916 | True |
| 8 | 8 | 1.00 | 27.259 | 8.863 | 3.341 | 0.588 | 26.338 | True |
| 12 | 1 | 12.00 | 22.245 | 15.690 | 5.610 | 2.634 | 24.210 | True |
| 12 | 2 | 6.00 | 28.677 | 11.079 | 5.918 | 1.553 | 25.049 | True |
| 12 | 4 | 3.00 | 29.685 | 11.062 | 5.707 | 1.145 | 28.741 | True |
| 12 | 8 | 1.50 | 29.516 | 11.678 | 5.315 | 1.018 | 33.915 | True |

## Best packing by population

- N=4: 4 body processes, 1.00 flies/process, 23.819 steps/s
- N=8: 4 body processes, 2.00 flies/process, 28.389 steps/s
- N=12: 4 body processes, 3.00 flies/process, 29.685 steps/s

## Best-packing scaling

- best N=12 / best N=4 aggregate throughput: 1.246x

Interpretation: population count and body-process count are independent variables. Each population is compared only across physically distinct process counts, and checkpoint/outcome parity is required within that population.
