# Flyppy process-body parent overhead profile

- generated_at_utc: 2026-09-15T07:30:08+00:00
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- short-run episode count: 8
- short-run max control steps: 32
- long N=8 max control steps: 128

## Short-run scaling

| N | aggregate steps | reported steps/s | teardown s | steady steps/s | observe wait s | act wait s | CNS batch s | commit s | checkpoint save s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 17.269 | 0.213 | 17.521 | 7.844 | 0.983 | 3.829 | 0.006 | 0.303 |
| 2 | 256 | 23.915 | 0.449 | 24.962 | 6.208 | 0.488 | 2.095 | 0.006 | 0.277 |
| 4 | 256 | 25.031 | 0.931 | 27.539 | 6.160 | 0.287 | 1.328 | 0.007 | 0.233 |
| 8 | 256 | 16.237 | 1.840 | 18.382 | 9.565 | 0.288 | 1.289 | 0.008 | 0.295 |

## Long N=8 run

- aggregate control steps: 547
- reported throughput: 16.121 steps/s
- worker teardown: 2.159 s
- teardown-excluded throughput: 17.217 steps/s
- observe wait: 13.728 s
- act wait: 2.354 s
- CNS batch: 12.204 s
- reinforcement CNS: 0.024 s
- commit: 0.008 s
- checkpoint load: 0.267 s
- checkpoint save: 0.452 s

Interpretation: reported throughput includes worker teardown because the production-equivalent trainer closes body workers before computing elapsed_seconds. The steady figure subtracts only measured body-worker close time; all CNS, IPC, checkpoint and episode work remains included.
