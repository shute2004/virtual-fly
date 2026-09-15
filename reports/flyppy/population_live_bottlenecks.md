# Flyppy eager-live population bottleneck profile

- generated_at_utc: 2026-09-15T05:45:18+00:00
- overall: PASS
- episodes per case: 8
- max control steps per episode: 32
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## End-to-end

| population | control steps | trainer elapsed s | aggregate steps/s | outer wall s |
|---:|---:|---:|---:|---:|
| 1 | 256 | 13.800 | 18.550484 | 16.434 |
| 8 | 256 | 10.950 | 23.377994 | 25.538 |

## Timed boundaries

Times are inclusive wall time at Python/CNS boundaries. Percentages use trainer elapsed and are diagnostic; small uninstrumented Python/course/I/O work appears as remainder.

| population | stage | seconds | calls | ms/call | ms/control-step | % trainer elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | retina | 7.862 | 256 | 30.712 | 30.712 | 56.97% |
| 1 | brain_batch | 3.654 | 256 | 14.275 | 14.275 | 26.48% |
| 1 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 1 | periphery | 0.071 | 256 | 0.278 | 0.278 | 0.52% |
| 1 | physics | 0.824 | 256 | 3.220 | 3.220 | 5.97% |
| 1 | brain_commit | 0.006 | 8 | 0.770 | 0.024 | 0.04% |
| 1 | brain_checkpoint_load | 0.108 | 1 | 108.200 | 0.423 | 0.78% |
| 1 | brain_checkpoint_save | 0.168 | 1 | 168.128 | 0.657 | 1.22% |
| 8 | retina | 6.755 | 256 | 26.388 | 26.388 | 61.69% |
| 8 | brain_batch | 1.515 | 32 | 47.331 | 5.916 | 13.83% |
| 8 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 8 | periphery | 0.059 | 256 | 0.231 | 0.231 | 0.54% |
| 8 | physics | 0.773 | 256 | 3.020 | 3.020 | 7.06% |
| 8 | brain_commit | 0.006 | 8 | 0.802 | 0.025 | 0.06% |
| 8 | brain_checkpoint_load | 0.468 | 1 | 467.771 | 1.827 | 4.27% |
| 8 | brain_checkpoint_save | 0.236 | 1 | 236.177 | 0.923 | 2.16% |

## Scaling signal

- N=8 / N=1 aggregate throughput: 1.260x
- N=8 / N=1 brain_batch total time for the same aggregate episode budget: 0.414x
- N=8 / N=1 retina total time: 0.859x
- N=8 / N=1 physics total time: 0.938x
