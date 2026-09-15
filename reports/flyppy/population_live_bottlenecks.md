# Flyppy eager-live population bottleneck profile

- generated_at_utc: 2026-09-15T05:33:53+00:00
- overall: PASS
- episodes per case: 8
- max control steps per episode: 32
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## End-to-end

| population | control steps | trainer elapsed s | aggregate steps/s | outer wall s |
|---:|---:|---:|---:|---:|
| 1 | 256 | 98.378 | 2.602202 | 101.016 |
| 8 | 256 | 96.167 | 2.662048 | 111.132 |

## Timed boundaries

Times are inclusive wall time at Python/CNS boundaries. Percentages use trainer elapsed and are diagnostic; small uninstrumented Python/course/I/O work appears as remainder.

| population | stage | seconds | calls | ms/call | ms/control-step | % trainer elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | retina | 8.850 | 256 | 34.570 | 34.570 | 9.00% |
| 1 | brain_batch | 3.291 | 256 | 12.855 | 12.855 | 3.35% |
| 1 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 1 | periphery | 0.065 | 256 | 0.253 | 0.253 | 0.07% |
| 1 | physics | 0.833 | 256 | 3.254 | 3.254 | 0.85% |
| 1 | brain_commit | 0.006 | 8 | 0.724 | 0.023 | 0.01% |
| 1 | brain_checkpoint_load | 0.091 | 1 | 90.679 | 0.354 | 0.09% |
| 1 | brain_checkpoint_save | 83.773 | 1 | 83773.407 | 327.240 | 85.15% |
| 8 | retina | 7.662 | 256 | 29.931 | 29.931 | 7.97% |
| 8 | brain_batch | 1.542 | 32 | 48.200 | 6.025 | 1.60% |
| 8 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 8 | periphery | 0.058 | 256 | 0.228 | 0.228 | 0.06% |
| 8 | physics | 0.702 | 256 | 2.743 | 2.743 | 0.73% |
| 8 | brain_commit | 0.006 | 8 | 0.736 | 0.023 | 0.01% |
| 8 | brain_checkpoint_load | 0.712 | 1 | 711.593 | 2.780 | 0.74% |
| 8 | brain_checkpoint_save | 84.182 | 1 | 84182.363 | 328.837 | 87.54% |

## Scaling signal

- N=8 / N=1 aggregate throughput: 1.023x
- N=8 / N=1 brain_batch total time for the same aggregate episode budget: 0.469x
- N=8 / N=1 retina total time: 0.866x
- N=8 / N=1 physics total time: 0.843x
