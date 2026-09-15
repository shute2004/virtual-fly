# Flyppy eager-live population bottleneck profile

- generated_at_utc: 2026-09-15T06:26:02+00:00
- overall: PASS
- episodes per case: 8
- max control steps per episode: 32
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## End-to-end

| population | control steps | trainer elapsed s | aggregate steps/s | outer wall s |
|---:|---:|---:|---:|---:|
| 1 | 256 | 13.883 | 18.439356 | 16.556 |
| 8 | 256 | 11.674 | 21.928939 | 26.192 |

## Timed boundaries

Retina readout/transduction rows are nested inside retina and therefore should not be added again when computing total wall time. Percentages use trainer elapsed and are diagnostic.

| population | stage | seconds | calls | ms/call | ms/control-step | % trainer elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | retina | 7.760 | 256 | 30.312 | 30.312 | 55.89% |
| 1 | retina_readout | 6.966 | 256 | 27.211 | 27.211 | 50.17% |
| 1 | retina_transduction | 0.793 | 256 | 3.098 | 3.098 | 5.71% |
| 1 | brain_batch | 3.329 | 256 | 13.005 | 13.005 | 23.98% |
| 1 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 1 | periphery | 0.061 | 256 | 0.239 | 0.239 | 0.44% |
| 1 | physics | 0.802 | 256 | 3.132 | 3.132 | 5.78% |
| 1 | brain_commit | 0.006 | 8 | 0.721 | 0.023 | 0.04% |
| 1 | brain_checkpoint_load | 0.103 | 1 | 103.344 | 0.404 | 0.74% |
| 1 | brain_checkpoint_save | 0.189 | 1 | 188.791 | 0.737 | 1.36% |
| 8 | retina | 7.083 | 256 | 27.668 | 27.668 | 60.67% |
| 8 | retina_readout | 6.310 | 256 | 24.649 | 24.649 | 54.05% |
| 8 | retina_transduction | 0.772 | 256 | 3.015 | 3.015 | 6.61% |
| 8 | brain_batch | 1.588 | 32 | 49.632 | 6.204 | 13.60% |
| 8 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 8 | periphery | 0.079 | 256 | 0.310 | 0.310 | 0.68% |
| 8 | physics | 0.896 | 256 | 3.502 | 3.502 | 7.68% |
| 8 | brain_commit | 0.008 | 8 | 1.014 | 0.032 | 0.07% |
| 8 | brain_checkpoint_load | 0.582 | 1 | 582.171 | 2.274 | 4.99% |
| 8 | brain_checkpoint_save | 0.325 | 1 | 324.908 | 1.269 | 2.78% |

## Scaling signal

- N=8 / N=1 aggregate throughput: 1.189x
- N=8 / N=1 brain_batch total time for the same aggregate episode budget: 0.477x
- N=8 / N=1 retina total time: 0.913x
- N=8 / N=1 retina readout total time: 0.906x
- N=8 / N=1 retina transduction total time: 0.973x
- N=8 / N=1 physics total time: 1.118x
