# Flyppy eager-live population bottleneck profile

- generated_at_utc: 2026-09-15T06:31:38+00:00
- overall: PASS
- episodes per case: 8
- max control steps per episode: 32
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## End-to-end

| population | control steps | trainer elapsed s | aggregate steps/s | outer wall s |
|---:|---:|---:|---:|---:|
| 1 | 256 | 14.146 | 18.097505 | 16.979 |
| 8 | 256 | 11.103 | 23.056544 | 25.726 |

## Timed boundaries

Retina substage rows are nested inside retina and therefore should not be added again when computing total wall time. `retina_render_residual` is `retina_readout - retina_fisheye - retina_hex`, covering MuJoCo update_scene/render plus small Python/allocation overhead.

| population | stage | seconds | calls | ms/call | ms/control-step | % trainer elapsed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | retina | 8.001 | 256 | 31.255 | 31.255 | 56.56% |
| 1 | retina_readout | 7.229 | 256 | 28.238 | 28.238 | 51.10% |
| 1 | retina_fisheye | 2.271 | 512 | 4.435 | 8.870 | 16.05% |
| 1 | retina_hex | 0.625 | 512 | 1.221 | 2.441 | 4.42% |
| 1 | retina_render_residual | 4.333 | 256 | 16.927 | 16.927 | 30.63% |
| 1 | retina_transduction | 0.772 | 256 | 3.014 | 3.014 | 5.46% |
| 1 | brain_batch | 3.498 | 256 | 13.664 | 13.664 | 24.73% |
| 1 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 1 | periphery | 0.059 | 256 | 0.232 | 0.232 | 0.42% |
| 1 | physics | 0.797 | 256 | 3.114 | 3.114 | 5.64% |
| 1 | brain_commit | 0.006 | 8 | 0.690 | 0.022 | 0.04% |
| 1 | brain_checkpoint_load | 0.100 | 1 | 99.960 | 0.390 | 0.71% |
| 1 | brain_checkpoint_save | 0.166 | 1 | 166.321 | 0.650 | 1.18% |
| 8 | retina | 6.824 | 256 | 26.656 | 26.656 | 61.46% |
| 8 | retina_readout | 6.096 | 256 | 23.812 | 23.812 | 54.90% |
| 8 | retina_fisheye | 0.403 | 512 | 0.788 | 1.575 | 3.63% |
| 8 | retina_hex | 0.323 | 512 | 0.630 | 1.260 | 2.91% |
| 8 | retina_render_residual | 5.370 | 256 | 20.976 | 20.976 | 48.36% |
| 8 | retina_transduction | 0.727 | 256 | 2.842 | 2.842 | 6.55% |
| 8 | brain_batch | 1.496 | 32 | 46.745 | 5.843 | 13.47% |
| 8 | brain_reinforcement | 0.000 | 0 | 0.000 | 0.000 | 0.00% |
| 8 | periphery | 0.059 | 256 | 0.230 | 0.230 | 0.53% |
| 8 | physics | 0.702 | 256 | 2.743 | 2.743 | 6.32% |
| 8 | brain_commit | 0.007 | 8 | 0.915 | 0.029 | 0.07% |
| 8 | brain_checkpoint_load | 0.595 | 1 | 594.681 | 2.323 | 5.36% |
| 8 | brain_checkpoint_save | 0.249 | 1 | 248.795 | 0.972 | 2.24% |

## Scaling signal

- N=8 / N=1 aggregate throughput: 1.274x
- N=8 / N=1 brain_batch total time for the same aggregate episode budget: 0.428x
- N=8 / N=1 retina total time: 0.853x
- N=8 / N=1 retina readout total time: 0.843x
- N=8 / N=1 retina fisheye total time: 0.178x
- N=8 / N=1 retina hex total time: 0.516x
- N=8 / N=1 retina render residual total time: 1.239x
- N=8 / N=1 retina transduction total time: 0.943x
- N=8 / N=1 physics total time: 0.881x
