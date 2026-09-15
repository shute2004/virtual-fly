# Flyppy direct ommatidia adaptation-gap probe

- generated_at_utc: 2026-09-15T09:24:34+00:00
- overall: PASS
- samples: 32
- physics steps between samples: 10
- spawn condition: x=1.782000 mm, z=11.467332 mm, vx=450.000000 mm/s
- direct sensor: body-excluded mj_multiRay; no RGB framebuffer
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Temporal transduction separation

`natural` lets raster and direct paths accumulate their own adaptation histories. `shared-baseline` injects direct light while restoring the raster path's pre-step adaptation baseline first.

| rays/ommatidium | sensor ms | light bias (direct-ref) | light MAE | affine a | affine b | affine MAE | affine R2 | natural current MAE | natural sign mismatch | shared-baseline current MAE | shared-baseline sign mismatch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3.079 | 0.117796 | 0.128436 | 1.063480 | -0.158240 | 0.093654 | 0.876475 | 0.113102 | 59.834% | 0.570979 | 61.843% |
| 7 | 6.804 | 0.118024 | 0.126827 | 1.082802 | -0.170796 | 0.089183 | 0.886834 | 0.097144 | 56.986% | 0.570139 | 61.312% |
| 13 | 12.393 | 0.118190 | 0.126733 | 1.087055 | -0.173688 | 0.088630 | 0.888327 | 0.091304 | 53.865% | 0.571877 | 61.244% |

## Current-error tails

| rays/ommatidium | natural p95 abs | shared-baseline p95 abs |
|---:|---:|---:|
| 3 | 0.442980 | 1.103079 |
| 7 | 0.401346 | 1.098483 |
| 13 | 0.390196 | 1.096682 |

The affine fit is diagnostic only. A low affine residual would indicate a mostly global photometric calibration gap; a large residual means geometry/material/local lighting differences remain.
