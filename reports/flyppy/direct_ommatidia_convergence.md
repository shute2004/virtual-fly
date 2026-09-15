# Flyppy direct ommatidia convergence

- generated_at_utc: 2026-09-15T09:30:27+00:00
- overall: PASS
- samples: 12
- reference: every valid FlyGym receptive-field source pixel queried directly with mj_multiRay
- RGB framebuffer used: no
- full-reference rays/readout: 95951
- full-reference sensor ms: 237.688
- production checkpoint modified: no

| rays/ommatidium | total rays | sensor ms | speedup vs full | light MAE vs full | light r | current MAE vs full | current r |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 406 | 1.170 | 203.123x | 0.021779 | 0.973922 | 0.074499 | 0.848338 |
| 3 | 1218 | 3.210 | 74.036x | 0.010605 | 0.992879 | 0.051567 | 0.931654 |
| 7 | 2842 | 7.178 | 33.115x | 0.005584 | 0.998073 | 0.026877 | 0.981554 |
| 13 | 5278 | 13.192 | 18.017x | 0.003953 | 0.999013 | 0.019294 | 0.990365 |
| 25 | 10150 | 25.085 | 9.475x | 0.002549 | 0.999575 | 0.011446 | 0.996346 |
| 49 | 19894 | 48.773 | 4.873x | 0.001275 | 0.999880 | 0.007112 | 0.998571 |

This measures convergence of the image-free sensor to its own uncompressed receptive-field integral. Raster rendering is not used as the reference.
