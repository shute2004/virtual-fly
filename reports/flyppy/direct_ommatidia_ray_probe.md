# Flyppy direct ommatidia ray probe

- generated_at_utc: 2026-09-15T09:01:50+00:00
- overall: PASS
- samples: 16
- physics steps between samples: 10
- spawn condition: x=1.782000 mm, z=11.467332 mm, vx=450.000000 mm/s
- required unique ommatidia: L=174, R=232, total=406
- direct path creates RGB framebuffer: no
- direct visibility query: MuJoCo mj_multiRay
- direct photometry: geom/material base channel + white sky + analytic ground checker approximation
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Performance

- reference raster sensor: 19.711 ms/readout
- reference raster sensor + transduction: 22.492 ms/control observation

| rays/ommatidium | total rays/readout | direct sensor ms | sensor speedup vs raster | direct sensor+transduction ms |
|---:|---:|---:|---:|---:|
| 1 | 406 | 1.150 | 17.138x | 3.471 |
| 3 | 1218 | 3.026 | 6.514x | 5.347 |
| 7 | 2842 | 6.979 | 2.824x | 9.293 |
| 13 | 5278 | 12.683 | 1.554x | 15.014 |

## Compatibility against FlyGym raster oracle

Errors cover only ommatidia that currently feed released MaleCNS retinal columns. The direct path intentionally does not attempt exact OpenGL lighting/texture filtering yet.

| rays/ommatidium | light MAE | light p95 abs | light max abs | MaleCNS current MAE | current p95 abs | current sign mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.224637 | 0.650000 | 0.650000 | 0.172966 | 1.250441 | 94.744% |
| 3 | 0.224637 | 0.650000 | 0.650000 | 0.172966 | 1.250441 | 94.744% |
| 7 | 0.224637 | 0.650000 | 0.650000 | 0.172966 | 1.250441 | 94.744% |
| 13 | 0.224637 | 0.650000 | 0.650000 | 0.172966 | 1.250441 | 94.744% |

## Interpretation contract

This probe validates the image-free sensor mechanics and measures the remaining photometric approximation gap. Production should not switch from the raster oracle solely because this probe executes; the next decision is based on both wall-clock gain and MaleCNS-current compatibility.
