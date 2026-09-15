# Flyppy direct ommatidia ray probe

- generated_at_utc: 2026-09-15T09:20:29+00:00
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

- reference raster sensor: 20.546 ms/readout
- reference raster sensor + transduction: 23.405 ms/control observation

| rays/ommatidium | total rays/readout | direct sensor ms | sensor speedup vs raster | direct sensor+transduction ms |
|---:|---:|---:|---:|---:|
| 1 | 406 | 1.213 | 16.937x | 3.739 |
| 3 | 1218 | 3.118 | 6.589x | 5.606 |
| 7 | 2842 | 7.161 | 2.869x | 9.657 |
| 13 | 5278 | 13.052 | 1.574x | 15.576 |

## Compatibility against FlyGym raster oracle

Errors cover only ommatidia that currently feed released MaleCNS retinal columns. The direct path intentionally does not attempt exact OpenGL lighting/texture filtering yet.

| rays/ommatidium | light MAE | light p95 abs | light max abs | MaleCNS current MAE | current p95 abs | current sign mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.135787 | 0.232530 | 0.487979 | 0.103401 | 0.591166 | 79.943% |
| 3 | 0.131225 | 0.206863 | 0.350000 | 0.072439 | 0.390196 | 68.839% |
| 7 | 0.129951 | 0.206863 | 0.350000 | 0.052909 | 0.305826 | 64.968% |
| 13 | 0.129851 | 0.206277 | 0.350000 | 0.047440 | 0.239691 | 61.910% |

## Interpretation contract

This probe validates the image-free sensor mechanics and measures the remaining photometric approximation gap. Production should not switch from the raster oracle solely because this probe executes; the next decision is based on both wall-clock gain and MaleCNS-current compatibility.
