# Flyppy direct ommatidia photometry probe

- generated_at_utc: 2026-09-15T09:06:50+00:00
- overall: PASS
- samples: 16
- rays per required ommatidium: 1
- total rays/readout: 406
- RGB framebuffer used by direct methods: no
- MuJoCo explicit lights: 0
- headlight active: True
- headlight ambient: [np.float64(0.5), np.float64(0.5), np.float64(0.5)]
- headlight diffuse: [np.float64(0.6000000238418579), np.float64(0.6000000238418579), np.float64(0.6000000238418579)]
- headlight specular: [np.float64(0.0), np.float64(0.0), np.float64(0.0)]
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Performance and raster-oracle compatibility

- reference raster sensor: 21.462 ms/readout
- reference raster sensor + transduction: 24.703 ms/control observation

| method | sensor ms | speedup vs raster | sensor+transduction ms | light MAE | light p95 | current MAE | current p95 | sign mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base-colour | 1.618 | 13.264x | 4.079 | 0.224637 | 0.650000 | 0.172966 | 1.250441 | 94.744% |
| headlight-lambert | 2.353 | 9.122x | 4.761 | 0.339003 | 0.825000 | 0.180464 | 1.338254 | 94.744% |

The raster path is used here only as a compatibility oracle. The direct sensor remains the architectural target; this probe determines whether the remaining gap is primarily photometric rather than geometric.
