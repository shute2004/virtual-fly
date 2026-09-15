# Flyppy direct-vision production cutover benchmark

- generated_at_utc: 2026-09-15T09:39:02+00:00
- overall: PASS
- population: 4
- episodes per case: 16
- max control steps per episode: 96
- direct rays/ommatidium: 13
- direct RGB framebuffer: no
- final checkpoint equality is intentionally not required because sensory semantics differ
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

| mode | valid | steps/s | elapsed s | aggregate steps | gates | collisions | global version end | neural step | retinal samples | mean active columns | mean active photoreceptors | mean retinal current | max retinal current |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raster | True | 20.736280 | 43.499 | 902 | 5 | 14 | 40 | 8366 | 134 | 681.694 | 2805.530 | 0.605662 | 2.000000 |
| direct-ray K=13 | True | 33.310066 | 27.109 | 903 | 5 | 16 | 40 | 8375 | 136 | 494.669 | 1952.456 | 0.756968 | 2.000000 |

## Throughput

- direct/raster aggregate throughput: **1.606x**
- raster checkpoint: `0e7a6d8cbedff09a62408c6b039d4461348f69025fb6227b0359199ef336b803`
- direct checkpoint: `525165fa68d38a43e2418196bb77fadd00dcce57b6ac5e4282cfe2844fb2c105`
- direct runtime metadata valid: True

A PASS means the image-free K-selected sensor completes the real shared-CNS learning loop with finite retinal drive, asynchronous commits, and a valid checkpoint while production state remains untouched. Behavioral equality to raster is not a cutover requirement.
