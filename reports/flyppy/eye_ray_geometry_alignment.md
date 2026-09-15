# Flyppy eye ray geometry alignment

- generated_at_utc: 2026-09-15T09:09:57+00:00
- overall: PASS
- oracle: MuJoCo segmentation rendering at FlyGym Retina raw resolution
- comparison: representative raw source pixel per required ommatidium
- required samples: L=174, R=232
- current convention match: 0.000% (L=0.000%, R=0.000%)
- best convention: matrix=R, sx=1, sy=1, sz=-1
- best match: 0.000% (L=0.000%, R=0.000%)
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## All camera-frame conventions

| rank | matrix | sx | sy | sz | combined match | L match | R match |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | R | 1 | 1 | -1 | 0.000% | 0.000% | 0.000% |
| 2 | R | 1 | 1 | 1 | 0.000% | 0.000% | 0.000% |
| 3 | R | 1 | -1 | -1 | 0.000% | 0.000% | 0.000% |
| 4 | R | 1 | -1 | 1 | 0.000% | 0.000% | 0.000% |
| 5 | R | -1 | 1 | -1 | 0.000% | 0.000% | 0.000% |
| 6 | R | -1 | 1 | 1 | 0.000% | 0.000% | 0.000% |
| 7 | R | -1 | -1 | -1 | 0.000% | 0.000% | 0.000% |
| 8 | R | -1 | -1 | 1 | 0.000% | 0.000% | 0.000% |
| 9 | R.T | 1 | 1 | -1 | 0.000% | 0.000% | 0.000% |
| 10 | R.T | 1 | 1 | 1 | 0.000% | 0.000% | 0.000% |
| 11 | R.T | 1 | -1 | -1 | 0.000% | 0.000% | 0.000% |
| 12 | R.T | 1 | -1 | 1 | 0.000% | 0.000% | 0.000% |
| 13 | R.T | -1 | 1 | -1 | 0.000% | 0.000% | 0.000% |
| 14 | R.T | -1 | 1 | 1 | 0.000% | 0.000% | 0.000% |
| 15 | R.T | -1 | -1 | -1 | 0.000% | 0.000% | 0.000% |
| 16 | R.T | -1 | -1 | 1 | 0.000% | 0.000% | 0.000% |

A high segmentation-ID match means camera projection/pose and ray geometry agree independently of lighting. If the current convention is not the best, direct sensor geometry should be corrected before further photometry work.
