# Flyppy eye ray depth alignment

- generated_at_utc: 2026-09-15T09:13:20+00:00
- overall: PASS
- oracle: MuJoCo metric depth rendering at FlyGym Retina raw resolution
- comparison: representative raw source pixel per required ommatidium
- required samples: L=174, R=232
- far plane: 10039.659511
- current convention: matrix=R.T, sx=1, sy=1, sz=-1
- current hit/background agreement: 74.138%
- current paired-hit depth MAE: 12.766526
- current paired-hit depth RMSE: 24.367084
- best convention: matrix=R, sx=-1, sy=-1, sz=-1
- best hit/background agreement: 74.138%
- best paired-hit depth MAE: 12.766526
- best paired-hit depth RMSE: 24.367084
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## All camera-frame conventions

| rank | matrix | sx | sy | sz | hit/bg agreement | paired hits | depth MAE | depth RMSE | L agreement | R agreement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | R | -1 | -1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 2 | R | -1 | -1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 3 | R | -1 | 1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 4 | R | -1 | 1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 5 | R | 1 | -1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 6 | R | 1 | -1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 7 | R | 1 | 1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 8 | R | 1 | 1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 9 | R.T | -1 | -1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 10 | R.T | -1 | -1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 11 | R.T | -1 | 1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 12 | R.T | -1 | 1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 13 | R.T | 1 | -1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 14 | R.T | 1 | -1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 15 | R.T | 1 | 1 | -1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |
| 16 | R.T | 1 | 1 | 1 | 74.138% | 301 | 12.766526 | 24.367084 | 76.437% | 72.414% |

Depth alignment avoids segmentation object-ID semantics entirely. A correct camera/ray convention should simultaneously maximize hit/background agreement and minimize paired-hit metric-depth error.
