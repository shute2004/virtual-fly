# Flyppy eye ray hit diagnostics

- generated_at_utc: 2026-09-15T09:16:47+00:00
- overall: PASS
- candidate direction fingerprints unique: 4/4
- candidate mj_multiRay output fingerprints unique: 1/4
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Camera ownership

- L: camera=virtual_fly/l_eye_cam_camera, body=virtual_fly/l_eye_cam_body, body_id=9, pos=[2.3309442623662084, 0.4000038099999999, 11.96519761320009]
- R: camera=virtual_fly/r_eye_cam_camera, body=virtual_fly/r_eye_cam_body, body_id=10, pos=[2.3309442623662084, -0.3999961899999999, 11.96519761320009]

## Candidate fingerprints

| candidate | matrix | sx | sy | sz | hits | near<=1e-4 | direction fp | output fp |
|---|---|---:|---:|---:|---:|---:|---|---|
| current | R.T | 1 | 1 | -1 | 406/406 | 0 | `be3504e8dbcbc54b` | `eefe16103d92c3b7` |
| flip-z | R.T | 1 | 1 | 1 | 406/406 | 0 | `50e0c72d05d45fea` | `eefe16103d92c3b7` |
| flip-xy | R.T | -1 | -1 | -1 | 406/406 | 0 | `2cd9d9210b0e7259` | `eefe16103d92c3b7` |
| matrix-R | R | 1 | 1 | -1 | 406/406 | 0 | `3453d1774d82788a` | `eefe16103d92c3b7` |

## current top hits

- geom groups: {4: 406}
- L: hits=174/174, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`dbdc66f859d6f2b2`, out_fp=`5b883e18e5ff1f97`
- R: hits=232/232, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`e2e433008646d4e7`, out_fp=`850cd40209f58ba0`

| count | geom | body | group |
|---:|---|---|---:|
| 232 | `virtual_fly/r_eye_cam_marker` | `virtual_fly/r_eye_cam_body` | 4 |
| 174 | `virtual_fly/l_eye_cam_marker` | `virtual_fly/l_eye_cam_body` | 4 |

## flip-z top hits

- geom groups: {4: 406}
- L: hits=174/174, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`6a8426cb80a3565c`, out_fp=`5b883e18e5ff1f97`
- R: hits=232/232, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`0399cdaa382f3207`, out_fp=`850cd40209f58ba0`

| count | geom | body | group |
|---:|---|---|---:|
| 232 | `virtual_fly/r_eye_cam_marker` | `virtual_fly/r_eye_cam_body` | 4 |
| 174 | `virtual_fly/l_eye_cam_marker` | `virtual_fly/l_eye_cam_body` | 4 |

## flip-xy top hits

- geom groups: {4: 406}
- L: hits=174/174, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`4b00dc86d7e4c6b0`, out_fp=`5b883e18e5ff1f97`
- R: hits=232/232, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`c4a9cb2acf86bde6`, out_fp=`850cd40209f58ba0`

| count | geom | body | group |
|---:|---|---|---:|
| 232 | `virtual_fly/r_eye_cam_marker` | `virtual_fly/r_eye_cam_body` | 4 |
| 174 | `virtual_fly/l_eye_cam_marker` | `virtual_fly/l_eye_cam_body` | 4 |

## matrix-R top hits

- geom groups: {4: 406}
- L: hits=174/174, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`6d1ddaf42d5719a2`, out_fp=`5b883e18e5ff1f97`
- R: hits=232/232, min_distance=0.060000000, median_distance=0.060000000, dir_fp=`3b45f73adf11a4d5`, out_fp=`850cd40209f58ba0`

| count | geom | body | group |
|---:|---|---|---:|
| 232 | `virtual_fly/r_eye_cam_marker` | `virtual_fly/r_eye_cam_body` | 4 |
| 174 | `virtual_fly/l_eye_cam_marker` | `virtual_fly/l_eye_cam_body` | 4 |

If direction fingerprints differ but mj_multiRay output fingerprints do not, the issue is downstream of direction construction (for example origin-inside/self geometry or API filtering). If both differ, the previous aggregate tie was caused by the depth/classification contract rather than an inert ray transform.
