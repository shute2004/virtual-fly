# Flyppy packed vision render-stage profile

- generated_at_utc: 2026-09-15T08:33:50+00:00
- overall: PASS
- concurrent body processes: 4
- samples per process: 64
- aggregate eye-readout calls: 256
- parent wall time: 2.931555 s
- aggregate readouts/s: 87.326
- renderer proxy installed in every child: True
- static ommatidia output bitwise stable: True

## Stage totals

Times below are summed across concurrent child processes. Nested stage times therefore describe work distribution, not parent wall-clock addends.

| stage | summed seconds | calls | ms/eye-readout | share of readout work |
|---|---:|---:|---:|---:|
| MuJoCo update_scene | 0.051800 | 512 | 0.202 | 0.44% |
| MuJoCo render | 10.505292 | 512 | 41.036 | 89.92% |
| fisheye correction | 0.647700 | 512 | 2.530 | 5.54% |
| hex/ommatidia aggregation | 0.360562 | 512 | 1.408 | 3.09% |
| other readout overhead | 0.118007 | 256 | 0.461 | 1.01% |

## Per-eye call cost

- update_scene: 0.101 ms/call
- render: 20.518 ms/call
- fisheye: 1.265 ms/call
- hex aggregation: 0.704 ms/call

Each eye-readout call contains the left and right compound-eye paths. The probe changes no physical state and requires bitwise-stable ommatidia output across all timed samples.
