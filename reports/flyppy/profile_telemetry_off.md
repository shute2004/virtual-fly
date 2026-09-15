# Flyppy P0 performance profile

- backend: `gpu:Apple M1 (Metal)`
- measured control steps: 256
- telemetry: `off`
- wall-clock: 20.917137 s
- control steps/s: 12.239
- neurons / edges: 166700 / 25582938
- neural/body/trajectory stride: 10/1/10

| stage | calls | total s | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| control_step_total | 256 | 20.908158 | 81.672 | 77.447 | 100.809 |
| neural_control_total | 256 | 10.877479 | 42.490 | 40.075 | 45.194 |
| bridge_request:step | 261 | 10.309581 | 39.500 | 37.785 | 41.781 |
| retina_total | 256 | 7.683316 | 30.013 | 29.199 | 38.189 |
| retina_eye_readout | 256 | 5.470703 | 21.370 | 20.425 | 26.655 |
| physics | 256 | 2.206508 | 8.619 | 5.513 | 16.482 |
| periphery | 256 | 0.096424 | 0.377 | 0.328 | 0.689 |
| episode_reset_total | 5 | 0.008756 | 1.751 | 1.735 | 1.932 |
| bridge_request:reset_dynamics | 5 | 0.004925 | 0.985 | 0.909 | 1.237 |
| reinforcement_total | 5 | 0.004299 | 0.860 | 0.872 | 0.896 |
| course_update | 256 | 0.002126 | 0.008 | 0.007 | 0.012 |
| trajectory_serialize | 34 | 0.001556 | 0.046 | 0.043 | 0.083 |
| collision_query | 256 | 0.000840 | 0.003 | 0.003 | 0.005 |
| trajectory_write | 34 | 0.000278 | 0.008 | 0.001 | 0.052 |
| trajectory_flush | 5 | 0.000003 | 0.001 | 0.001 | 0.001 |

## Protocol

- request bytes: 18865407
- response bytes: 2319154
- request count: `{"reset_dynamics": 5, "step": 261}`
- mean retinal stimulus pairs/step: 2535.6
- mean read_body IDs/step: 268.0

`retina_total` includes `retina_eye_readout`. `bridge_request:step` overlaps conceptually with neural/reinforcement totals. This pass loads but never saves the production checkpoint. GPU kernel-level splitting is deferred until the wall-clock profile shows the neural bridge is dominant.
