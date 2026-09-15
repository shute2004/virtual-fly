# Flyppy P0 performance profile

- backend: `gpu:Apple M1 (Metal)`
- measured control steps: 256
- telemetry: `on`
- wall-clock: 20.327913 s
- control steps/s: 12.594
- neurons / edges: 166700 / 25582938
- neural/body/trajectory stride: 10/1/10

| stage | calls | total s | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| control_step_total | 256 | 20.316463 | 79.361 | 75.965 | 86.507 |
| neural_control_total | 256 | 10.731957 | 41.922 | 39.199 | 45.203 |
| bridge_request:step | 261 | 10.173593 | 38.979 | 37.022 | 42.678 |
| retina_total | 256 | 7.267034 | 28.387 | 28.260 | 32.933 |
| retina_eye_readout | 256 | 5.125272 | 20.021 | 19.643 | 22.808 |
| physics | 256 | 1.924151 | 7.516 | 4.626 | 14.728 |
| telemetry_body | 256 | 0.216986 | 0.848 | 0.838 | 1.085 |
| periphery | 256 | 0.086803 | 0.339 | 0.335 | 0.438 |
| telemetry_neural | 34 | 0.021145 | 0.622 | 0.590 | 1.028 |
| episode_reset_total | 5 | 0.011186 | 2.237 | 2.181 | 2.709 |
| bridge_request:reset_dynamics | 5 | 0.005053 | 1.011 | 0.931 | 1.191 |
| reinforcement_total | 5 | 0.004523 | 0.905 | 0.917 | 1.021 |
| telemetry_status | 5 | 0.001900 | 0.380 | 0.336 | 0.576 |
| course_update | 256 | 0.001786 | 0.007 | 0.007 | 0.009 |
| trajectory_serialize | 34 | 0.001413 | 0.042 | 0.041 | 0.065 |
| collision_query | 256 | 0.000841 | 0.003 | 0.003 | 0.004 |
| trajectory_write | 34 | 0.000306 | 0.009 | 0.001 | 0.059 |
| trajectory_flush | 5 | 0.000003 | 0.001 | 0.001 | 0.001 |

## Protocol

- request bytes: 19312268
- response bytes: 4568533
- request count: `{"reset_dynamics": 5, "step": 261}`
- mean retinal stimulus pairs/step: 2535.6
- mean read_body IDs/step: 539.3

`retina_total` includes `retina_eye_readout`. `bridge_request:step` overlaps conceptually with neural/reinforcement totals. This pass loads but never saves the production checkpoint. GPU kernel-level splitting is deferred until the wall-clock profile shows the neural bridge is dominant.
