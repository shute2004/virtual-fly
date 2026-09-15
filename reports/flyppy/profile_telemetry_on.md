# Flyppy P0 performance profile

- backend: `gpu:Apple M1 (Metal)`
- measured control steps: 256
- telemetry: `on`
- wall-clock: 20.140259 s
- control steps/s: 12.711
- neurons / edges: 166700 / 25582938
- neural/body/trajectory stride: 10/1/10

| stage | calls | total s | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| control_step_total | 256 | 20.129990 | 78.633 | 75.312 | 83.077 |
| neural_control_total | 256 | 10.575521 | 41.311 | 38.885 | 42.491 |
| bridge_request:step | 261 | 10.019029 | 38.387 | 36.558 | 39.604 |
| retina_total | 256 | 7.260422 | 28.361 | 28.502 | 34.193 |
| retina_eye_readout | 256 | 5.119104 | 19.997 | 19.660 | 23.580 |
| physics | 256 | 1.899679 | 7.421 | 4.643 | 14.745 |
| telemetry_body | 256 | 0.218000 | 0.852 | 0.846 | 1.194 |
| periphery | 256 | 0.084515 | 0.330 | 0.330 | 0.418 |
| telemetry_neural | 34 | 0.023444 | 0.690 | 0.561 | 1.204 |
| episode_reset_total | 5 | 0.009924 | 1.985 | 1.940 | 2.140 |
| bridge_request:reset_dynamics | 5 | 0.004642 | 0.928 | 0.935 | 1.036 |
| reinforcement_total | 5 | 0.003995 | 0.799 | 0.826 | 0.865 |
| course_update | 256 | 0.001776 | 0.007 | 0.007 | 0.009 |
| telemetry_status | 5 | 0.001536 | 0.307 | 0.322 | 0.367 |
| collision_query | 256 | 0.001496 | 0.006 | 0.002 | 0.003 |
| trajectory_serialize | 34 | 0.001475 | 0.043 | 0.041 | 0.063 |
| trajectory_write | 34 | 0.000375 | 0.011 | 0.001 | 0.056 |
| trajectory_flush | 5 | 0.000003 | 0.001 | 0.000 | 0.001 |

## Protocol

- request bytes: 19312268
- response bytes: 4568533
- request count: `{"reset_dynamics": 5, "step": 261}`
- mean retinal stimulus pairs/step: 2535.6
- mean read_body IDs/step: 539.3

`retina_total` includes `retina_eye_readout`. `bridge_request:step` overlaps conceptually with neural/reinforcement totals. This pass loads but never saves the production checkpoint. GPU kernel-level splitting is deferred until the wall-clock profile shows the neural bridge is dominant.
