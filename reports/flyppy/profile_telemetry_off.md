# Flyppy P0 performance profile

- backend: `gpu:Apple M1 (Metal)`
- measured control steps: 256
- telemetry: `off`
- wall-clock: 20.220593 s
- control steps/s: 12.660
- neurons / edges: 166700 / 25582938
- neural/body/trajectory stride: 10/1/10

| stage | calls | total s | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| control_step_total | 256 | 20.212501 | 78.955 | 75.698 | 87.041 |
| neural_control_total | 256 | 10.574042 | 41.305 | 38.974 | 43.579 |
| bridge_request:step | 261 | 10.022402 | 38.400 | 36.464 | 40.534 |
| retina_total | 256 | 7.471860 | 29.187 | 29.048 | 37.084 |
| retina_eye_readout | 256 | 5.251694 | 20.514 | 19.857 | 25.040 |
| physics | 256 | 2.039899 | 7.968 | 4.768 | 14.865 |
| periphery | 256 | 0.087559 | 0.342 | 0.339 | 0.495 |
| episode_reset_total | 5 | 0.007869 | 1.574 | 1.538 | 1.815 |
| bridge_request:reset_dynamics | 5 | 0.004490 | 0.898 | 0.864 | 1.120 |
| reinforcement_total | 5 | 0.003983 | 0.797 | 0.788 | 0.892 |
| course_update | 256 | 0.001776 | 0.007 | 0.006 | 0.010 |
| trajectory_serialize | 34 | 0.001531 | 0.045 | 0.042 | 0.068 |
| collision_query | 256 | 0.000748 | 0.003 | 0.003 | 0.004 |
| trajectory_write | 34 | 0.000418 | 0.012 | 0.001 | 0.072 |
| trajectory_flush | 5 | 0.000003 | 0.001 | 0.000 | 0.001 |

## Protocol

- request bytes: 18865407
- response bytes: 2319154
- request count: `{"reset_dynamics": 5, "step": 261}`
- mean retinal stimulus pairs/step: 2535.6
- mean read_body IDs/step: 268.0

`retina_total` includes `retina_eye_readout`. `bridge_request:step` overlaps conceptually with neural/reinforcement totals. This pass loads but never saves the production checkpoint. GPU kernel-level splitting is deferred until the wall-clock profile shows the neural bridge is dominant.
