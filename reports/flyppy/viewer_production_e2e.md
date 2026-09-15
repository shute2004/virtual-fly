# Flyppy production viewer E2E probe

- diagnosis: **LIVE_VIEWER_PRODUCTION_E2E_OK**
- experiment: `/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-viewer-e2e-probe`
- production trainer: `train_flyppy_population_packed.py`
- vision: `direct-ray`, K=13
- health samples: 187
- distinct body keys observed: 56
- distinct fly.png mtimes observed: 60

## Checks

- viewer_server_started: PASS
- body_renderer_started: PASS
- trainer_completed: PASS
- viewer_stream_connected: PASS
- body_json_seen: PASS
- body_control_step_advanced: PASS
- neural_json_seen: PASS
- status_json_seen: PASS
- fly_png_seen: PASS
- fly_png_updated: PASS
- viewer_server_survived: PASS
- body_renderer_survived: PASS
- http_status.json: PASS
- http_body.json: PASS
- http_neural.json: PASS
- http_fly.png: PASS

## Viewer server log tail

```text
127.0.0.1 - - [15/Sep/2026 20:04:07] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:07] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:07] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:07] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:07] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:08] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:09] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:10] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:11] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:12] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:12] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:12] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:12] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 20:04:12] "GET /api/viewer-health HTTP/1.1" 200 -
```

## Body renderer log tail

```text

```

## Trainer log tail

```text
body_process_packing=packed population=1 body_processes=1 flies_per_process_mean=1.000 vision_mode=direct-ray rays_per_ommatidium=13
population_runtime=enabled population=1 shared_weight=true weight_averaging=false environment=v3 motor_boundary=whole-body body_runtime=process telemetry=viewer-demand
population_commit episode=0 slot=0 source_v=0 from_v=0 -> v1 staleness=0 steps=40 passed=1 collision=False
population_commit episode=1 slot=0 source_v=1 from_v=1 -> v2 staleness=0 steps=40 passed=1 collision=False
summary=artifacts/experiments/flyppy-viewer-e2e-probe/summary.json
trajectory=artifacts/experiments/flyppy-viewer-e2e-probe/trajectory.jsonl
commit_log=artifacts/experiments/flyppy-viewer-e2e-probe/commit-log.jsonl
checkpoint=artifacts/experiments/flyppy-viewer-e2e-probe/checkpoint
elapsed=5.926s
aggregate_control_steps_per_second=13.501
flyppy_async_shared_weight_process_population=PASS
```
