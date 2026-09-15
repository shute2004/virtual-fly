# Flyppy viewer runtime overhead benchmark

- diagnosis: **VIEWER_RUNTIME_OVERHEAD_MEASURED**
- trainer: production `train_flyppy_population_packed.py`
- population: 4
- vision: direct-ray K=13
- measurement: trajectory records/s (aggregate control-step samples)
- attached phase: Unix stream lease + 960x540 detached MuJoCo renderer at 30 Hz
- browser/Three.js rendering: excluded from throughput measurement
- exception: `none`

## Throughput

- headless: 46.907 records/s
- viewer attached: 19.732 records/s
- attached/headless: 0.421x
- viewer detached again: 12.095 records/s
- detached/headless: 0.258x

## State checks

- trainer_warmed: PASS
- headless_no_body_json: PASS
- headless_no_neural_json: PASS
- headless_phase_completed: PASS
- viewer_connected: PASS
- viewer_body_json: PASS
- viewer_neural_json: PASS
- body_renderer_alive: PASS
- viewer_server_alive: PASS
- viewer_phase_completed: PASS
- detached_phase_completed: PASS
- training_advanced_after_detach: PASS
- body_json_stopped_after_detach: PASS
- neural_json_stopped_after_detach: PASS
- trainer_completed: PASS

## Trainer log tail

```text
body_process_packing=packed population=4 body_processes=4 flies_per_process_mean=1.000 vision_mode=direct-ray rays_per_ommatidium=13
population_runtime=enabled population=4 shared_weight=true weight_averaging=false environment=v3 motor_boundary=whole-body body_runtime=process telemetry=viewer-demand
population_commit episode=1 slot=1 source_v=0 from_v=0 -> v1 staleness=0 steps=5 passed=0 collision=True
population_commit episode=3 slot=3 source_v=0 from_v=1 -> v2 staleness=1 steps=5 passed=0 collision=True
population_commit episode=5 slot=3 source_v=2 from_v=2 -> v3 staleness=0 steps=4 passed=0 collision=True
population_commit episode=4 slot=1 source_v=1 from_v=3 -> v4 staleness=2 steps=5 passed=0 collision=True
population_commit episode=6 slot=3 source_v=3 from_v=4 -> v5 staleness=1 steps=1 passed=0 collision=True
population_commit episode=7 slot=1 source_v=4 from_v=5 -> v6 staleness=1 steps=1 passed=0 collision=True
population_commit episode=8 slot=3 source_v=5 from_v=6 -> v7 staleness=1 steps=1 passed=0 collision=True
population_commit episode=9 slot=1 source_v=6 from_v=7 -> v8 staleness=1 steps=1 passed=0 collision=True
population_commit episode=10 slot=3 source_v=7 from_v=8 -> v9 staleness=1 steps=1 passed=0 collision=True
population_commit episode=11 slot=1 source_v=8 from_v=9 -> v10 staleness=1 steps=1 passed=0 collision=True
population_commit episode=12 slot=3 source_v=9 from_v=10 -> v11 staleness=1 steps=1 passed=0 collision=True
population_commit episode=13 slot=1 source_v=10 from_v=11 -> v12 staleness=1 steps=1 passed=0 collision=True
population_commit episode=14 slot=3 source_v=11 from_v=12 -> v13 staleness=1 steps=1 passed=0 collision=True
population_commit episode=15 slot=1 source_v=12 from_v=13 -> v14 staleness=1 steps=1 passed=0 collision=True
population_commit episode=16 slot=3 source_v=13 from_v=14 -> v15 staleness=1 steps=1 passed=0 collision=True
population_commit episode=17 slot=1 source_v=14 from_v=15 -> v16 staleness=1 steps=1 passed=0 collision=True
population_commit episode=18 slot=3 source_v=15 from_v=16 -> v17 staleness=1 steps=1 passed=0 collision=True
population_commit episode=19 slot=1 source_v=16 from_v=17 -> v18 staleness=1 steps=1 passed=0 collision=True
population_commit episode=20 slot=3 source_v=17 from_v=18 -> v19 staleness=1 steps=1 passed=0 collision=True
population_commit episode=21 slot=1 source_v=18 from_v=19 -> v20 staleness=1 steps=1 passed=0 collision=True
population_commit episode=22 slot=3 source_v=19 from_v=20 -> v21 staleness=1 steps=1 passed=0 collision=True
population_commit episode=23 slot=1 source_v=20 from_v=21 -> v22 staleness=1 steps=1 passed=0 collision=True
population_commit episode=24 slot=3 source_v=21 from_v=22 -> v23 staleness=1 steps=1 passed=0 collision=True
population_commit episode=25 slot=1 source_v=22 from_v=23 -> v24 staleness=1 steps=1 passed=0 collision=True
population_commit episode=26 slot=3 source_v=23 from_v=24 -> v25 staleness=1 steps=1 passed=0 collision=True
population_commit episode=27 slot=1 source_v=24 from_v=25 -> v26 staleness=1 steps=1 passed=0 collision=True
population_commit episode=28 slot=3 source_v=25 from_v=26 -> v27 staleness=1 steps=1 passed=0 collision=True
population_commit episode=29 slot=1 source_v=26 from_v=27 -> v28 staleness=1 steps=1 passed=0 collision=True
population_commit episode=30 slot=3 source_v=27 from_v=28 -> v29 staleness=1 steps=1 passed=0 collision=True
population_commit episode=31 slot=1 source_v=28 from_v=29 -> v30 staleness=1 steps=1 passed=0 collision=True
population_commit episode=32 slot=3 source_v=29 from_v=30 -> v31 staleness=1 steps=1 passed=0 collision=True
population_commit episode=33 slot=1 source_v=30 from_v=31 -> v32 staleness=1 steps=1 passed=0 collision=True
population_commit episode=34 slot=3 source_v=31 from_v=32 -> v33 staleness=1 steps=1 passed=0 collision=True
population_commit episode=35 slot=1 source_v=32 from_v=33 -> v34 staleness=1 steps=1 passed=0 collision=True
population_commit episode=36 slot=3 source_v=33 from_v=34 -> v35 staleness=1 steps=1 passed=0 collision=True
population_commit episode=37 slot=1 source_v=34 from_v=35 -> v36 staleness=1 steps=1 passed=0 collision=True
population_commit episode=38 slot=3 source_v=35 from_v=36 -> v37 staleness=1 steps=1 passed=0 collision=True
population_commit episode=39 slot=1 source_v=36 from_v=37 -> v38 staleness=1 steps=1 passed=0 collision=True
population_commit episode=2 slot=2 source_v=0 from_v=38 -> v39 staleness=38 steps=73 passed=1 collision=True
population_commit episode=0 slot=0 source_v=0 from_v=39 -> v40 staleness=39 steps=116 passed=2 collision=True
summary=artifacts/experiments/flyppy-viewer-overhead-benchmark/summary.json
trajectory=artifacts/experiments/flyppy-viewer-overhead-benchmark/trajectory.jsonl
commit_log=artifacts/experiments/flyppy-viewer-overhead-benchmark/commit-log.jsonl
checkpoint=artifacts/experiments/flyppy-viewer-overhead-benchmark/checkpoint
elapsed=14.398s
aggregate_control_steps_per_second=16.808
flyppy_async_shared_weight_process_population=PASS
```

## Viewer server log tail

```text
127.0.0.1 - - [15/Sep/2026 21:22:13] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:22:13] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:22:13] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:22:13] "GET /api/viewer-health HTTP/1.1" 200 -
```

## Body renderer log tail

```text

```
