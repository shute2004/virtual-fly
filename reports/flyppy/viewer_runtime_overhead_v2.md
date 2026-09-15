# Flyppy viewer runtime overhead benchmark v2

- diagnosis: **VIEWER_RUNTIME_OVERHEAD_V2_FAILED**
- trainer: production `train_flyppy_population_packed.py`
- population: 4, replenished throughout all measured phases
- vision: direct-ray K=13
- measurement: trajectory records/s with `--trajectory-stride=1` (one record per production control step)
- warmup records: 120
- records per phase: 180
- episode budget: 10000 (intentionally not completed)
- attached phase: Unix stream lease + 960x540 detached MuJoCo renderer at 30 Hz
- browser/Three.js rendering: excluded
- exception: `none`

## Throughput

- headless before attach: 27.862 control steps/s
- viewer attached: 26.265 control steps/s
- attached/headless-before: 0.943x
- headless after detach: 29.927 control steps/s
- after/before: 1.074x

The before/after ratio is reported as an observation, not a correctness threshold; learning state evolves during the run. Correct detach behavior is established separately by telemetry mtimes stopping while training continues.

## State checks

- trainer_warmed: PASS
- headless_no_body_json: PASS
- headless_no_neural_json: PASS
- headless_before_completed: PASS
- viewer_connected: PASS
- viewer_body_json: PASS
- viewer_neural_json: PASS
- body_renderer_alive: PASS
- viewer_server_alive: PASS
- viewer_phase_completed: PASS
- headless_after_completed: PASS
- training_advanced_after_detach: PASS
- body_json_stopped_after_detach: FAIL
- neural_json_stopped_after_detach: FAIL
- trainer_still_running_after_measurement: PASS
- episode_budget_not_exhausted: PASS

### headless_before
- start trajectory records: 120
- end trajectory records: 300
- elapsed: 6.460s
- rate: 27.862 control steps/s

### viewer_attached
- start trajectory records: 324
- end trajectory records: 504
- elapsed: 6.853s
- rate: 26.265 control steps/s

### headless_after
- start trajectory records: 916
- end trajectory records: 1096
- elapsed: 6.015s
- rate: 29.927 control steps/s

## Trainer log tail
```text
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/train_flyppy_population_packed.py", line 116, in main
    result = trainer.main()
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/train_flyppy_population_process.py", line 537, in main
    reset_slot(
    ~~~~~~~~~~^
        slot,
        ^^^^^
    ...<3 lines>...
        condition=current_adaptive_condition(state),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/train_flyppy_population_process.py", line 103, in reset_slot
    initial = worker.reset(
        episode=slot.episode,
    ...<3 lines>...
        initial_speed_mm_s=slot.initial_speed_mm_s,
    )
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/flyppy_packed_slot_adapter.py", line 62, in reset
    return self.call(
           ~~~~~~~~~^
        "reset",
        ^^^^^^^^
    ...<6 lines>...
        },
        ^^
    )
    ^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/flyppy_packed_slot_adapter.py", line 51, in call
    return self.receive(command)
           ~~~~~~~~~~~~^^^^^^^^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/flyppy_packed_slot_adapter.py", line 47, in receive
    return self._unwrap(phase, self._worker.receive(phase))
                               ~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/flyppy_packed_body_worker.py", line 347, in receive
    status, payload = self._recv(phase)
                      ~~~~~~~~~~^^^^^^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/flyppy_packed_body_worker.py", line 317, in _recv
    if not self.connection.poll(self.timeout_s):
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/connection.py", line 257, in poll
    return self._poll(timeout)
           ~~~~~~~~~~^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/connection.py", line 440, in _poll
    r = wait([self], timeout)
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/multiprocessing/connection.py", line 1148, in wait
    ready = selector.select(timeout)
  File "/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/lib/python3.13/selectors.py", line 398, in select
    fd_event_list = self._selector.poll(timeout)
KeyboardInterrupt
```

## Viewer server log tail
```text
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
127.0.0.1 - - [15/Sep/2026 21:25:29] "GET /api/viewer-health HTTP/1.1" 200 -
```

## Body renderer log tail
```text

```
