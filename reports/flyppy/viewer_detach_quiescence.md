# Flyppy viewer detach quiescence probe

- diagnosis: **VIEWER_DETACH_QUIESCENCE_OK**
- trainer: production `train_flyppy_population_packed.py`
- population: 4
- vision: direct-ray K=13
- viewer demand: real `ViewerTelemetryLease` Unix stream
- exception: `none`

## Checks

- trainer_warmed: PASS
- headless_no_body_json: PASS
- headless_no_neural_json: PASS
- lease_connected: PASS
- telemetry_materialized: PASS
- body_advanced_while_attached: PASS
- neural_advanced_while_attached: PASS
- training_advanced_after_detach: PASS
- body_quiescent_after_settle: PASS
- neural_quiescent_after_settle: PASS
- trainer_still_running: PASS

## Observations

- settled_body_mtime_ns: `1789475337614070698`
- final_body_mtime_ns: `1789475337614070698`
- settled_neural_mtime_ns: `1789475337613467796`
- final_neural_mtime_ns: `1789475337613467796`
- detached_start_records: `136`
- final_records: `256`

## Trainer log tail

```text
population_commit episode=93 slot=1 source_v=90 from_v=91 -> v92 staleness=1 steps=1 passed=0 collision=True
population_commit episode=94 slot=3 source_v=91 from_v=92 -> v93 staleness=1 steps=1 passed=0 collision=True
population_commit episode=95 slot=1 source_v=92 from_v=93 -> v94 staleness=1 steps=1 passed=0 collision=True
population_commit episode=96 slot=3 source_v=93 from_v=94 -> v95 staleness=1 steps=1 passed=0 collision=True
population_commit episode=97 slot=1 source_v=94 from_v=95 -> v96 staleness=1 steps=1 passed=0 collision=True
population_commit episode=98 slot=3 source_v=95 from_v=96 -> v97 staleness=1 steps=1 passed=0 collision=True
population_commit episode=99 slot=1 source_v=96 from_v=97 -> v98 staleness=1 steps=1 passed=0 collision=True
population_commit episode=100 slot=3 source_v=97 from_v=98 -> v99 staleness=1 steps=1 passed=0 collision=True
population_commit episode=101 slot=1 source_v=98 from_v=99 -> v100 staleness=1 steps=1 passed=0 collision=True
population_commit episode=102 slot=3 source_v=99 from_v=100 -> v101 staleness=1 steps=1 passed=0 collision=True
population_commit episode=103 slot=1 source_v=100 from_v=101 -> v102 staleness=1 steps=1 passed=0 collision=True
population_commit episode=104 slot=3 source_v=101 from_v=102 -> v103 staleness=1 steps=1 passed=0 collision=True
population_commit episode=105 slot=1 source_v=102 from_v=103 -> v104 staleness=1 steps=1 passed=0 collision=True
population_commit episode=106 slot=3 source_v=103 from_v=104 -> v105 staleness=1 steps=1 passed=0 collision=True
population_commit episode=107 slot=1 source_v=104 from_v=105 -> v106 staleness=1 steps=1 passed=0 collision=True
population_commit episode=108 slot=3 source_v=105 from_v=106 -> v107 staleness=1 steps=1 passed=0 collision=True
population_commit episode=109 slot=1 source_v=106 from_v=107 -> v108 staleness=1 steps=1 passed=0 collision=True
population_commit episode=110 slot=3 source_v=107 from_v=108 -> v109 staleness=1 steps=1 passed=0 collision=True
population_commit episode=111 slot=1 source_v=108 from_v=109 -> v110 staleness=1 steps=1 passed=0 collision=True
population_commit episode=112 slot=3 source_v=109 from_v=110 -> v111 staleness=1 steps=1 passed=0 collision=True
population_commit episode=113 slot=1 source_v=110 from_v=111 -> v112 staleness=1 steps=1 passed=0 collision=True
population_commit episode=114 slot=3 source_v=111 from_v=112 -> v113 staleness=1 steps=1 passed=0 collision=True
Traceback (most recent call last):
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/train_flyppy_population_packed.py", line 123, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/train_flyppy_population_packed.py", line 116, in main
    result = trainer.main()
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/embodiment/train_flyppy_population_process.py", line 290, in main
    slot.slot: by_slot[slot.slot].receive("observe")
               ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
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
