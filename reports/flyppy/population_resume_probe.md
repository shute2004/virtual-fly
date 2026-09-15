# Flyppy population resume probe

- generated_at_utc: 2026-09-15T03:16:23+00:00
- mutates training artifacts: false
- overall: FAIL

## Stages

| stage | ok | seconds | error |
|---|---|---:|---|
| validate_inputs | no | 0.000 | FileNotFoundError: missing inputs: /Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/embodiment-groups-v0.json |

## Details

```json
{
  "experiment": "/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3",
  "population": 2,
  "snapshot": "/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0"
}
```

## Failure

```text
Traceback (most recent call last):
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/analysis/probe_flyppy_population_resume.py", line 156, in main
    record_stage(stages, "validate_inputs", validate_inputs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/analysis/probe_flyppy_population_resume.py", line 67, in record_stage
    value = fn()
  File "/Users/<local-user>/Desktop/virtual-fly/scripts/analysis/probe_flyppy_population_resume.py", line 154, in validate_inputs
    raise FileNotFoundError("missing inputs: " + ", ".join(missing))
FileNotFoundError: missing inputs: /Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/embodiment-groups-v0.json
```
