# Flyppy population resume probe

- generated_at_utc: 2026-09-15T03:20:47+00:00
- mutates training artifacts: false
- overall: PASS

## Stages

| stage | ok | seconds | error |
|---|---|---:|---|
| validate_inputs | yes | 0.000 | - |
| read_checkpoint_manifest | yes | 0.000 | - |
| read_curriculum_state | yes | 0.000 | - |
| read_population_state | yes | 0.000 | - |
| construct_flybody_slots | yes | 4.778 | - |
| start_population_bridge | yes | 4.629 | - |
| bridge_ping | yes | 0.000 | - |
| load_production_checkpoint | yes | 0.404 | - |

## Details

```json
{
  "bridge_backend": "gpu-population:Apple M1 (Metal)",
  "bridge_global_weight_version": 0,
  "bridge_ping": null,
  "checkpoint_step": 5757,
  "constructed_slots": 2,
  "curriculum_episodes": 72,
  "experiment": "/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3",
  "initial_global_weight_version": 0,
  "loaded_checkpoint": {
    "event": "checkpoint_loaded",
    "global_weight_version": 0,
    "ok": true,
    "path": "/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3/checkpoint",
    "step": 5757
  },
  "population": 2,
  "snapshot": "/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0"
}
```
