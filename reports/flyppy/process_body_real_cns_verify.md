# Flyppy process-body real-CNS parity verification

- generated_at_utc: 2026-09-15T06:54:17+00:00
- overall: PASS
- population: 4
- max control steps per episode: 32
- physics steps per control step: 10
- reference: current local-body train_flyppy_population.py
- candidate: process-isolated bodies + shared parent GPU CNS
- final checkpoint byte/tree digest equal: True
- episode and commit outcomes equal: True
- global version and neural step equal: True
- production checkpoint modified: False
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- reference final checkpoint digest: `8e8061a1a711a97af87391836742c38467f746d0da39cdf9add494a49d187d31`
- candidate final checkpoint digest: `8e8061a1a711a97af87391836742c38467f746d0da39cdf9add494a49d187d31`

## Reference episode results

```json
[
  {
    "episode": 0,
    "slot": 0,
    "source_weight_version": 24,
    "commit_from_version": 24,
    "commit_weight_version": 25,
    "version_staleness": 0,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  },
  {
    "episode": 1,
    "slot": 1,
    "source_weight_version": 24,
    "commit_from_version": 25,
    "commit_weight_version": 26,
    "version_staleness": 1,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  },
  {
    "episode": 2,
    "slot": 2,
    "source_weight_version": 24,
    "commit_from_version": 26,
    "commit_weight_version": 27,
    "version_staleness": 2,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  },
  {
    "episode": 3,
    "slot": 3,
    "source_weight_version": 24,
    "commit_from_version": 27,
    "commit_weight_version": 28,
    "version_staleness": 3,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  }
]
```

## Process-body episode results

```json
[
  {
    "episode": 0,
    "slot": 0,
    "source_weight_version": 24,
    "commit_from_version": 24,
    "commit_weight_version": 25,
    "version_staleness": 0,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  },
  {
    "episode": 1,
    "slot": 1,
    "source_weight_version": 24,
    "commit_from_version": 25,
    "commit_weight_version": 26,
    "version_staleness": 1,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  },
  {
    "episode": 2,
    "slot": 2,
    "source_weight_version": 24,
    "commit_from_version": 26,
    "commit_weight_version": 27,
    "version_staleness": 2,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  },
  {
    "episode": 3,
    "slot": 3,
    "source_weight_version": 24,
    "commit_from_version": 27,
    "commit_weight_version": 28,
    "version_staleness": 3,
    "control_steps": 32,
    "passed_gates": 0,
    "collision": false,
    "collision_reason": null,
    "finished": false,
    "max_x_mm": 9.069236885061002,
    "min_z_mm": 11.214420351238072,
    "max_z_mm": 11.461808353565818,
    "final_vx_mm_s": 452.4992152233506,
    "spawn_x_mm": 1.7820000000000022,
    "spawn_z_mm": 11.467332247573482,
    "initial_speed_mm_s": 450.0,
    "reward_events": 0,
    "aversive_events": 0
  }
]
```
