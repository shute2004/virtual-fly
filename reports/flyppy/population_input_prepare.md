# Flyppy population input preparation

- generated_at_utc: 2026-09-16T03:45:56+00:00
- overall: PASS
- mutates Flyppy training state: false

## Inputs

- snapshot: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0`
- groups: `/Users/<local-user>/Desktop/virtual-fly/20`
- wing motor map: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/wing-motor-neurons-v0.json`
- body motor map: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/body-motor-neurons-v0.json`
- retinotopic map: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/retinotopic-vision-v1.json`

## Stages

### snapshot — PASS

required snapshot files present

### generate_groups — PASS

already present

### generate_wing_motor_map — PASS

already present

### generate_body_motor_map — PASS

already present

### generate_retinotopic_map — PASS

already present

### validate_generated_inputs — PASS

groups=12 reward_dan+aversive_dan present
