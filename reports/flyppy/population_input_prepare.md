# Flyppy population input preparation

- generated_at_utc: 2026-09-15T03:20:36+00:00
- overall: PASS
- mutates Flyppy training state: false

## Inputs

- snapshot: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0`
- groups: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/embodiment-groups-v0.json`
- wing motor map: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/wing-motor-neurons-v0.json`
- body motor map: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/body-motor-neurons-v0.json`
- retinotopic map: `/Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/retinotopic-vision-v1.json`

## Stages

### snapshot — PASS

required snapshot files present

### generate_groups — PASS

- returncode: 0

stdout:
```text
aversive_dan=2
flight_thrust_left=15
flight_thrust_right=14
reward_dan=44
vision_t4c_left=895
vision_t4c_right=883
vision_t4d_left=850
vision_t4d_right=860
vision_t5c_left=862
vision_t5c_right=858
vision_t5d_left=812
vision_t5d_right=808
reward_dan_type=PAM01
aversive_dan_type=PPL101
wrote /Users/<local-user>/Desktop/virtual-fly/artifacts/malecns-v1.0/embodiment-groups-v0.json
```

### generate_wing_motor_map — PASS

already present

### generate_body_motor_map — PASS

already present

### generate_retinotopic_map — PASS

already present

### validate_generated_inputs — PASS

groups=12 reward_dan+aversive_dan present
