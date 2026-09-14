#!/usr/bin/env python3
"""Calibration-only smoke test for the grounded whole-body motor boundary.

Synthetic peripheral spikes are permitted here only to verify wiring and body
mechanics. They are never used by the training path as a controller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flybody_neuromuscular_adapter import FlyBodyNeuromuscularAdapter
from whole_body_periphery import WholeBodyPeriphery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wing-motor-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"),
    )
    parser.add_argument(
        "--body-motor-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/body-motor-neurons-v0.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    periphery = WholeBodyPeriphery(args.wing_motor_map, args.body_motor_map)
    leg_units = [unit for unit in periphery.units if unit.leg_prefix]
    haltere_units = [
        unit for unit in periphery.units if unit.subclass == "hm" and unit.action == "power"
    ]
    if not leg_units:
        raise RuntimeError("whole-body boundary contains no grounded leg motor unit")

    body = FlyBodyNeuromuscularAdapter(tethered=True, enable_vision=False)
    dt_s = body.timestep * 10
    spikes = {body_id: False for body_id in periphery.body_ids}

    # Pick one released, grounded leg MN and verify that its local peripheral
    # activation reaches the corresponding generalized force and changes the
    # biological joint state without exceeding the FlyBody joint range.
    unit = leg_units[0]
    joint_name = body._joint_name(unit.leg_prefix, unit.joint_role)
    qpos_address, _ = body.joint_state_addresses(joint_name)
    low, high = body.joint_range_rad(joint_name)
    before = float(body.sim.mj_data.qpos[qpos_address])
    peak_torque = 0.0
    for _ in range(6):
        spikes[unit.body_id] = True
        state = periphery.step(spikes, dt_s=dt_s)
        body.step_muscles(state, physics_steps=10)
        spikes[unit.body_id] = False
        peak_torque = max(
            peak_torque,
            abs(float(body.last_somatic_torque.get(joint_name, 0.0))),
        )
    after = float(body.sim.mj_data.qpos[qpos_address])
    if peak_torque <= 0.0:
        raise RuntimeError(
            f"grounded leg MN {unit.body_id} did not reach physical joint torque {joint_name}"
        )
    if abs(after - before) <= 1e-7:
        raise RuntimeError(
            f"grounded leg MN {unit.body_id} produced torque but no joint motion at {joint_name}"
        )
    if not (low - 1e-5 <= after <= high + 1e-5):
        raise RuntimeError(
            f"leg joint exceeded biological range: {joint_name} angle={after} range={(low, high)}"
        )

    haltere_peak_torque = 0.0
    if haltere_units:
        hunit = haltere_units[0]
        side = "left" if hunit.side == "L" else "right"
        prefix = "l" if side == "left" else "r"
        haltere_name = f"c_thorax-{prefix}_haltere-pitch"
        for _ in range(8):
            spikes[hunit.body_id] = True
            state = periphery.step(spikes, dt_s=dt_s)
            body.step_muscles(state, physics_steps=10)
            spikes[hunit.body_id] = False
            haltere_peak_torque = max(
                haltere_peak_torque,
                abs(float(body.last_somatic_torque.get(haltere_name, 0.0))),
            )
        if haltere_peak_torque <= 0.0:
            raise RuntimeError(
                f"grounded hDVM MN {hunit.body_id} did not reach haltere torque"
            )

    print(f"whole_body_motor_units={len(periphery.body_ids)}")
    print(f"grounded_somatic_units={len(periphery.units)}")
    print(f"unresolved_somatic_units={len(periphery.excluded)}")
    print(f"tested_leg_mn={unit.body_id} target={unit.target!r}")
    print(f"tested_joint={joint_name} before={before:.6f} after={after:.6f}")
    print(f"tested_leg_peak_torque={peak_torque:.6f}")
    print(f"grounded_haltere_power_units={len(haltere_units)}")
    if haltere_units:
        print(f"tested_haltere_peak_torque={haltere_peak_torque:.6f}")
    print("synthetic_spikes=CALIBRATION_ONLY")
    print("whole_body_motor_boundary=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
