#!/usr/bin/env python3
"""Smoke-test the individual wing-MN -> peripheral muscle -> FlyBody torque path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from wing_muscle_periphery import WingMusclePeriphery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motor-map",
        type=Path,
        default=Path("artifacts/malecns-v1.0/wing-motor-neurons-v0.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    periphery = WingMusclePeriphery(args.motor_map)
    if periphery.selected_count != 60:
        raise RuntimeError(
            f"expected current MaleCNS boundary to select 60 identified motor units, "
            f"got {periphery.selected_count}"
        )
    if periphery.excluded_count != 7:
        raise RuntimeError(
            f"expected 7 putative/variable/unresolved motor units to remain excluded, "
            f"got {periphery.excluded_count}"
        )

    b1_left = next(
        unit
        for unit in periphery.units
        if unit.side == "left" and unit.target_muscle == "b1"
    )
    spikes = {body_id: False for body_id in periphery.body_ids}
    spikes[b1_left.body_id] = True
    state = periphery.step(spikes, dt_s=5.0e-4)
    if state.muscle("left", "b1") <= 0.0:
        raise RuntimeError("b1 spike did not produce local b1 muscle activation")
    if state.muscle("right", "b1") != 0.0:
        raise RuntimeError("left b1 spike leaked into right b1 muscle state")

    body = FlyBodyMuscleAdapter(
        tethered=True,
        enable_vision=False,
        enable_wing_aerodynamics=False,
    )
    body.step_muscles(state, physics_steps=4)
    torques = np.asarray(list(body.last_wing_torque.values()), dtype=np.float64)
    if torques.shape != (6,) or not np.all(np.isfinite(torques)):
        raise RuntimeError("virtual-muscle torque output is invalid")
    if not np.any(np.abs(torques) > 0.0):
        raise RuntimeError("virtual-muscle boundary produced no wing torque")

    print(f"selected_motor_units={periphery.selected_count}")
    print(f"excluded_motor_units={periphery.excluded_count}")
    print(f"stimulated_body_id={b1_left.body_id}")
    print(f"left_b1_activation={state.muscle('left', 'b1'):.6f}")
    print(
        "wing_torque="
        + ",".join(
            f"{name}:{value:+.6f}"
            for name, value in sorted(body.last_wing_torque.items())
        )
    )
    print("wing_muscle_boundary_smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
