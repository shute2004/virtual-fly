#!/usr/bin/env python3
"""Instantiate FlyGym 2.1 FlyBody and directly exercise its wing DoFs.

This is a body-interface smoke test, not a controller. It verifies that the
selected FlyBody version can be composed, compiled, stepped, and driven through
its six wing position-actuator DoFs without involving any RL policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flygym import Simulation
from flygym.compose import ActuatorType, KinematicPosePreset, TetheredWorld
from flygym.compose.fly import FlyBody
from flygym.flybody.anatomy_flybody import (
    FlyBodyAxisOrder,
    FlyBodyJointPreset,
    FlyBodySkeleton,
)
from flygym.utils.math import Rotation3D

from flybody_flight_physics import (
    FLIGHT_PHYSICS_TIMESTEP_S,
    add_flight_position_actuators,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.04)
    parser.add_argument("--frequency-hz", type=float, default=180.0)
    parser.add_argument("--amplitude-rad", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-wing-smoke.json"),
    )
    return parser.parse_args()


def dof_key(dof) -> tuple[str, str, str]:
    return (dof.parent.name, dof.child.name, dof.axis.value)


def build_simulation() -> tuple[FlyBody, Simulation]:
    fly = FlyBody(name="virtual_fly")
    skeleton = FlyBodySkeleton(
        joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL,
        axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
    )
    fly.add_joints(skeleton, KinematicPosePreset.FLYBODY_NEUTRAL)

    # Use the same six-wing-DOF actuator boundary as the real flight adapter.
    # In particular, do not use FlyGym's ALL preset: it includes non-flight joints
    # such as halteres and would exercise a different actuator topology than Flyppy.
    add_flight_position_actuators(fly, list(skeleton.iter_jointdofs()))
    fly.add_tendons()
    fly.add_tendon_actuators()

    world = TetheredWorld()
    world.add_fly(
        fly,
        (0.0, 0.0, 0.0),
        Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
    )
    return fly, Simulation(world, timestep=FLIGHT_PHYSICS_TIMESTEP_S)


def main() -> int:
    args = parse_args()
    if args.seconds <= 0 or args.frequency_hz <= 0 or args.amplitude_rad <= 0:
        raise SystemExit("seconds, frequency-hz, and amplitude-rad must be positive")

    fly, sim = build_simulation()
    sim.reset()

    all_dofs = list(fly.get_jointdofs_order())
    actuated_dofs = list(fly.get_actuated_jointdofs_order(ActuatorType.POSITION))
    if len(actuated_dofs) != 6 or not all(dof.child.is_wing() for dof in actuated_dofs):
        raise RuntimeError(
            "wing smoke must expose exactly six wing POSITION actuator DoFs, got "
            f"{[dof_key(dof) for dof in actuated_dofs]}"
        )

    all_index = {dof_key(dof): i for i, dof in enumerate(all_dofs)}
    neutral_joint_angles = np.asarray(sim.get_joint_angles(fly.name), dtype=np.float64)
    target = np.asarray(
        [neutral_joint_angles[all_index[dof_key(dof)]] for dof in actuated_dofs],
        dtype=np.float64,
    )

    wing_actuator_indices = np.arange(len(actuated_dofs), dtype=np.int64)

    # Use opposite phases on left/right stroke-related axes. This is only an
    # interface excitation pattern; it is not claimed to be a biological motor program.
    wing_phase = np.ones(len(wing_actuator_indices), dtype=np.float64)
    wing_names = []
    for local_i, actuator_i in enumerate(wing_actuator_indices):
        dof = actuated_dofs[int(actuator_i)]
        wing_names.append("|".join(dof_key(dof)))
        if dof.child.name.startswith("r_"):
            wing_phase[local_i] = -1.0

    timestep = float(sim.mj_model.opt.timestep)
    n_steps = max(2, math.ceil(args.seconds / timestep))
    samples = np.empty((n_steps, len(wing_actuator_indices)), dtype=np.float64)

    for step in range(n_steps):
        t = step * timestep
        command = target.copy()
        oscillation = args.amplitude_rad * math.sin(2.0 * math.pi * args.frequency_hz * t)
        command[wing_actuator_indices] += wing_phase * oscillation
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, command)
        sim.step()

        joint_angles = np.asarray(sim.get_joint_angles(fly.name), dtype=np.float64)
        for local_i, actuator_i in enumerate(wing_actuator_indices):
            dof = actuated_dofs[int(actuator_i)]
            samples[step, local_i] = joint_angles[all_index[dof_key(dof)]]

    peak_to_peak = np.ptp(samples, axis=0)
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("FlyBody produced non-finite wing joint state")
    if float(np.max(peak_to_peak)) < 1e-3:
        raise RuntimeError("wing actuator commands did not move any wing DoF")

    result = {
        "flygym_version_target": "2.1.0",
        "timestep_seconds": timestep,
        "simulated_seconds": n_steps * timestep,
        "steps": n_steps,
        "frequency_hz": args.frequency_hz,
        "amplitude_rad": args.amplitude_rad,
        "wing_dofs": wing_names,
        "wing_peak_to_peak_rad": peak_to_peak.tolist(),
        "max_peak_to_peak_rad": float(np.max(peak_to_peak)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wing_dofs={len(wing_actuator_indices)}")
    print(f"max_peak_to_peak_rad={result['max_peak_to_peak_rad']:.6f}")
    print(f"result={args.output}")
    print("flybody_wing_smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
