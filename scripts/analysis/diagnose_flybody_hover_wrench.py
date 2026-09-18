#!/usr/bin/env python3
from __future__ import annotations

import math
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, "scripts/embodiment")
from flybody_measured_wingbeat import AXES, DEFAULT_PATTERN, MeasuredWingbeatCycle
from flybody_v3_muscle_flight_diagnosis import fly_mass_g, make_body, root_reference


def main() -> int:
    body = make_body()
    cycle = MeasuredWingbeatCycle(DEFAULT_PATTERN)
    mass_g = fly_mass_g(body)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    samples: list[np.ndarray] = []

    for index in range(240):
        phase = 2.0 * math.pi * index / 240.0
        body.sim.mj_data.qpos[root_qpos_address : root_qpos_address + 7] = root_qpos
        body.sim.mj_data.qvel[root_qvel_address : root_qvel_address + 6] = 0.0
        angles = cycle.angles(phase, 1.0)
        velocities = cycle.velocities(phase, 1.0, body.wingbeat_hz)
        for side in ("left", "right"):
            for axis in AXES:
                actuator_index = body._wing_indices[(side, axis)]
                dof = body._actuated_dofs[actuator_index]
                qpos_address, qvel_address = body._wing_state_addresses(dof)
                body.sim.mj_data.qpos[qpos_address] = angles[axis]
                body.sim.mj_data.qvel[qvel_address] = velocities[axis]
        body.sim.mj_data.qfrc_applied[:] = 0.0
        mj.mj_forward(body.sim.mj_model, body.sim.mj_data)
        samples.append(
            np.asarray(
                body.sim.mj_data.qfrc_fluid[
                    root_qvel_address : root_qvel_address + 6
                ],
                dtype=np.float64,
            ).copy()
        )

    array = np.asarray(samples)
    mean = np.mean(array, axis=0)
    rms = np.sqrt(np.mean(array * array, axis=0))
    weight = mass_g * 9810.0
    print(f"mass_g={mass_g:.12g} weight={weight:.12g}")
    print("mean_wrench=" + repr(mean.tolist()))
    print("force_to_BW=" + repr((mean[:3] / weight).tolist()))
    print("mean_torque=" + repr(mean[3:].tolist()))
    print("rms_wrench=" + repr(rms.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
