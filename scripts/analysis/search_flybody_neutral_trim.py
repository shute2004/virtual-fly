#!/usr/bin/env python3
from __future__ import annotations

import math
import sys

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "scripts/embodiment")
from flybody_measured_wingbeat import AXES, DEFAULT_PATTERN, MeasuredWingbeatCycle
from flybody_v3_muscle_flight_diagnosis import fly_mass_g, make_body, root_reference


SAMPLES = 96
OFFSET_BOUND = 0.18
SCALE_BOUND = 0.12
PHASE_BOUND = 0.30


def evaluate(body, cycle, root_qpos, root_qpos_address, root_qvel_address, params):
    offsets = np.asarray(params[0:3], dtype=np.float64)
    scales = 1.0 + np.asarray(params[3:6], dtype=np.float64)
    phase_shifts = np.asarray(params[6:9], dtype=np.float64)
    samples = []
    for index in range(SAMPLES):
        phase = 2.0 * math.pi * index / SAMPLES
        body.sim.mj_data.qpos[root_qpos_address : root_qpos_address + 7] = root_qpos
        body.sim.mj_data.qvel[root_qvel_address : root_qvel_address + 6] = 0.0
        for side in ("left", "right"):
            for axis_index, axis in enumerate(AXES):
                shifted = phase + float(phase_shifts[axis_index])
                raw = cycle._sample_vector(shifted)
                vel = cycle._derivative_vector(shifted, body.wingbeat_hz)
                target = (
                    float(cycle.mean[axis_index])
                    + float(scales[axis_index])
                    * (float(raw[axis_index]) - float(cycle.mean[axis_index]))
                    + float(offsets[axis_index])
                )
                target_velocity = float(scales[axis_index]) * float(vel[axis_index])
                actuator_index = body._wing_indices[(side, axis)]
                dof = body._actuated_dofs[actuator_index]
                qpos_address, qvel_address = body._wing_state_addresses(dof)
                body.sim.mj_data.qpos[qpos_address] = target
                body.sim.mj_data.qvel[qvel_address] = target_velocity
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
    return np.mean(np.asarray(samples), axis=0)


def main() -> int:
    body = make_body()
    cycle = MeasuredWingbeatCycle(DEFAULT_PATTERN)
    mass_g = fly_mass_g(body)
    weight = mass_g * 9810.0
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    zero = np.zeros(9, dtype=np.float64)
    baseline = evaluate(
        body, cycle, root_qpos, root_qpos_address, root_qvel_address, zero
    )
    baseline_pitch_torque = max(abs(float(baseline[4])), 1e-9)

    def residual(params):
        wrench = evaluate(
            body, cycle, root_qpos, root_qpos_address, root_qvel_address, params
        )
        # Preserve the source baseline translational wrench while removing pitch
        # torque.  The remaining residuals softly prefer the smallest kinematic
        # departure from the measured wingbeat rather than inventing a new gait.
        force = (wrench[:3] - baseline[:3]) / weight
        pitch = np.asarray([wrench[4] / baseline_pitch_torque])
        regularization = np.concatenate(
            (
                np.asarray(params[0:3]) / OFFSET_BOUND,
                np.asarray(params[3:6]) / SCALE_BOUND,
                np.asarray(params[6:9]) / PHASE_BOUND,
            )
        )
        return np.concatenate((3.0 * force, 4.0 * pitch, 0.035 * regularization))

    lower = np.asarray(
        [-OFFSET_BOUND] * 3 + [-SCALE_BOUND] * 3 + [-PHASE_BOUND] * 3,
        dtype=np.float64,
    )
    upper = -lower
    result = least_squares(
        residual,
        zero,
        bounds=(lower, upper),
        max_nfev=180,
        xtol=2e-6,
        ftol=2e-6,
        gtol=2e-6,
        verbose=1,
    )
    fitted = evaluate(
        body,
        cycle,
        root_qpos,
        root_qpos_address,
        root_qvel_address,
        result.x,
    )
    print("baseline_wrench=" + repr(baseline.tolist()))
    print("trimmed_wrench=" + repr(fitted.tolist()))
    print("baseline_force_BW=" + repr((baseline[:3] / weight).tolist()))
    print("trimmed_force_BW=" + repr((fitted[:3] / weight).tolist()))
    print("pitch_torque_ratio=" + repr(float(fitted[4] / baseline_pitch_torque)))
    print("offsets_rad=" + repr(result.x[0:3].tolist()))
    print("amplitude_scales=" + repr((1.0 + result.x[3:6]).tolist()))
    print("phase_shifts_rad=" + repr(result.x[6:9].tolist()))
    print("cost=" + repr(float(result.cost)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
