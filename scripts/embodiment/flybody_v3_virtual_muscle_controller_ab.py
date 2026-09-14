#!/usr/bin/env python3
"""A/B Flyppy-v3 virtual-muscle controller semantics against source-force flight.

The v3 body/fluid port is already source-equivalent under exact measured kinematics.
This diagnostic isolates why the production virtual-muscle seam produces less
vertical support than the upstream-style source-force controller despite similar
wing tracking error.

Compared modes:

* source_force: upstream-style 0.2 ms held proportional force controller.
* virtual_default: production virtual muscle controller (P + D at 50 us).
* virtual_kd0: production controller with only the D term removed.
* virtual_kd0_source_rate: P-only virtual muscle torque recomputed every 0.2 ms
  and held across four physics steps.

The root is re-anchored between physics steps so attitude/fall does not confound
wing-force capability. No CNS, reward, task observation, policy, or gain fitting is
used. This script changes no production parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flybody_measured_wingbeat import (
    AXES,
    DEFAULT_PATTERN,
    SOURCE_CONTROL_TIMESTEP_S,
    MeasuredWingbeatCycle,
)
from flybody_reference_lift_calibration import disable_position_wing_actuators
from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flybody_v3_muscle_flight_diagnosis import (
    anchor_root,
    collect_tracking,
    fly_mass_g,
    fluid_vector,
    root_reference,
    run_source_force,
    summarize_fluid,
    tracking_summary,
)
from flybody_vertical_flight_capability import power_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--warmup-wingbeats", type=float, default=2.0)
    parser.add_argument("--measure-wingbeats", type=float, default=6.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/embodiment/flybody-v3-virtual-muscle-controller-ab.json"
        ),
    )
    return parser.parse_args()


def make_body(*, virtual_power_kd: float) -> FlyBodyV3MuscleAdapter:
    return FlyBodyV3MuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 8.91),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
        virtual_power_kd=float(virtual_power_kd),
    )


def _measurement_schedule(body, warmup_wingbeats: float, measure_wingbeats: float):
    wing_period_s = 1.0 / body.wingbeat_hz
    warmup_s = warmup_wingbeats * wing_period_s
    total_s = (warmup_wingbeats + measure_wingbeats) * wing_period_s
    steps = max(1, math.ceil(total_s / body.timestep))
    return warmup_s, steps


def run_step_muscles(
    cycle: MeasuredWingbeatCycle,
    *,
    virtual_power_kd: float,
    mode: str,
    warmup_wingbeats: float,
    measure_wingbeats: float,
) -> dict[str, object]:
    body = make_body(virtual_power_kd=virtual_power_kd)
    mass_g = fly_mass_g(body)
    state = power_state(1.0, 1.0)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    warmup_s, steps = _measurement_schedule(
        body, warmup_wingbeats, measure_wingbeats
    )

    vectors: list[np.ndarray] = []
    errors: dict[str, list[float]] = {axis: [] for axis in AXES}
    peak_torque = 0.0

    for _ in range(steps):
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        body.step_muscles(state, physics_steps=1)
        peak_torque = max(
            peak_torque,
            max(abs(float(value)) for value in body.last_wing_torque.values()),
        )
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        if body._time < warmup_s:
            continue
        vectors.append(fluid_vector(body, root_qvel_address))
        phase = (2.0 * math.pi * body.wingbeat_hz * body._time) % (2.0 * math.pi)
        collect_tracking(body, cycle.angles(phase, 1.0), errors)

    result = summarize_fluid(vectors, mass_g=mass_g)
    result.update(
        {
            "mode": mode,
            "virtual_power_kd": float(virtual_power_kd),
            "control_update_s": body.timestep,
            "body_mass_mg": mass_g * 1000.0,
            "peak_abs_virtual_muscle_torque": peak_torque,
            "tracking": tracking_summary(cycle, errors),
        }
    )
    return result


def _compute_p_only_torque(body, phase: float) -> dict[int, float]:
    target = body._wing_pattern(phase, 1.0)
    command: dict[int, float] = {}
    for side in ("left", "right"):
        for axis in AXES:
            actuator_index = body._wing_indices[(side, axis)]
            dof = body._actuated_dofs[actuator_index]
            qpos_address, qvel_address = body._wing_state_addresses(dof)
            angle = float(body.sim.mj_data.qpos[qpos_address])
            torque = body.virtual_power_kp * (float(target[axis]) - angle)
            torque = min(body.max_abs_torque, max(-body.max_abs_torque, torque))
            command[qvel_address] = torque
    return command


def run_p_only_source_rate(
    cycle: MeasuredWingbeatCycle,
    *,
    warmup_wingbeats: float,
    measure_wingbeats: float,
) -> dict[str, object]:
    body = make_body(virtual_power_kd=0.0)
    disable_position_wing_actuators(body)
    mass_g = fly_mass_g(body)
    root_qpos, root_qpos_address, root_qvel_address = root_reference(body)
    warmup_s, steps = _measurement_schedule(
        body, warmup_wingbeats, measure_wingbeats
    )

    ratio = SOURCE_CONTROL_TIMESTEP_S / body.timestep
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("source control timestep is not an integer physics multiple")
    control_stride = int(round(ratio))

    held_command: dict[int, float] = {}
    vectors: list[np.ndarray] = []
    errors: dict[str, list[float]] = {axis: [] for axis in AXES}
    peak_torque = 0.0

    for physics_index in range(steps):
        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        if physics_index % control_stride == 0:
            phase = (2.0 * math.pi * body.wingbeat_hz * body._time) % (
                2.0 * math.pi
            )
            held_command = _compute_p_only_torque(body, phase)

        body.sim.mj_data.qfrc_applied[:] = 0.0
        for qvel_address, torque in held_command.items():
            body.sim.mj_data.qfrc_applied[qvel_address] += torque
            peak_torque = max(peak_torque, abs(float(torque)))
        body.sim.step()
        body.sim.mj_data.qfrc_applied[:] = 0.0
        body._time += body.timestep

        anchor_root(body, root_qpos, root_qpos_address, root_qvel_address)
        if body._time < warmup_s:
            continue
        vectors.append(fluid_vector(body, root_qvel_address))
        phase = (2.0 * math.pi * body.wingbeat_hz * body._time) % (2.0 * math.pi)
        collect_tracking(body, cycle.angles(phase, 1.0), errors)

    result = summarize_fluid(vectors, mass_g=mass_g)
    result.update(
        {
            "mode": "virtual_kd0_source_rate",
            "virtual_power_kd": 0.0,
            "control_update_s": SOURCE_CONTROL_TIMESTEP_S,
            "body_mass_mg": mass_g * 1000.0,
            "peak_abs_virtual_muscle_torque": peak_torque,
            "tracking": tracking_summary(cycle, errors),
        }
    )
    return result


def _tracking(row: dict[str, object]) -> float:
    return float(row["tracking"]["max_normalized_mean_abs_error"])


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(
            f"measured wing pattern missing: {args.pattern}; run "
            "uv run python scripts/dev/prefetch_flybody_flight_data.py"
        )
    if args.warmup_wingbeats < 0.0 or args.measure_wingbeats <= 0.0:
        raise SystemExit("wingbeat counts are invalid")

    cycle = MeasuredWingbeatCycle(args.pattern)
    source = run_source_force(
        cycle,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    default = run_step_muscles(
        cycle,
        virtual_power_kd=0.08,
        mode="virtual_default",
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    kd0 = run_step_muscles(
        cycle,
        virtual_power_kd=0.0,
        mode="virtual_kd0",
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    source_rate = run_p_only_source_rate(
        cycle,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    rows = [source, default, kd0, source_rate]

    source_support = float(source["vertical_support_ratio_to_weight"])
    default_support = float(default["vertical_support_ratio_to_weight"])
    kd0_support = float(kd0["vertical_support_ratio_to_weight"])
    source_rate_support = float(source_rate["vertical_support_ratio_to_weight"])

    if kd0_support >= default_support + 0.08 and kd0_support >= 0.95:
        diagnosis = "VIRTUAL_MUSCLE_DERIVATIVE_TERM_REDUCES_LIFT"
    elif source_rate_support >= kd0_support + 0.08 and source_rate_support >= 0.95:
        diagnosis = "VIRTUAL_MUSCLE_UPDATE_CADENCE_REDUCES_LIFT"
    elif source_rate_support >= 0.95 and abs(source_rate_support - source_support) <= 0.15:
        diagnosis = "VIRTUAL_MUSCLE_P_ONLY_SOURCE_RATE_RECOVERS_SOURCE_LIFT"
    elif default_support >= 0.95:
        diagnosis = "VIRTUAL_MUSCLE_DEFAULT_ALREADY_SUPPORTS_WEIGHT"
    else:
        diagnosis = "VIRTUAL_MUSCLE_CONTROLLER_DIFFERENCE_NOT_YET_ISOLATED"

    result = {
        "schema_version": 1,
        "flight_physics_version": "v3-source-equivalent",
        "pattern": str(args.pattern),
        "cases": rows,
        "diagnosis": diagnosis,
        "comparisons": {
            "default_support_fraction_of_source": default_support / source_support,
            "kd0_support_fraction_of_source": kd0_support / source_support,
            "source_rate_support_fraction_of_source": source_rate_support / source_support,
            "default_tracking_delta_from_source": _tracking(default) - _tracking(source),
            "kd0_tracking_delta_from_source": _tracking(kd0) - _tracking(source),
            "source_rate_tracking_delta_from_source": _tracking(source_rate) - _tracking(source),
        },
        "interpretation": (
            "Controller-only A/B on the same source-equivalent v3 body. Thresholds "
            "are diagnostic labels only; no production gain is fitted or changed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            "v3_controller_ab mode={} vertical={:+.3f} magnitude={:.3f} "
            "track_nmae_max={:.4f} update_us={:.1f}".format(
                row["mode"],
                float(row["vertical_support_ratio_to_weight"]),
                float(row["mean_force_magnitude_ratio_to_weight"]),
                _tracking(row),
                float(row.get("control_update_s", SOURCE_CONTROL_TIMESTEP_S)) * 1e6,
            )
        )
    print(f"diagnosis={diagnosis}")
    print(f"result={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
