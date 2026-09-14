#!/usr/bin/env python3
"""Hard capability gate for the production Flyppy-v3 virtual-muscle seam.

The source-equivalent body/fluid port is checked separately against upstream
FlyBody.  This gate therefore asks a narrower question: after replacing the
upstream position/force action seam with the MaleCNS -> peripheral muscle ->
physical torque seam, does the production v3 controller preserve enough of the
source controller's vertical aerodynamic support?

The root is re-anchored only for this capability measurement so open-loop attitude
instability cannot masquerade as missing aerodynamic authority.  No neural state,
policy, reward, target observation, or gain fitting is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flybody_measured_wingbeat import DEFAULT_PATTERN, SOURCE_CONTROL_TIMESTEP_S, MeasuredWingbeatCycle
from flybody_v3_adapter import FlyBodyV3MuscleAdapter
from flybody_v3_muscle_flight_diagnosis import run_source_force, run_virtual_muscle


MIN_ABSOLUTE_VERTICAL_SUPPORT = 0.95
MIN_SOURCE_SUPPORT_FRACTION = 0.90
MAX_POSITION_ACTUATOR_DELTA = 0.10
MAX_TRACKING_DEGRADATION = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--warmup-wingbeats", type=float, default=2.0)
    parser.add_argument("--measure-wingbeats", type=float, default=6.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/embodiment/flybody-v3-production-muscle-capability.json"
        ),
    )
    return parser.parse_args()


def tracking(row: dict[str, object]) -> float:
    return float(row["tracking"]["max_normalized_mean_abs_error"])


def main() -> int:
    args = parse_args()
    if not args.pattern.exists():
        raise SystemExit(
            f"measured wing pattern missing: {args.pattern}; run "
            "uv run python scripts/dev/prefetch_flybody_flight_data.py"
        )

    # Fail closed if production v3 silently drifts away from the source flight
    # command cadence or reintroduces the provisional derivative term.
    probe = FlyBodyV3MuscleAdapter(
        tethered=False,
        spawn_position_mm=(0.0, 0.0, 8.91),
        initial_linear_velocity_mm_s=(0.0, 0.0, 0.0),
        enable_vision=False,
        enable_wing_aerodynamics=True,
    )
    if abs(probe.virtual_control_timestep_s - SOURCE_CONTROL_TIMESTEP_S) > 1e-12:
        raise RuntimeError(
            "Flyppy v3 production muscle control cadence drifted from upstream "
            f"FlyBody: {probe.virtual_control_timestep_s} vs {SOURCE_CONTROL_TIMESTEP_S}"
        )
    if abs(probe.virtual_power_kd) > 1e-12:
        raise RuntimeError(
            "Flyppy v3 production virtual-muscle seam reintroduced a derivative "
            f"term: kd={probe.virtual_power_kd}"
        )

    cycle = MeasuredWingbeatCycle(args.pattern)
    source = run_source_force(
        cycle,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    production = run_virtual_muscle(
        cycle,
        disable_position=False,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )
    production_no_position = run_virtual_muscle(
        cycle,
        disable_position=True,
        warmup_wingbeats=args.warmup_wingbeats,
        measure_wingbeats=args.measure_wingbeats,
    )

    source_support = float(source["vertical_support_ratio_to_weight"])
    production_support = float(production["vertical_support_ratio_to_weight"])
    no_position_support = float(
        production_no_position["vertical_support_ratio_to_weight"]
    )
    source_fraction = production_support / source_support
    position_delta = no_position_support - production_support
    tracking_delta = tracking(production) - tracking(source)

    failures: list[str] = []
    if source_support < MIN_ABSOLUTE_VERTICAL_SUPPORT:
        failures.append(
            f"source-force support {source_support:.3f} BW is below "
            f"{MIN_ABSOLUTE_VERTICAL_SUPPORT:.3f} BW"
        )
    if production_support < MIN_ABSOLUTE_VERTICAL_SUPPORT:
        failures.append(
            f"production support {production_support:.3f} BW is below "
            f"{MIN_ABSOLUTE_VERTICAL_SUPPORT:.3f} BW"
        )
    if source_fraction < MIN_SOURCE_SUPPORT_FRACTION:
        failures.append(
            f"production/source support fraction {source_fraction:.3f} is below "
            f"{MIN_SOURCE_SUPPORT_FRACTION:.3f}"
        )
    if position_delta > MAX_POSITION_ACTUATOR_DELTA:
        failures.append(
            f"disabling legacy position actuators adds {position_delta:.3f} BW, "
            "indicating mechanical interference"
        )
    if tracking_delta > MAX_TRACKING_DEGRADATION:
        failures.append(
            f"production tracking NMAE exceeds source by {tracking_delta:.3f}, "
            f"above {MAX_TRACKING_DEGRADATION:.3f}"
        )

    passed = not failures
    result = {
        "schema_version": 1,
        "flight_physics_version": "v3-source-equivalent",
        "production_control_timestep_s": probe.virtual_control_timestep_s,
        "production_virtual_power_kd": probe.virtual_power_kd,
        "source_force": source,
        "production_virtual_muscle": production,
        "production_virtual_muscle_no_position": production_no_position,
        "metrics": {
            "source_vertical_support_bw": source_support,
            "production_vertical_support_bw": production_support,
            "production_support_fraction_of_source": source_fraction,
            "no_position_minus_production_support_bw": position_delta,
            "production_tracking_delta_from_source": tracking_delta,
        },
        "thresholds": {
            "min_absolute_vertical_support_bw": MIN_ABSOLUTE_VERTICAL_SUPPORT,
            "min_source_support_fraction": MIN_SOURCE_SUPPORT_FRACTION,
            "max_position_actuator_delta_bw": MAX_POSITION_ACTUATOR_DELTA,
            "max_tracking_degradation": MAX_TRACKING_DEGRADATION,
        },
        "passed": passed,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "v3_production_muscle source_vertical={:+.3f} production_vertical={:+.3f} "
        "source_fraction={:.3f} tracking_delta={:+.4f} position_delta={:+.3f} "
        "control_update_us={:.1f} kd={:.3f}".format(
            source_support,
            production_support,
            source_fraction,
            tracking_delta,
            position_delta,
            probe.virtual_control_timestep_s * 1e6,
            probe.virtual_power_kd,
        )
    )
    print(f"flybody_v3_production_muscle_capability={'PASS' if passed else 'FAIL'}")
    print(f"result={args.output}")
    if failures:
        for failure in failures:
            print(f"failure={failure}")
        raise RuntimeError(
            "Flyppy v3 production virtual-muscle capability is insufficient; "
            "do not start neural training"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
