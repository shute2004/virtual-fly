#!/usr/bin/env python3
"""Select a robust static virtual-muscle calibration from a recorded motor trace.

This is a body-interface calibration, not a controller. It replays an already
recorded peripheral DLM/DVM/direct-steering history through FlyBody while sweeping
only three calibrated muscle->hinge parameters:

- aggregate power gain;
- aggregate direct-steering gain;
- the reversible DLM/DVM assignment to the abstract hinge half-cycle.

The CNS, visual input, reward and gate coordinates never enter the parameter
selection. Candidates must remain inside the Flyppy vertical corridor with a
safety margin, preserve forward motion, survive an added tail-hold horizon, and
remain viable under small amplitude/asymmetry perturbations. The selected values
are written to JSON and consumed as static parameters by the next training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybody_muscle_adapter import FlyBodyMuscleAdapter
from wing_muscle_periphery import PeripheralSnapshot

ACTIVE_STEERING = frozenset({"b1", "b2", "b3", "i1"})
FLOOR_CENTER_LIMIT_MM = 0.65
CEILING_CENTER_LIMIT_MM = 9.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v1/trajectory.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/flybody-muscle-calibration-v1.json"),
    )
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--initial-forward-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--tail-hold-steps", type=int, default=80)
    parser.add_argument("--minimum-base-clearance-mm", type=float, default=0.75)
    parser.add_argument("--minimum-stress-clearance-mm", type=float, default=0.40)
    parser.add_argument("--minimum-base-max-x-mm", type=float, default=6.50)
    parser.add_argument("--minimum-stress-max-x-mm", type=float, default=5.50)
    parser.add_argument("--minimum-base-final-vx-mm-s", type=float, default=15.0)
    return parser.parse_args()


def load_latest_episode(path: Path) -> tuple[int, list[dict[str, object]]]:
    by_episode: dict[int, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            episode = int(record.get("episode", -1))
            by_episode.setdefault(episode, []).append(record)
    if not by_episode:
        raise RuntimeError(f"trajectory contains no records: {path}")
    episode = max(by_episode)
    records = sorted(by_episode[episode], key=lambda item: int(item["control_step"]))
    actual = [int(item["control_step"]) for item in records]
    if actual != list(range(len(records))):
        raise RuntimeError(
            "motor calibration requires trajectory stride 1 and contiguous control steps"
        )
    return episode, records


def state_from_record(
    record: dict[str, object],
    *,
    power_scale: float = 1.0,
    steering_scale: float = 1.0,
    left_scale: float = 1.0,
    right_scale: float = 1.0,
) -> PeripheralSnapshot:
    motor = dict(record.get("motor_periphery", {}) or {})
    steering_raw = dict(motor.get("steering", {}) or {})
    steering: dict[str, float] = {}
    for key, value in steering_raw.items():
        key = str(key)
        side, _, muscle = key.partition(":")
        if muscle not in ACTIVE_STEERING:
            continue
        side_scale = left_scale if side == "left" else right_scale
        steering[key] = float(np.clip(float(value) * steering_scale * side_scale, 0.0, 1.0))

    return PeripheralSnapshot(
        activation_by_body={},
        muscle_activation=steering,
        dlm_activation={
            "left": float(np.clip(float(motor.get("dlm_left", 0.0)) * power_scale * left_scale, 0.0, 1.0)),
            "right": float(np.clip(float(motor.get("dlm_right", 0.0)) * power_scale * right_scale, 0.0, 1.0)),
        },
        dvm_activation={
            "left": float(np.clip(float(motor.get("dvm_left", 0.0)) * power_scale * left_scale, 0.0, 1.0)),
            "right": float(np.clip(float(motor.get("dvm_right", 0.0)) * power_scale * right_scale, 0.0, 1.0)),
        },
        active_spikes=int(motor.get("spikes", 0)),
        selected_motor_units=1,
    )


class Harness:
    def __init__(self, *, physics_steps: int, initial_speed: float) -> None:
        # No Flyppy gates are present here: this calibration measures only whether
        # the muscle->body seam can sustain a safe free-flight operating envelope.
        self.body = FlyBodyMuscleAdapter(
            tethered=False,
            spawn_position_mm=(0.0, 0.0, 5.0),
            initial_linear_velocity_mm_s=(initial_speed, 0.0, 0.0),
            enable_vision=False,
            # Explicit values below are overwritten for every candidate. Keeping
            # them explicit also makes this search independent of adapter defaults.
            power_gain=1.0,
            steering_gain=1.0,
            swap_dlm_dvm_phase=False,
        )
        self.physics_steps = physics_steps

    def run(
        self,
        records: list[dict[str, object]],
        *,
        power_gain: float,
        steering_gain: float,
        swap_phase: bool,
        tail_hold_steps: int,
        power_state_scale: float = 1.0,
        steering_state_scale: float = 1.0,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ) -> dict[str, float | bool]:
        self.body.reset()
        self.body.power_gain = float(power_gain)
        self.body.steering_gain = float(steering_gain)
        self.body.swap_dlm_dvm_phase = bool(swap_phase)

        states = [
            state_from_record(
                record,
                power_scale=power_state_scale,
                steering_scale=steering_state_scale,
                left_scale=left_scale,
                right_scale=right_scale,
            )
            for record in records
        ]
        if tail_hold_steps > 0:
            states.extend([states[-1]] * tail_hold_steps)

        min_z = float("inf")
        max_z = float("-inf")
        max_x = float("-inf")
        for state in states:
            self.body.step_muscles(state, physics_steps=self.physics_steps)
            position = self.body.thorax_position_mm()
            x_mm = float(position[0])
            z_mm = float(position[2])
            min_z = min(min_z, z_mm)
            max_z = max(max_z, z_mm)
            max_x = max(max_x, x_mm)
            # Continue the whole horizon even after crossing the nominal course
            # boundary so candidate ranking measures the magnitude of instability.

        final_velocity = self.body.root_linear_velocity_mm_s()
        clearance = min(
            min_z - FLOOR_CENTER_LIMIT_MM,
            CEILING_CENTER_LIMIT_MM - max_z,
        )
        return {
            "min_z_mm": min_z,
            "max_z_mm": max_z,
            "max_x_mm": max_x,
            "final_vx_mm_s": float(final_velocity[0]),
            "clearance_mm": clearance,
        }


def base_viable(sample: dict[str, float | bool], args: argparse.Namespace) -> bool:
    return (
        float(sample["clearance_mm"]) >= args.minimum_base_clearance_mm
        and float(sample["max_x_mm"]) >= args.minimum_base_max_x_mm
        and float(sample["final_vx_mm_s"]) >= args.minimum_base_final_vx_mm_s
    )


def stress_viable(sample: dict[str, float | bool], args: argparse.Namespace) -> bool:
    return (
        float(sample["clearance_mm"]) >= args.minimum_stress_clearance_mm
        and float(sample["max_x_mm"]) >= args.minimum_stress_max_x_mm
        and float(sample["final_vx_mm_s"]) > 0.0
    )


def main() -> int:
    args = parse_args()
    if args.physics_steps < 1 or args.tail_hold_steps < 0:
        raise SystemExit("physics-steps must be >= 1 and tail-hold-steps must be >= 0")
    if not args.trajectory.exists():
        raise SystemExit(f"trajectory not found: {args.trajectory}")

    episode, records = load_latest_episode(args.trajectory)
    harness = Harness(
        physics_steps=args.physics_steps,
        initial_speed=args.initial_forward_speed_mm_s,
    )

    candidates: list[dict[str, object]] = []
    power_gains = np.round(np.arange(0.35, 0.901, 0.05), 2)
    steering_gains = np.round(np.arange(0.20, 0.901, 0.05), 2)
    for swap_phase in (False, True):
        for power_gain in power_gains:
            for steering_gain in steering_gains:
                sample = harness.run(
                    records,
                    power_gain=float(power_gain),
                    steering_gain=float(steering_gain),
                    swap_phase=swap_phase,
                    tail_hold_steps=args.tail_hold_steps,
                )
                if not base_viable(sample, args):
                    continue
                candidates.append(
                    {
                        "power_gain": float(power_gain),
                        "steering_gain": float(steering_gain),
                        "swap_dlm_dvm_phase": swap_phase,
                        "nominal": sample,
                    }
                )

    if not candidates:
        raise RuntimeError(
            "no joint power/steering/phase calibration satisfied the base safety envelope"
        )

    # Prefer nominally well-centered flight, then retain values close to unity so
    # calibration does not suppress CNS output more than necessary.
    candidates.sort(
        key=lambda item: (
            float(item["nominal"]["clearance_mm"]),
            float(item["nominal"]["max_x_mm"]),
            float(item["nominal"]["final_vx_mm_s"]),
            -abs(1.0 - float(item["power_gain"]))
            - abs(1.0 - float(item["steering_gain"])),
        ),
        reverse=True,
    )

    stresses = (
        ("nominal", 1.0, 1.0, 1.0, 1.0),
        ("power_minus_10pct", 0.9, 1.0, 1.0, 1.0),
        ("power_plus_10pct", 1.1, 1.0, 1.0, 1.0),
        ("steering_minus_10pct", 1.0, 0.9, 1.0, 1.0),
        ("steering_plus_10pct", 1.0, 1.1, 1.0, 1.0),
        ("left_plus_right_minus_5pct", 1.0, 1.0, 1.05, 0.95),
        ("left_minus_right_plus_5pct", 1.0, 1.0, 0.95, 1.05),
    )

    selected: dict[str, object] | None = None
    robust_results: list[dict[str, object]] = []
    # Stress-test the best nominal candidates first. If necessary continue through
    # every base-viable candidate, so the search still fails closed rather than
    # silently accepting a marginal operating point.
    for candidate in candidates:
        stress_results: dict[str, object] = {}
        all_stable = True
        for name, power_state, steering_state, left_scale, right_scale in stresses:
            sample = harness.run(
                records,
                power_gain=float(candidate["power_gain"]),
                steering_gain=float(candidate["steering_gain"]),
                swap_phase=bool(candidate["swap_dlm_dvm_phase"]),
                tail_hold_steps=args.tail_hold_steps,
                power_state_scale=power_state,
                steering_state_scale=steering_state,
                left_scale=left_scale,
                right_scale=right_scale,
            )
            stress_results[name] = sample
            all_stable = all_stable and stress_viable(sample, args)
        min_clearance = min(
            float(sample["clearance_mm"]) for sample in stress_results.values()
        )
        min_max_x = min(float(sample["max_x_mm"]) for sample in stress_results.values())
        min_final_vx = min(
            float(sample["final_vx_mm_s"]) for sample in stress_results.values()
        )
        robust_results.append(
            {
                **candidate,
                "all_stresses_stable": all_stable,
                "minimum_stress_clearance_mm": min_clearance,
                "minimum_stress_max_x_mm": min_max_x,
                "minimum_stress_final_vx_mm_s": min_final_vx,
                "stresses": stress_results,
            }
        )

    stable = [item for item in robust_results if bool(item["all_stresses_stable"])]
    if not stable:
        raise RuntimeError(
            "base-viable motor calibrations exist, but none survived the robustness envelope"
        )
    stable.sort(
        key=lambda item: (
            float(item["minimum_stress_clearance_mm"]),
            float(item["minimum_stress_max_x_mm"]),
            float(item["minimum_stress_final_vx_mm_s"]),
            -abs(1.0 - float(item["power_gain"]))
            - abs(1.0 - float(item["steering_gain"])),
        ),
        reverse=True,
    )
    selected = stable[0]

    result = {
        "schema_version": 1,
        "source_trajectory": str(args.trajectory),
        "source_episode": episode,
        "source_records": len(records),
        "tail_hold_steps": args.tail_hold_steps,
        "selection_scope": (
            "static muscle-to-hinge calibration only; no CNS state, visual feature, reward, "
            "or per-step action selection is used"
        ),
        "power_gain": selected["power_gain"],
        "steering_gain": selected["steering_gain"],
        "swap_dlm_dvm_phase": selected["swap_dlm_dvm_phase"],
        "nominal": selected["nominal"],
        "minimum_stress_clearance_mm": selected["minimum_stress_clearance_mm"],
        "minimum_stress_max_x_mm": selected["minimum_stress_max_x_mm"],
        "minimum_stress_final_vx_mm_s": selected["minimum_stress_final_vx_mm_s"],
        "stresses": selected["stresses"],
        "base_viable_candidates": len(candidates),
        "robust_candidates": len(stable),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "selected power_gain={:.3f} steering_gain={:.3f} swap_dlm_dvm_phase={}".format(
            float(result["power_gain"]),
            float(result["steering_gain"]),
            result["swap_dlm_dvm_phase"],
        )
    )
    print(
        "robustness base_candidates={} robust_candidates={} nominal_clearance={:.3f} "
        "min_stress_clearance={:.3f} min_stress_max_x={:.3f} min_stress_final_vx={:.3f}".format(
            result["base_viable_candidates"],
            result["robust_candidates"],
            float(result["nominal"]["clearance_mm"]),
            float(result["minimum_stress_clearance_mm"]),
            float(result["minimum_stress_max_x_mm"]),
            float(result["minimum_stress_final_vx_mm_s"]),
        )
    )
    print(f"calibration={args.output}")
    print("motor_interface_calibration=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
