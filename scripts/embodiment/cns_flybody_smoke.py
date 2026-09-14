#!/usr/bin/env python3
"""End-to-end smoke for MaleCNS -> DNg02 readout -> FlyBody wing actuation.

This deliberately stimulates DNg02 directly to verify the brain/body transport
path. It is not the Flyppy learning task and makes no behavioral-learning claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybody_adapter import FlyBodyWingAdapter, WingDrive
from neural_bridge_client import NeuralBridgeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"),
    )
    parser.add_argument("--control-steps", type=int, default=32)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--current", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/cns-flybody-smoke.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.control_steps < 4 or args.physics_steps < 1:
        raise SystemExit("control-steps must be >= 4 and physics-steps >= 1")

    adapter = FlyBodyWingAdapter(tethered=True)
    wing_history: list[dict[str, float]] = []
    drive_history: list[tuple[float, float]] = []

    with NeuralBridgeClient(
        snapshot=args.snapshot,
        groups=args.groups,
        backend="gpu",
    ) as brain:
        brain.ping()
        midpoint = args.control_steps // 2
        for step in range(args.control_steps):
            stimulate = (
                {"flight_thrust_left": args.current}
                if step < midpoint
                else {"flight_thrust_right": args.current}
            )
            read = brain.step(
                stimulate=stimulate,
                read=("flight_thrust_left", "flight_thrust_right"),
                plasticity=False,
            )
            left = float(read["flight_thrust_left"]["spike_fraction"])
            right = float(read["flight_thrust_right"]["spike_fraction"])
            drive_history.append((left, right))
            adapter.step(
                WingDrive(left=left, right=right),
                physics_steps=args.physics_steps,
            )
            wing_history.append(adapter.wing_joint_angles_rad())

        ready = brain.ready

    if not wing_history:
        raise RuntimeError("no FlyBody samples were collected")

    names = sorted(wing_history[0])
    wing_samples = np.asarray(
        [[sample[name] for name in names] for sample in wing_history],
        dtype=np.float64,
    )
    p2p = np.ptp(wing_samples, axis=0)
    drives = np.asarray(drive_history, dtype=np.float64)
    first_half = drives[: len(drives) // 2]
    second_half = drives[len(drives) // 2 :]

    left_response = float(np.mean(first_half[:, 0]))
    right_response = float(np.mean(second_half[:, 1]))
    if left_response <= 0.0 or right_response <= 0.0:
        raise RuntimeError(
            "direct DNg02 stimulation did not produce bilateral spike readout"
        )
    if float(np.max(p2p)) < 1e-3:
        raise RuntimeError("CNS readout did not drive measurable FlyBody wing motion")

    result = {
        "bridge_backend": ready.get("backend"),
        "neurons": int(ready.get("neurons", 0)),
        "edges": int(ready.get("edges", 0)),
        "control_steps": args.control_steps,
        "physics_steps_per_control": args.physics_steps,
        "left_stimulation_mean_spike_fraction": left_response,
        "right_stimulation_mean_spike_fraction": right_response,
        "wing_dofs": names,
        "wing_peak_to_peak_rad": p2p.tolist(),
        "thorax_position_mm": adapter.thorax_position_mm().tolist(),
        "interpretation": (
            "Interface smoke only: direct DNg02 excitation is transported through "
            "the persistent MaleCNS runtime and converted by the motor adapter into "
            "FlyBody wing kinematics. No Flyppy sensory learning is claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"left_response={left_response:.6f}")
    print(f"right_response={right_response:.6f}")
    print(f"max_wing_peak_to_peak_rad={float(np.max(p2p)):.6f}")
    print(f"result={args.output}")
    print("cns_flybody_smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
