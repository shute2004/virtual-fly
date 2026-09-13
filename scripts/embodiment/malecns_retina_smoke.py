#!/usr/bin/env python3
"""Smoke-test local eye light -> actual MaleCNS R1-R6 current injection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flybody_adapter import FlyBodyWingAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from malecns_retina import MaleCNSRetina
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
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"),
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    course = FlyppyCourse(seed=0, gate_count=2)
    world = FlyppyWorld(course)
    body = FlyBodyWingAdapter(
        tethered=False,
        world=world,
        spawn_position_mm=(0.0, 0.0, 5.0),
        initial_linear_velocity_mm_s=(300.0, 0.0, 0.0),
        enable_vision=True,
    )
    retina = MaleCNSRetina(args.mapping)
    drive = retina.encode(body.sim, body.fly)

    if drive.active_columns <= 0 or drive.active_photoreceptors <= 0:
        raise RuntimeError("retinotopic retina generated no local photoreceptor current")
    body_ids = [body_id for body_id, _ in drive.body_currents]
    currents = np.asarray([current for _, current in drive.body_currents], dtype=np.float64)
    if len(body_ids) != len(set(body_ids)):
        raise RuntimeError("one R1-R6 neuron received current from multiple optic columns")
    if not np.all(np.isfinite(currents)) or np.any(currents <= 0.0):
        raise RuntimeError("photoreceptor currents must be finite and positive")
    if drive.active_photoreceptors < drive.active_columns:
        raise RuntimeError("fewer active R1-R6 neurons than active optic columns")

    # Exercise the exact protocol used by Flyppy: stable MaleCNS body IDs are
    # resolved by the Rust bridge and injected as per-neuron external current.
    # Plasticity is disabled here because this is an interface smoke test.
    with NeuralBridgeClient(
        snapshot=args.snapshot,
        groups=args.groups,
        backend=args.backend,
    ) as brain:
        brain.ping()
        brain.step(
            stimulate_body=drive.body_currents,
            read=(),
            plasticity=False,
        )

    print(f"active_columns={drive.active_columns}")
    print(f"active_r1_r6={drive.active_photoreceptors}")
    print(f"mean_current={drive.mean_current:.6f}")
    print(f"max_current={drive.max_current:.6f}")
    print("external_visual_features=NONE")
    print("direct_body_id_stimulation=PASS")
    print("malecns_retina=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
