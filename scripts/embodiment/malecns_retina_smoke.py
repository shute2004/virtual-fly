#!/usr/bin/env python3
"""Smoke-test local eye light -> actual MaleCNS R1-R6 -> downstream CNS activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

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
    parser.add_argument(
        "--propagation-steps",
        type=int,
        default=6,
        help="repeat the same local retinal drive long enough to test internal propagation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.propagation_steps < 2:
        raise ValueError("propagation-steps must be >= 2")

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
    # Plasticity is disabled because this is an interface/propagation smoke test.
    with tempfile.TemporaryDirectory(prefix="virtual-fly-retina-smoke-") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoint"
        with NeuralBridgeClient(
            snapshot=args.snapshot,
            groups=args.groups,
            backend=args.backend,
        ) as brain:
            brain.ping()
            for _ in range(args.propagation_steps):
                brain.step(
                    stimulate_body=drive.body_currents,
                    read=(),
                    plasticity=False,
                )
            brain.save_checkpoint(checkpoint)

        manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
        traces = np.memmap(
            checkpoint / str(manifest["activity_trace_file"]),
            dtype="<f4",
            mode="r",
        )
        events = np.memmap(
            checkpoint / str(manifest["spikes_file"]), dtype="<u4", mode="r"
        )
        nonzero_trace = int(np.count_nonzero(np.abs(np.asarray(traces)) > 1e-12))
        positive_events = int(np.count_nonzero(np.asarray(events) == 1))
        hyperpolarizing_events = int(np.count_nonzero(np.asarray(events) == 2))

    # Directly driven R1-R6 account for at most active_photoreceptors traces. A
    # larger count proves that activity crossed at least one released CNS edge.
    if nonzero_trace <= drive.active_photoreceptors:
        raise RuntimeError(
            "retinal current reached R1-R6 but did not propagate beyond directly "
            f"stimulated photoreceptors: traces={nonzero_trace} "
            f"r1_r6={drive.active_photoreceptors}"
        )

    print(f"active_columns={drive.active_columns}")
    print(f"active_r1_r6={drive.active_photoreceptors}")
    print(f"mean_current={drive.mean_current:.6f}")
    print(f"max_current={drive.max_current:.6f}")
    print(f"propagation_steps={args.propagation_steps}")
    print(f"nonzero_activity_traces={nonzero_trace}")
    print(f"positive_events_final_step={positive_events}")
    print(f"hyperpolarizing_events_final_step={hyperpolarizing_events}")
    print("external_visual_features=NONE")
    print("direct_body_id_stimulation=PASS")
    print("released_connectome_propagation=PASS")
    print("malecns_retina=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
