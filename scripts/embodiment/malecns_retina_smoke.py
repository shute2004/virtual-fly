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
        help="one light-on retinal pulse followed by adapted static samples",
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
    initial_drive = retina.encode(body.sim, body.fly)

    if initial_drive.active_columns <= 0 or initial_drive.active_photoreceptors <= 0:
        raise RuntimeError("retinotopic retina generated no light-on photoreceptor current")
    body_ids = [body_id for body_id, _ in initial_drive.body_currents]
    currents = np.asarray(
        [current for _, current in initial_drive.body_currents], dtype=np.float64
    )
    if len(body_ids) != len(set(body_ids)):
        raise RuntimeError("one R1-R6 neuron received current from multiple optic columns")
    if not np.all(np.isfinite(currents)) or np.any(currents <= 0.0):
        raise RuntimeError("initial light-on photoreceptor currents must be finite and positive")
    if initial_drive.active_photoreceptors < initial_drive.active_columns:
        raise RuntimeError("fewer active R1-R6 neurons than active optic columns")

    # With an unchanged static image, the second sample should be locally adapted
    # rather than repeating the absolute luminance as DC current.
    adapted_drive = retina.encode(body.sim, body.fly)
    if adapted_drive.active_photoreceptors != 0 or adapted_drive.max_current != 0.0:
        raise RuntimeError(
            "static R1-R6 input did not adapt to its local luminance baseline: "
            f"active={adapted_drive.active_photoreceptors} "
            f"max_current={adapted_drive.max_current}"
        )

    # Exercise the exact Flyppy body-ID protocol. One physical light-on pulse is
    # followed by locally adapted static samples. Plasticity is disabled: this
    # test asks whether a transient can enter and propagate through the connectome.
    with tempfile.TemporaryDirectory(prefix="virtual-fly-retina-smoke-") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoint"
        with NeuralBridgeClient(
            snapshot=args.snapshot,
            groups=args.groups,
            backend=args.backend,
        ) as brain:
            brain.ping()
            brain.step(
                stimulate_body=initial_drive.body_currents,
                read=(),
                plasticity=False,
            )
            for _ in range(args.propagation_steps - 1):
                static_drive = retina.encode(body.sim, body.fly)
                brain.step(
                    stimulate_body=static_drive.body_currents,
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

    if nonzero_trace <= initial_drive.active_photoreceptors:
        raise RuntimeError(
            "retinal transient reached R1-R6 but did not propagate beyond directly "
            f"stimulated photoreceptors: traces={nonzero_trace} "
            f"r1_r6={initial_drive.active_photoreceptors}"
        )

    print(f"active_columns={initial_drive.active_columns}")
    print(f"active_r1_r6={initial_drive.active_photoreceptors}")
    print(f"mean_current={initial_drive.mean_current:.6f}")
    print(f"max_current={initial_drive.max_current:.6f}")
    print("adapted_static_active_r1_r6=0")
    print("adapted_static_max_current=0.000000")
    print(f"propagation_steps={args.propagation_steps}")
    print(f"nonzero_activity_traces={nonzero_trace}")
    print(f"positive_events_final_step={positive_events}")
    print(f"hyperpolarizing_events_final_step={hyperpolarizing_events}")
    print("external_visual_features=NONE")
    print("local_photoreceptor_adaptation=PASS")
    print("direct_body_id_stimulation=PASS")
    print("released_connectome_propagation=PASS")
    print("malecns_retina=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
