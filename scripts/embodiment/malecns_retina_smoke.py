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


def read_checkpoint_arrays(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    traces = np.memmap(
        checkpoint / str(manifest["activity_trace_file"]),
        dtype="<f4",
        mode="r",
    )
    events = np.memmap(
        checkpoint / str(manifest["spikes_file"]), dtype="<u4", mode="r"
    )
    return np.asarray(traces), np.asarray(events)


def snapshot_indices_for_body_ids(snapshot: Path, body_ids: list[int]) -> np.ndarray:
    snapshot_body_ids = np.fromfile(snapshot / "body_ids.u64le", dtype="<u8")
    index_by_body = {int(body_id): index for index, body_id in enumerate(snapshot_body_ids)}
    try:
        indices = np.asarray([index_by_body[int(body_id)] for body_id in body_ids], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"R1-R6 body ID missing from snapshot: {exc.args[0]}") from exc
    if len(indices) != len(np.unique(indices)):
        raise RuntimeError("stimulated R1-R6 body IDs do not map one-to-one to snapshot indices")
    return indices


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
    # rather than repeating absolute luminance as DC current.
    adapted_drive = retina.encode(body.sim, body.fly)
    if adapted_drive.active_photoreceptors != 0 or adapted_drive.max_current != 0.0:
        raise RuntimeError(
            "static R1-R6 input did not adapt to its local luminance baseline: "
            f"active={adapted_drive.active_photoreceptors} "
            f"max_current={adapted_drive.max_current}"
        )

    stimulated_indices = snapshot_indices_for_body_ids(args.snapshot, body_ids)

    # Exercise the exact Flyppy body-ID protocol. Save state immediately after the
    # physical light-on pulse and again after the propagation horizon. Propagation
    # is defined only as activity outside the entire directly stimulated R1-R6 set.
    with tempfile.TemporaryDirectory(prefix="virtual-fly-retina-smoke-") as temp_dir:
        initial_checkpoint = Path(temp_dir) / "initial"
        final_checkpoint = Path(temp_dir) / "final"
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
            brain.save_checkpoint(initial_checkpoint)
            for _ in range(args.propagation_steps - 1):
                static_drive = retina.encode(body.sim, body.fly)
                if static_drive.body_currents:
                    raise RuntimeError(
                        "unchanged static retinal sample unexpectedly produced current "
                        "during propagation smoke test"
                    )
                brain.step(
                    stimulate_body=(),
                    read=(),
                    plasticity=False,
                )
            brain.save_checkpoint(final_checkpoint)

        initial_traces, initial_events = read_checkpoint_arrays(initial_checkpoint)
        final_traces, final_events = read_checkpoint_arrays(final_checkpoint)

    if len(initial_traces) != len(final_traces):
        raise RuntimeError("initial/final checkpoint neuron arrays differ in length")

    eps = 1e-12
    initial_event_mask = (initial_events == 1) | (initial_events == 2)
    stimulated_mask = np.zeros(len(initial_traces), dtype=bool)
    stimulated_mask[stimulated_indices] = True
    initial_events_outside_r1_r6 = int(np.count_nonzero(initial_event_mask & ~stimulated_mask))
    if initial_events_outside_r1_r6 != 0:
        raise RuntimeError(
            "first neural step produced activity outside directly stimulated R1-R6; "
            "one-step propagation ordering invariant was violated"
        )

    initial_active_r1_r6 = int(np.count_nonzero(initial_event_mask & stimulated_mask))
    final_trace_mask = np.abs(final_traces) > eps
    downstream_trace_nonzero = int(np.count_nonzero(final_trace_mask & ~stimulated_mask))
    total_trace_nonzero = int(np.count_nonzero(final_trace_mask))
    positive_events = int(np.count_nonzero(final_events == 1))
    hyperpolarizing_events = int(np.count_nonzero(final_events == 2))

    if initial_active_r1_r6 <= 0:
        raise RuntimeError("light-on current did not produce any R1-R6 activity event")
    if downstream_trace_nonzero <= 0:
        raise RuntimeError(
            "retinal transient produced R1-R6 events but no activity trace outside "
            f"the directly stimulated R1-R6 set: initial_events={initial_active_r1_r6}"
        )

    print(f"active_columns={initial_drive.active_columns}")
    print(f"stimulated_r1_r6={initial_drive.active_photoreceptors}")
    print(f"initial_event_r1_r6={initial_active_r1_r6}")
    print(f"mean_current={initial_drive.mean_current:.6f}")
    print(f"max_current={initial_drive.max_current:.6f}")
    print("adapted_static_active_r1_r6=0")
    print("adapted_static_max_current=0.000000")
    print(f"propagation_steps={args.propagation_steps}")
    print(f"nonzero_activity_traces={total_trace_nonzero}")
    print(f"downstream_trace_nonzero={downstream_trace_nonzero}")
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
