#!/usr/bin/env python3
"""Calibrate whole-CNS fast-synapse scale from task-independent pulse stability.

The calibration uses one physical FlyBody compound-eye light-on sample only to
identify actual R1-R6 body currents. For every candidate fast-synapse scale, the
Rust probe applies that pulse once and then removes *all* external input. A usable
bootstrap regime must satisfy both:

1. activity propagates after the directly stimulated R1-R6 step; and
2. recurrent activity returns exactly to silence in the final four zero-input
   steps, because this bootstrap model contains no intrinsic baseline/noise source.

No Flyppy gate, reward, motor output, collision, or task score enters selection.
The selected scale is the middle of the contiguous stable candidate range rather
than its largest edge, giving margin from the self-sustaining transition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from flybody_adapter import FlyBodyWingAdapter
from flyppy_course import FlyppyCourse
from flyppy_world import FlyppyWorld
from malecns_retina import MaleCNSRetina


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("artifacts/malecns-v1.0/retinotopic-vision-v1.json"),
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--zero-input-steps", type=int, default=24)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(
            0.00025,
            0.0005,
            0.001,
            0.002,
            0.004,
            0.006,
            0.008,
            0.012,
            0.016,
            0.020,
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodiment/neural-runtime-calibration-v1.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.zero_input_steps < 4:
        raise SystemExit("zero-input-steps must be >= 4")
    if not args.scales or any(
        not math.isfinite(value) or value <= 0.0 for value in args.scales
    ):
        raise SystemExit("scales must be finite and positive")

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
    if drive.active_photoreceptors <= 0:
        raise RuntimeError("physical light-on sample produced no R1-R6 current")

    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="virtual-fly-neural-stability-") as temp:
        stimuli_path = Path(temp) / "stimuli.json"
        stimuli_path.write_text(
            json.dumps(
                [
                    {"body_id": int(body_id), "current": float(current)}
                    for body_id, current in drive.body_currents
                ]
            ),
            encoding="utf-8",
        )
        command = [
            "cargo",
            "run",
            "-q",
            "-p",
            "vf-runner",
            "--bin",
            "neural_stability_probe",
            "--release",
            "--",
            "--snapshot",
            str(args.snapshot),
            "--stimuli-json",
            str(stimuli_path),
            "--backend",
            args.backend,
            "--zero-input-steps",
            str(args.zero_input_steps),
            "--synapse-scales",
            ",".join(f"{value:.9g}" for value in sorted(set(args.scales))),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
    probe = json.loads(completed.stdout)

    results = sorted(probe["results"], key=lambda item: float(item["synapse_scale"]))
    stable = [
        item
        for item in results
        if bool(item["propagated_after_input"]) and bool(item["zero_tail"])
    ]
    if not stable:
        print("neural_stability_candidates:")
        for item in results:
            print(
                "  scale={:.6g} propagated={} zero_tail={} peak={} final={}".format(
                    float(item["synapse_scale"]),
                    item["propagated_after_input"],
                    item["zero_tail"],
                    item["peak_total_events"],
                    item["final_total_events"],
                )
            )
        raise RuntimeError(
            "no tested fast-synapse scale both propagated the retinal pulse and "
            "returned to zero-input silence; the bootstrap neuron dynamics need "
            "structural revision rather than another Flyppy/body calibration"
        )

    # Split stable candidates into contiguous runs in the tested scale ordering.
    stable_scales = {float(item["synapse_scale"]) for item in stable}
    runs: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for item in results:
        if float(item["synapse_scale"]) in stable_scales:
            current.append(item)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    widest = max(runs, key=lambda run: (len(run), float(run[-1]["synapse_scale"])))
    selected = widest[len(widest) // 2]

    selected_scale = float(selected["synapse_scale"])
    result = {
        "schema_version": 1,
        "selection_basis": (
            "one physical R1-R6 light-on pulse followed by zero external input; "
            "requires downstream propagation and four final silent steps; no task score"
        ),
        "stimulated_r1_r6": drive.active_photoreceptors,
        "neuron_count": int(probe["neuron_count"]),
        "edge_count": int(probe["edge_count"]),
        "zero_input_steps": int(probe["zero_input_steps"]),
        "synapse_scale": selected_scale,
        "stable_scales": [float(item["synapse_scale"]) for item in stable],
        "selected_stable_run": [float(item["synapse_scale"]) for item in widest],
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "neural_stability stimulated_r1_r6={} neurons={} edges={}".format(
            drive.active_photoreceptors,
            result["neuron_count"],
            result["edge_count"],
        )
    )
    for item in results:
        print(
            "scale={:.6g} propagated={} zero_tail={} peak_events={} final_events={}".format(
                float(item["synapse_scale"]),
                item["propagated_after_input"],
                item["zero_tail"],
                item["peak_total_events"],
                item["final_total_events"],
            )
        )
    print(f"stable_scales={result['stable_scales']}")
    print(f"selected_synapse_scale={selected_scale:.9g}")
    print(f"calibration={args.output}")
    print("neural_runtime_stability=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
