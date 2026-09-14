#!/usr/bin/env python3
"""Inspect a completed Flyppy run without advancing the neural simulation.

The diagnostic reads the saved CNS checkpoint and trajectory only. It reports
whether the learning chain actually existed at the end of the run:

    sensory/CNS activity -> eligibility
    DAN activity/connectivity -> postsynaptic modulation
    eligibility + modulation on the same fast synapse -> possible weight update

It also summarizes sampled individual wing-MN activity from trajectory.jsonl.
No state is modified and no learning step is executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_EXPERIMENT = Path("artifacts/experiments/flyppy-v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/malecns-v1.0")
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint directory (default: EXPERIMENT/checkpoint)",
    )
    parser.add_argument("--chunk-edges", type=int, default=1_000_000)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--weight-epsilon", type=float, default=1e-7)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00001,
        help="must match NeuralParams::default().learning_rate for predicted-update diagnostics",
    )
    parser.add_argument("--synapse-scale", type=float, default=0.02)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def memmap_for(directory: Path, filename: object, dtype: str) -> np.memmap:
    return np.memmap(directory / str(filename), dtype=dtype, mode="r")


def summarize_trajectory(path: Path) -> dict[str, object]:
    records = 0
    episodes: set[int] = set()
    sampled_motor_spikes = 0
    max_motor_spikes = 0
    max_active_motor_units = 0
    max_active_photoreceptors = 0
    max_retinal_current = 0.0
    reward_events = 0
    aversive_events = 0
    retinal_after_startup: list[int] = []
    first_record_seen: set[int] = set()
    collision_reason: str | None = None
    collision_x_mm: float | None = None
    collision_z_mm: float | None = None

    if not path.exists():
        return {"present": False}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records += 1
            episode = int(record.get("episode", -1))
            episodes.add(episode)
            motor = record.get("motor_periphery", {}) or {}
            spikes = int(motor.get("spikes", 0))
            active_units = int(motor.get("active_motor_units", 0))
            sampled_motor_spikes += spikes
            max_motor_spikes = max(max_motor_spikes, spikes)
            max_active_motor_units = max(max_active_motor_units, active_units)
            retinal = record.get("retinal_input", {}) or {}
            active_photo = int(retinal.get("active_photoreceptors", 0))
            max_active_photoreceptors = max(max_active_photoreceptors, active_photo)
            max_retinal_current = max(
                max_retinal_current,
                float(retinal.get("max_current", 0.0)),
            )
            if episode in first_record_seen:
                retinal_after_startup.append(active_photo)
            else:
                first_record_seen.add(episode)

            reward_events += int(bool(record.get("reward_stimulated", False)))
            aversive_events += int(bool(record.get("aversive_stimulated", False)))
            if bool(record.get("collision", False)):
                collision_x_mm = float(record.get("x_mm", 0.0))
                collision_z_mm = float(record.get("z_mm", 0.0))
                if collision_z_mm <= 0.65:
                    collision_reason = "floor"
                elif collision_z_mm >= 9.35:
                    collision_reason = "ceiling"
                else:
                    collision_reason = "gate_or_other"

    return {
        "present": True,
        "records": records,
        "episodes": len(episodes),
        "sampled_motor_spikes": sampled_motor_spikes,
        "max_motor_spikes_per_sample": max_motor_spikes,
        "max_active_motor_units": max_active_motor_units,
        "max_active_photoreceptors": max_active_photoreceptors,
        "mean_active_photoreceptors_after_startup": (
            float(np.mean(retinal_after_startup)) if retinal_after_startup else 0.0
        ),
        "max_retinal_current": max_retinal_current,
        "reward_events": reward_events,
        "aversive_events": aversive_events,
        "collision_reason": collision_reason,
        "collision_x_mm": collision_x_mm,
        "collision_z_mm": collision_z_mm,
    }


def main() -> int:
    args = parse_args()
    if args.chunk_edges < 1:
        raise SystemExit("chunk-edges must be >= 1")
    if args.epsilon < 0.0 or args.weight_epsilon < 0.0:
        raise SystemExit("epsilon values must be >= 0")
    if args.learning_rate <= 0.0 or args.synapse_scale <= 0.0:
        raise SystemExit("learning-rate and synapse-scale must be > 0")

    snapshot_manifest = read_manifest(args.snapshot / "manifest.json")
    checkpoint = args.checkpoint or (args.experiment / "checkpoint")
    checkpoint_manifest = read_manifest(checkpoint / "manifest.json")

    print(f"experiment={args.experiment}")
    print(f"checkpoint={checkpoint}")

    neuron_count = int(snapshot_manifest["neuron_count"])
    edge_count = int(snapshot_manifest["edge_count"])
    if int(checkpoint_manifest["neuron_count"]) != neuron_count:
        raise RuntimeError("checkpoint neuron count does not match snapshot")
    if int(checkpoint_manifest["edge_count"]) != edge_count:
        raise RuntimeError("checkpoint edge count does not match snapshot")

    row_offsets = memmap_for(
        args.snapshot, snapshot_manifest["row_offsets_file"], "<u4"
    )
    synapse_counts = memmap_for(
        args.snapshot, snapshot_manifest["synapse_counts_file"], "<u4"
    )
    modulation = memmap_for(
        checkpoint, checkpoint_manifest["modulation_file"], "<f4"
    )
    eligibility = memmap_for(
        checkpoint, checkpoint_manifest["eligibility_file"], "<f4"
    )
    weights = memmap_for(checkpoint, checkpoint_manifest["weights_file"], "<f4")
    spikes = memmap_for(checkpoint, checkpoint_manifest["spikes_file"], "<u4")
    traces = memmap_for(
        checkpoint, checkpoint_manifest["activity_trace_file"], "<f4"
    )

    for name, values, expected in (
        ("row_offsets", row_offsets, neuron_count + 1),
        ("modulation", modulation, neuron_count),
        ("spikes", spikes, neuron_count),
        ("activity_trace", traces, neuron_count),
        ("synapse_counts", synapse_counts, edge_count),
        ("eligibility", eligibility, edge_count),
        ("weights", weights, edge_count),
    ):
        if len(values) != expected:
            raise RuntimeError(f"{name} length={len(values)} expected={expected}")

    eps = float(args.epsilon)
    modulation_abs = np.abs(np.asarray(modulation))
    eligibility_abs = np.abs(np.asarray(eligibility))
    trace_abs = np.abs(np.asarray(traces))

    modulation_nonzero = int(np.count_nonzero(modulation_abs > eps))
    eligibility_nonzero = int(np.count_nonzero(eligibility_abs > eps))
    trace_nonzero = int(np.count_nonzero(trace_abs > eps))
    current_spikes = int(np.count_nonzero(np.asarray(spikes)))

    overlap_edges = 0
    max_abs_overlap_product = 0.0
    predicted_update_over_monitor_threshold = 0
    changed_weights = 0
    max_abs_weight_delta = 0.0

    for start in range(0, edge_count, args.chunk_edges):
        end = min(edge_count, start + args.chunk_edges)
        edge_ids = np.arange(start, end, dtype=np.int64)
        posts = np.searchsorted(row_offsets, edge_ids, side="right") - 1
        edge_eligibility = np.asarray(eligibility[start:end], dtype=np.float32)
        post_modulation = np.asarray(modulation[posts], dtype=np.float32)
        product = post_modulation * edge_eligibility
        abs_product = np.abs(product)
        overlap_edges += int(np.count_nonzero(abs_product > eps))
        if len(abs_product):
            max_abs_overlap_product = max(
                max_abs_overlap_product, float(np.max(abs_product))
            )
        predicted_abs_update = float(args.learning_rate) * abs_product
        predicted_update_over_monitor_threshold += int(
            np.count_nonzero(predicted_abs_update > args.weight_epsilon)
        )

        initial = (
            np.asarray(synapse_counts[start:end], dtype=np.float32)
            * np.float32(args.synapse_scale)
        )
        delta = np.asarray(weights[start:end], dtype=np.float32) - initial
        abs_delta = np.abs(delta)
        changed_weights += int(np.count_nonzero(abs_delta > args.weight_epsilon))
        if len(abs_delta):
            max_abs_weight_delta = max(max_abs_weight_delta, float(np.max(abs_delta)))

    trajectory = summarize_trajectory(args.experiment / "trajectory.jsonl")

    print(f"checkpoint_step={int(checkpoint_manifest['step'])}")
    print(
        "neural_activity current_spikes={} trace_nonzero={} max_trace={:.9g}".format(
            current_spikes,
            trace_nonzero,
            float(np.max(trace_abs)) if len(trace_abs) else 0.0,
        )
    )
    print(
        "dopamine_modulation nonzero_neurons={} max_abs={:.9g}".format(
            modulation_nonzero,
            float(np.max(modulation_abs)) if len(modulation_abs) else 0.0,
        )
    )
    print(
        "eligibility nonzero_edges={} max_abs={:.9g}".format(
            eligibility_nonzero,
            float(np.max(eligibility_abs)) if len(eligibility_abs) else 0.0,
        )
    )
    print(
        "plasticity_overlap edges={} max_abs_modulation_x_eligibility={:.9g} "
        "predicted_updates_gt_{:.1e}={}".format(
            overlap_edges,
            max_abs_overlap_product,
            args.weight_epsilon,
            predicted_update_over_monitor_threshold,
        )
    )
    print(
        "weights changed_gt_{:.1e}={} max_abs_delta={:.9g}".format(
            args.weight_epsilon,
            changed_weights,
            max_abs_weight_delta,
        )
    )

    if trajectory.get("present"):
        print(
            "trajectory episodes={} records={} sampled_motor_spikes={} "
            "max_motor_spikes_per_sample={} max_active_motor_units={} "
            "max_active_photoreceptors={} mean_active_photoreceptors_after_startup={:.3f} "
            "max_retinal_current={:.6g} reward_events={} aversive_events={}".format(
                trajectory["episodes"],
                trajectory["records"],
                trajectory["sampled_motor_spikes"],
                trajectory["max_motor_spikes_per_sample"],
                trajectory["max_active_motor_units"],
                trajectory["max_active_photoreceptors"],
                trajectory["mean_active_photoreceptors_after_startup"],
                trajectory["max_retinal_current"],
                trajectory["reward_events"],
                trajectory["aversive_events"],
            )
        )
        if trajectory.get("collision_reason") is not None:
            print(
                "collision reason={} x_mm={:.3f} z_mm={:.3f}".format(
                    trajectory["collision_reason"],
                    trajectory["collision_x_mm"],
                    trajectory["collision_z_mm"],
                )
            )

    if modulation_nonzero == 0:
        diagnosis = "NO_DOPAMINE_MODULATION"
    elif eligibility_nonzero == 0:
        diagnosis = "NO_ELIGIBILITY"
    elif overlap_edges == 0:
        diagnosis = "NO_MODULATION_ELIGIBILITY_OVERLAP"
    elif changed_weights == 0:
        if predicted_update_over_monitor_threshold == 0:
            diagnosis = "UPDATES_BELOW_MONITOR_THRESHOLD"
        else:
            diagnosis = "OVERLAP_EXISTS_BUT_WEIGHTS_UNCHANGED"
    else:
        diagnosis = "PLASTICITY_ACTIVE"
    print(f"diagnosis={diagnosis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
