#!/usr/bin/env python3
"""Profile the exact live plasticity sets produced by real Flyppy experience.

The actual Flyppy v3 loop runs once against a temporary copy of production
state. A recording client captures the exact CNS stimuli/commit sequence. That
sequence is then replayed from the same starting checkpoint by a diagnostic
frontier runtime which samples internal traces, eligibility and modulation.

This separates several notions that matter for an exact sparse implementation:

* eligibility_before_nonzero: edges an eager implementation must decay,
* local_nonzero: edges receiving a new local eligibility contribution,
* eager_live_union: eligibility_before_nonzero OR local_nonzero,
* delta_nonzero: edges whose weight actually changes this step,
* lazy_immediate_union: local_nonzero OR delta_nonzero.

The last set is the immediate-work target if eligibility decay is represented
lazily while preserving the current numerical model. No production checkpoint
is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-plasticity-live/run")
DEFAULT_REPORT = Path("reports/flyppy/plasticity_live_profile.md")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-control-steps", type=int, default=96)
    parser.add_argument("--sample-stride", type=int, default=12)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


def pct(value: float, denominator: int) -> float:
    return 100.0 * value / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.max_control_steps < 1 or args.sample_stride < 1:
        raise SystemExit("episodes, max-control-steps and sample-stride must be >= 1")

    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    snapshot = absolute(args.snapshot)
    checkpoint = production / "checkpoint"
    groups_path = snapshot / "embodiment-groups-v0.json"
    required = [
        checkpoint / "manifest.json",
        production / "curriculum-state.json",
        snapshot / "manifest.json",
        groups_path,
        snapshot / "retinotopic-vision-v1.json",
        snapshot / "wing-motor-neurons-v0.json",
        snapshot / "body-motor-neurons-v0.json",
        ROOT / DEFAULT_CALIBRATION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing plasticity-live profile inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    body_ids = np.memmap(snapshot / manifest["body_ids_file"], dtype="<u8", mode="r")
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}
    groups_payload = json.loads(groups_path.read_text(encoding="utf-8"))["groups"]
    group_indices = {
        name: tuple(body_to_index[int(body_id)] for body_id in payload["body_ids"])
        for name, payload in groups_payload.items()
    }

    def resolve_stimuli(named, body) -> list[list[float | int]]:
        stimuli: list[list[float | int]] = []
        for name, current in dict(named or {}).items():
            for neuron in group_indices[str(name)]:
                stimuli.append([int(neuron), float(current)])
        for body_id, current in body or ():
            stimuli.append([int(body_to_index[int(body_id)]), float(current)])
        return stimuli

    before = tree_digest(checkpoint)
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    replay_checkpoint = temp / "replay-checkpoint"
    shutil.copytree(checkpoint, replay_checkpoint)
    shutil.copytree(checkpoint, temp / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", temp / "curriculum-state.json")
    copy_if_exists(production / "population-state.json", temp / "population-state.json")

    import population_neural_bridge_client as bridge_module
    import train_flyppy_population as trainer

    BaseClient = bridge_module.PopulationNeuralBridgeClient
    records: list[dict[str, object]] = []
    normal_step_counter = 0

    class RecordingClient(BaseClient):
        def step_batch(self, slot_requests, *, plasticity: bool = True):
            nonlocal normal_step_counter
            if len(slot_requests) != 1 or int(slot_requests[0]["slot"]) != 0:
                raise RuntimeError("plasticity-live profiler requires population=1")
            result = super().step_batch(slot_requests, plasticity=plasticity)
            request = slot_requests[0]
            records.append(
                {
                    "type": "step",
                    "stimuli": resolve_stimuli(
                        request.get("stimulate", {}), request.get("stimulate_body", ())
                    ),
                    "plasticity": bool(plasticity),
                    "steps": 1,
                    "sample": normal_step_counter % args.sample_stride == 0,
                    "normal_step": normal_step_counter,
                }
            )
            normal_step_counter += 1
            return result

        def step_slot(
            self,
            slot: int,
            *,
            stimulate=None,
            stimulate_body=(),
            plasticity: bool = True,
            steps: int = 1,
        ) -> None:
            super().step_slot(
                slot,
                stimulate=stimulate,
                stimulate_body=stimulate_body,
                plasticity=plasticity,
                steps=steps,
            )
            records.append(
                {
                    "type": "step",
                    "stimuli": resolve_stimuli(stimulate or {}, stimulate_body),
                    "plasticity": bool(plasticity),
                    "steps": int(steps),
                    "sample": False,
                    "normal_step": None,
                }
            )

        def commit_slot(self, slot: int, *, source_weight_version: int) -> dict:
            response = super().commit_slot(slot, source_weight_version=source_weight_version)
            records.append({"type": "commit"})
            return response

        def restart_slot(self, slot: int) -> int:
            version = super().restart_slot(slot)
            records.append({"type": "restart"})
            return version

    trainer.PopulationNeuralBridgeClient = RecordingClient

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(Path(trainer.__file__).resolve()),
            "--episodes", str(args.episodes),
            "--population", "1",
            "--snapshot", str(snapshot),
            "--groups", str(groups_path),
            "--retinotopic-map", str(snapshot / "retinotopic-vision-v1.json"),
            "--wing-motor-map", str(snapshot / "wing-motor-neurons-v0.json"),
            "--body-motor-map", str(snapshot / "body-motor-neurons-v0.json"),
            "--output-dir", str(temp),
            "--max-control-steps", str(args.max_control_steps),
            "--checkpoint-every", str(args.episodes + 1),
            "--trajectory-stride", "1000000",
        ]
        rc = int(trainer.main())
    finally:
        sys.argv = old_argv

    after = tree_digest(checkpoint)
    if before != after:
        raise RuntimeError("production checkpoint changed during plasticity-live profiling")
    if rc != 0:
        raise RuntimeError(f"population trainer returned non-zero status {rc}")
    if not records:
        raise RuntimeError("plasticity-live profiler recorded no CNS operations")

    sequence = temp / "cns-sequence.jsonl"
    sequence.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )

    command = [
        "cargo", "run", "-q", "-p", "vf-runner",
        "--bin", "plasticity_live_replay", "--release", "--",
        "--snapshot", str(snapshot),
        "--checkpoint", str(replay_checkpoint),
        "--sequence", str(sequence),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "plasticity live replay failed\nstdout:\n{}\nstderr:\n{}".format(
                completed.stdout, completed.stderr
            )
        )
    payload = json.loads(completed.stdout)
    samples = list(payload.get("samples", []))
    if not samples:
        raise RuntimeError("plasticity live replay produced no samples")

    p = int(payload["plastic_edge_count"])
    metric_names = [
        "eligibility_before_nonzero",
        "local_nonzero",
        "eager_live_union",
        "eligibility_after_nonzero",
        "plastic_edges_under_modulation",
        "delta_nonzero",
        "lazy_immediate_union",
    ]
    metrics = {name: [int(sample[name]) for sample in samples] for name in metric_names}

    lines = [
        "# Flyppy plasticity live-frontier profile",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- overall: PASS",
        "- training target: temporary copy of production state",
        "- production checkpoint modified: no",
        f"- production checkpoint digest: `{before}`",
        f"- N: {int(payload['neuron_count']):,}",
        f"- E: {int(payload['edge_count']):,}",
        f"- P: {p:,}",
        f"- episodes: {args.episodes}",
        f"- max control steps / episode: {args.max_control_steps}",
        f"- sample stride: {args.sample_stride}",
        f"- sampled normal steps: {len(samples)}",
        f"- replayed neural steps: {int(payload['replayed_neural_steps']):,}",
        "",
        "## Aggregate live-set density",
        "",
    ]
    labels = {
        "eligibility_before_nonzero": "eligibility before != 0",
        "local_nonzero": "new local contribution != 0",
        "eager_live_union": "eager live union",
        "eligibility_after_nonzero": "eligibility after != 0",
        "plastic_edges_under_modulation": "plastic edges under nonzero post modulation",
        "delta_nonzero": "actual weight delta != 0",
        "lazy_immediate_union": "lazy immediate union (local OR delta)",
    }
    for name in metric_names:
        values = metrics[name]
        lines.append(
            f"- {labels[name]} mean: {mean(values):,.1f} ({pct(mean(values), p):.4f}% of P); "
            f"median: {median(values):,.1f}; max: {max(values):,} ({pct(max(values), p):.4f}%)"
        )

    eager_mean = mean(metrics["eager_live_union"])
    lazy_mean = mean(metrics["lazy_immediate_union"])
    lines.extend(
        [
            "",
            "## Dense-P replacement signal",
            "",
            f"- current plasticity dispatch proxy: {p:,} edges / neural step",
            f"- eager active-set proxy mean: {eager_mean:,.1f} ({eager_mean / p:.6f} of dense P; {p / eager_mean:.3f}x reduction)" if eager_mean else "- eager active-set proxy mean: 0",
            f"- lazy immediate-work proxy mean: {lazy_mean:,.1f} ({lazy_mean / p:.6f} of dense P; {p / lazy_mean:.3f}x reduction)" if lazy_mean else "- lazy immediate-work proxy mean: 0",
            "",
            "`eager live union` is the set an implementation must touch if eligibility decay remains explicit every step. `lazy immediate union` is the stronger optimization target if the geometric decay of untouched eligibility is deferred and reconstructed exactly when a new local contribution or nonzero modulation makes that edge causally relevant again.",
            "",
            "## Samples",
            "",
            "| sample | normal step | eligibility before | local | eager union | eligibility after | edges under modulation | delta | lazy union |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in samples:
        lines.append(
            "| {sample} | {normal_step} | {eligibility_before_nonzero:,} | {local_nonzero:,} | {eager_live_union:,} | {eligibility_after_nonzero:,} | {plastic_edges_under_modulation:,} | {delta_nonzero:,} | {lazy_immediate_union:,} |".format(**item)
        )
    lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"plasticity_live_profile={report}")
    print("production_checkpoint_modified=false")
    print(f"samples={len(samples)}")
    print(f"eager_live_fraction_mean={eager_mean / p:.8f}")
    print(f"lazy_immediate_fraction_mean={lazy_mean / p:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
