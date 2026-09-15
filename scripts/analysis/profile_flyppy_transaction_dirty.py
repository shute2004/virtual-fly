#!/usr/bin/env python3
"""Measure real Flyppy episode transaction dirtiness without touching production state.

The current P-backed runtime still stores a dense 12-byte transaction transform
for every PlasticFastGraph edge.  Before replacing that with sparse storage we
need the real episode dirty fraction D/P.  This profiler:

* copies the production checkpoint/curriculum into a temporary experiment,
* runs real Flyppy v3 body/retina/physics episodes with population=1,
* reads transaction state only immediately before each commit,
* never writes to the production experiment,
* verifies the production checkpoint digest before/after.

The large transaction readback is deliberately diagnostic-only and is not part
of the intended training hot path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-transaction-dirty/run")
DEFAULT_REPORT = Path("reports/flyppy/transaction_dirty_profile.md")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-control-steps", type=int, default=96)
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


def mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if args.max_control_steps < 1:
        raise SystemExit("--max-control-steps must be >= 1")

    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    checkpoint = production / "checkpoint"
    required = [
        checkpoint / "manifest.json",
        production / "curriculum-state.json",
        ROOT / "artifacts/malecns-v1.0/manifest.json",
        ROOT / "artifacts/malecns-v1.0/embodiment-groups-v0.json",
        ROOT / "artifacts/malecns-v1.0/retinotopic-vision-v1.json",
        ROOT / "artifacts/malecns-v1.0/wing-motor-neurons-v0.json",
        ROOT / "artifacts/malecns-v1.0/body-motor-neurons-v0.json",
        ROOT / DEFAULT_CALIBRATION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing transaction-profile inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    before = tree_digest(checkpoint)
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, temp / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", temp / "curriculum-state.json")
    copy_if_exists(production / "population-state.json", temp / "population-state.json")

    import population_neural_bridge_client as bridge_module
    import train_flyppy_population as trainer

    samples: list[dict[str, object]] = []
    BaseClient = bridge_module.PopulationNeuralBridgeClient

    class ProfilingClient(BaseClient):
        def commit_slot(self, slot: int, *, source_weight_version: int) -> dict:
            stats = self.transaction_stats(slot)
            samples.append(
                {
                    "commit_index": len(samples),
                    "slot": int(slot),
                    "source_weight_version": int(source_weight_version),
                    "plastic_edges": int(stats["plastic_edges"]),
                    "dirty_edges": int(stats["dirty_edges"]),
                    "dirty_fraction": float(stats["dirty_fraction"]),
                    "nonzero_shift_edges": int(stats["nonzero_shift_edges"]),
                    "lower_bound_changed_edges": int(stats["lower_bound_changed_edges"]),
                    "upper_bound_changed_edges": int(stats["upper_bound_changed_edges"]),
                }
            )
            return super().commit_slot(slot, source_weight_version=source_weight_version)

    trainer.PopulationNeuralBridgeClient = ProfilingClient

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(Path(trainer.__file__).resolve()),
            "--episodes",
            str(args.episodes),
            "--population",
            "1",
            "--output-dir",
            str(temp),
            "--max-control-steps",
            str(args.max_control_steps),
            "--checkpoint-every",
            str(args.episodes + 1),
            "--trajectory-stride",
            "1000000",
        ]
        rc = int(trainer.main())
    finally:
        sys.argv = old_argv

    after = tree_digest(checkpoint)
    if before != after:
        raise RuntimeError("production checkpoint changed during transaction profiling")
    if rc != 0:
        raise RuntimeError(f"population trainer returned non-zero status {rc}")
    if len(samples) != args.episodes:
        raise RuntimeError(
            f"expected {args.episodes} transaction samples, observed {len(samples)}"
        )

    summary_path = temp / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    results = list(summary.get("episode_results", []))
    for index, sample in enumerate(samples):
        if index < len(results):
            result = results[index]
            for key in (
                "episode",
                "control_steps",
                "passed_gates",
                "collision",
                "collision_reason",
                "reward_events",
                "aversive_events",
            ):
                if key in result:
                    sample[key] = result[key]

    dirty = [int(item["dirty_edges"]) for item in samples]
    fractions = [float(item["dirty_fraction"]) for item in samples]
    p = int(samples[0]["plastic_edges"]) if samples else 0
    n = int(json.loads((ROOT / "artifacts/malecns-v1.0/manifest.json").read_text())["neuron_count"])

    neuron_local_bytes = 28 * n
    plastic_synapse_bytes = 8 * p
    dirty_bitmap_bytes = (p + 7) // 8
    base_with_bitmap = neuron_local_bytes + plastic_synapse_bytes + dirty_bitmap_bytes
    max_dirty = max(dirty) if dirty else 0
    # Lower-bound compact representation: u32 plastic-edge id + shift/lo/hi.
    # It does not include the lookup structure needed for online repeated updates.
    compact_record_bytes_at_max_d = 16 * max_dirty

    lines = [
        "# Flyppy transaction dirty profile",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- training target: temporary copy of production state",
        "- production checkpoint modified: no",
        f"- production checkpoint digest: `{before}`",
        f"- episodes: {args.episodes}",
        "- population: 1",
        f"- max control steps / episode: {args.max_control_steps}",
        f"- PlasticFastGraph P: {p:,}",
        "",
        "## Per-episode dirty transaction set",
        "",
        "| sample | episode | control steps | dirty edges D | D/P | nonzero shift | lower bound changed | upper bound changed | gates | collision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, item in enumerate(samples):
        lines.append(
            "| {} | {} | {} | {:,} | {:.6%} | {:,} | {:,} | {:,} | {} | {} |".format(
                index,
                item.get("episode", "-"),
                item.get("control_steps", "-"),
                int(item["dirty_edges"]),
                float(item["dirty_fraction"]),
                int(item["nonzero_shift_edges"]),
                int(item["lower_bound_changed_edges"]),
                int(item["upper_bound_changed_edges"]),
                item.get("passed_gates", "-"),
                item.get("collision", "-"),
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- min D/P: {min(fractions):.6%}",
            f"- median D/P: {median(fractions):.6%}",
            f"- mean D/P: {mean(fractions):.6%}",
            f"- max D/P: {max(fractions):.6%}",
            f"- max dirty edges: {max_dirty:,}",
            "",
            "## Memory implication",
            "",
            f"- P weight+eligibility + neuron state + P-bit dirty bitmap: {mib(base_with_bitmap):.2f} MiB / slot",
            f"- 16-byte compact records for the observed maximum D: {mib(compact_record_bytes_at_max_d):.2f} MiB / slot",
            f"- lower-bound combined size at observed maximum D: {mib(base_with_bitmap + compact_record_bytes_at_max_d):.2f} MiB / slot",
            "",
            "The compact-record number is a lower bound, not the final implementation size: an online sparse transaction table also needs a lookup/indexing strategy so repeated deltas for the same edge update the same shift/lo/hi transform. This profile is used to size and choose that structure rather than guessing D/P.",
            "",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"transaction_dirty_profile={report}")
    print(f"production_checkpoint_modified=false")
    print(f"dirty_fraction_mean={mean(fractions):.8f}")
    print(f"dirty_fraction_max={max(fractions):.8f}")
    print(f"max_dirty_edges={max_dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
