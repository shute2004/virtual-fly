#!/usr/bin/env python3
"""Profile real Flyppy propagation-frontier sparsity without changing dynamics.

The profiler copies production state into a temporary experiment and runs the
actual Flyppy v3 body/retina/physics loop with population=1.  On a sparse set of
normal control steps it asks the existing population bridge to return all MaleCNS
events by temporarily expanding read_body to all released body IDs.  The normal
trainer receives only its originally requested motor/viewer subset back.

Frontier metrics are then computed offline from the canonical incoming CSR:

  A      active presynaptic neurons
  F      outgoing edges sourced by A
  C      posts reached by at least one edge in F
  I(C)   canonical incoming edges scanned for those candidate posts

No neural kernel or production checkpoint is modified.  The deliberately large
all-neuron readout exists only on sampled diagnostic steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-frontier-activity/run")
DEFAULT_REPORT = Path("reports/flyppy/frontier_activity_profile.md")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
P = 10_871_322


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


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.max_control_steps < 1 or args.sample_stride < 1:
        raise SystemExit("episodes, max-control-steps, and sample-stride must be >= 1")

    production = absolute(args.production)
    temp = absolute(args.temp)
    report = absolute(args.report)
    snapshot_dir = absolute(args.snapshot)
    checkpoint = production / "checkpoint"
    required = [
        checkpoint / "manifest.json",
        production / "curriculum-state.json",
        snapshot_dir / "manifest.json",
        snapshot_dir / "embodiment-groups-v0.json",
        snapshot_dir / "retinotopic-vision-v1.json",
        snapshot_dir / "wing-motor-neurons-v0.json",
        snapshot_dir / "body-motor-neurons-v0.json",
        ROOT / DEFAULT_CALIBRATION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing frontier-profile inputs:\n  " + "\n  ".join(missing))

    calibration = json.loads((ROOT / DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    n = int(manifest["neuron_count"])
    e = int(manifest["edge_count"])
    body_ids = np.memmap(snapshot_dir / manifest["body_ids_file"], dtype="<u8", mode="r")
    row_offsets = np.memmap(snapshot_dir / manifest["row_offsets_file"], dtype="<u4", mode="r")
    pre_indices = np.memmap(snapshot_dir / manifest["pre_indices_file"], dtype="<u4", mode="r")
    if body_ids.size != n or row_offsets.size != n + 1 or pre_indices.size != e:
        raise RuntimeError("snapshot dimensions do not match manifest")
    all_body_ids = tuple(int(value) for value in body_ids)
    indegree = np.diff(row_offsets.astype(np.int64, copy=False))

    # Reused offline scratch.  This deliberately scans all E on the CPU only
    # for sampled diagnostic states; it is not part of the neural runtime.
    active_pre = np.zeros(n, dtype=np.bool_)
    active_edge = np.empty(e, dtype=np.bool_)
    prefix = np.empty(e + 1, dtype=np.uint32)
    prefix[0] = 0

    before = tree_digest(checkpoint)
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, temp / "checkpoint")
    shutil.copy2(production / "curriculum-state.json", temp / "curriculum-state.json")
    copy_if_exists(production / "population-state.json", temp / "population-state.json")

    import population_neural_bridge_client as bridge_module
    import train_flyppy_population as trainer

    BaseClient = bridge_module.PopulationNeuralBridgeClient
    samples: list[dict[str, float | int]] = []
    normal_step_counter = 0

    def analyze_events(events_by_body: dict[int, bool], neural_step: int, sample_index: int) -> None:
        active_pre.fill(False)
        # The bridge returns bool=True only for depolarizing events.  Motor-facing
        # readout intentionally collapses the signed internal event code, so for
        # frontier profiling we must recover both signs.  Therefore all-body
        # readout alone is insufficient if hyperpolarizing events are present.
        # Refuse to silently under-count; a signed diagnostic bridge is required.
        raise RuntimeError(
            "frontier profiler requires signed all-neuron events; current bridge read_body exposes only depolarizing bool"
        )

    class ProfilingClient(BaseClient):
        def step_batch(self, slot_requests, *, plasticity: bool = True):
            nonlocal normal_step_counter
            should_sample = normal_step_counter % args.sample_stride == 0
            normal_step_counter += 1
            if not should_sample:
                return super().step_batch(slot_requests, plasticity=plasticity)

            expanded = []
            requested_by_slot: dict[int, tuple[int, ...]] = {}
            for request in slot_requests:
                item = dict(request)
                slot = int(item["slot"])
                requested = tuple(int(v) for v in item.get("read_body", ()))
                requested_by_slot[slot] = requested
                item["read_body"] = all_body_ids
                expanded.append(item)
            result = super().step_batch(expanded, plasticity=plasticity)
            full = result.get(0, {})
            analyze_events(full, self.last_step, len(samples))
            return {
                slot: {body_id: bool(result.get(slot, {}).get(body_id, False)) for body_id in requested}
                for slot, requested in requested_by_slot.items()
            }

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
    except RuntimeError as exc:
        # Write an explicit blocked report instead of pretending the bool motor
        # protocol can measure a signed 0/1/2 frontier correctly.
        after = tree_digest(checkpoint)
        lines = [
            "# Flyppy propagation frontier activity profile",
            "",
            f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "- overall: BLOCKED",
            "- production checkpoint modified: " + ("no" if before == after else "YES"),
            f"- reason: {exc}",
            "",
            "The current population bridge converts internal 0/1/2 activity to a boolean motor-facing readout (`event == 1`). Hyperpolarizing events (2) also traverse outgoing edges, so treating the boolean readout as the complete active-presynaptic set would under-count the propagation frontier. The profiler intentionally refuses that invalid measurement.",
            "",
        ]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"frontier_activity_profile={report}")
        print("frontier_activity_profile=BLOCKED_SIGNED_EVENT_API_REQUIRED")
        return 2
    finally:
        sys.argv = old_argv

    after = tree_digest(checkpoint)
    if before != after:
        raise RuntimeError("production checkpoint changed during frontier profiling")
    if rc != 0:
        raise RuntimeError(f"population trainer returned non-zero status {rc}")
    if not samples:
        raise RuntimeError("frontier profiler produced no samples")

    # This path becomes reachable once the bridge exposes signed events.
    active_counts = [int(s["active_pre"]) for s in samples]
    frontier_edges = [int(s["frontier_edges"]) for s in samples]
    candidate_posts = [int(s["candidate_posts"]) for s in samples]
    candidate_incoming = [int(s["candidate_incoming_edges"]) for s in samples]
    current_work = [int(s["frontier_edges"]) + int(s["candidate_incoming_edges"]) + P for s in samples]
    old_work = e + P

    lines = [
        "# Flyppy propagation frontier activity profile",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- overall: PASS",
        "- training target: temporary copy of production state",
        "- production checkpoint modified: no",
        f"- production checkpoint digest: `{before}`",
        f"- N: {n:,}",
        f"- E: {e:,}",
        f"- P: {P:,}",
        f"- samples: {len(samples)}",
        "",
        "## Aggregate frontier density",
        "",
        f"- active pre mean: {mean(active_counts):,.1f} ({pct(mean(active_counts), n):.4f}% of N)",
        f"- active pre max: {max(active_counts):,} ({pct(max(active_counts), n):.4f}% of N)",
        f"- active outgoing F mean: {mean(frontier_edges):,.1f} ({pct(mean(frontier_edges), e):.4f}% of E)",
        f"- candidate posts C mean: {mean(candidate_posts):,.1f} ({pct(mean(candidate_posts), n):.4f}% of N)",
        f"- candidate incoming I(C) mean: {mean(candidate_incoming):,.1f} ({pct(mean(candidate_incoming), e):.4f}% of E)",
        f"- propagation edge-touch mean F+I(C): {mean([a+b for a,b in zip(frontier_edges,candidate_incoming)]):,.1f} ({pct(mean([a+b for a,b in zip(frontier_edges,candidate_incoming)]), e):.4f}% of E)",
        "",
        "## Current total edge-work proxy",
        "",
        f"- previous dense propagation + dense-P plasticity: E+P = {old_work:,}",
        f"- current frontier propagation + dense-P plasticity mean: {mean(current_work):,.1f}",
        f"- current/previous mean edge-work ratio: {mean(current_work) / old_work:.6f}",
        f"- implied reduction: {old_work / mean(current_work):.3f}x",
        "",
        "Plasticity still scans all P every neural step, so this ratio deliberately includes that remaining dense cost. It is the basis for deciding whether the next optimization should be a live plasticity frontier.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"frontier_activity_profile={report}")
    print("production_checkpoint_modified=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
