#!/usr/bin/env python3
"""Measure real Flyppy propagation-frontier sparsity without touching production state.

A diagnostic-only bridge uses the exact production frontier runtime but exposes a
signed activity readback internally (both event codes 1 and 2 count as active).
The bridge computes frontier statistics from shared CPU-side topology only when
this profiler requests them. Normal production training does not use this path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_PRODUCTION = Path("artifacts/experiments/flyppy-v3")
DEFAULT_TEMP = Path("artifacts/profiles/flyppy-frontier-activity/run")
DEFAULT_REPORT = Path("reports/flyppy/frontier_activity_profile.md")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_CALIBRATION = Path("artifacts/embodiment/neural-runtime-calibration-v1.json")
P_EXPECTED = 10_871_322


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
    samples: list[dict[str, int | float]] = []
    normal_step_counter = 0

    class ProfilingClient(BaseClient):
        def __init__(self, *, snapshot: Path, groups: Path, slots: int = 1, repo_root: Path | None = None):
            if slots < 1:
                raise ValueError("slots must be >= 1")
            root = repo_root or ROOT
            command = [
                "cargo", "run", "-q", "-p", "vf-runner",
                "--bin", "population_frontier_profile_bridge", "--release", "--",
                "--snapshot", str(snapshot),
                "--groups", str(groups),
                "--slots", str(slots),
            ]
            self._proc = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            if self._proc.stdin is None or self._proc.stdout is None:
                raise bridge_module.PopulationNeuralBridgeError("failed to open frontier profile bridge pipes")
            self._stdin = self._proc.stdin
            self._stdout = self._proc.stdout
            self.last_step = 0
            self.ready = self._read_response()
            if self.ready.get("event") != "ready":
                self.close(force=True)
                raise bridge_module.PopulationNeuralBridgeError(
                    f"unexpected frontier profile bridge startup response: {self.ready}"
                )
            self.global_weight_version = int(self.ready.get("global_weight_version", 0))
            self.slots = int(self.ready.get("slots", slots))

        def frontier_stats(self, slot: int) -> dict:
            response = self._request({"type": "frontier_stats", "slot": int(slot)})
            if response.get("event") != "frontier_stats":
                raise bridge_module.PopulationNeuralBridgeError(
                    f"unexpected frontier stats response: {response}"
                )
            return response

        def step_batch(self, slot_requests, *, plasticity: bool = True):
            nonlocal normal_step_counter
            result = super().step_batch(slot_requests, plasticity=plasticity)
            if normal_step_counter % args.sample_stride == 0:
                stats = self.frontier_stats(0)
                samples.append(
                    {
                        "sample": len(samples),
                        "neural_step": self.last_step,
                        "normal_step": normal_step_counter,
                        "active_pre": int(stats["active_pre"]),
                        "frontier_edges": int(stats["frontier_edges"]),
                        "candidate_posts": int(stats["candidate_posts"]),
                        "candidate_incoming_edges": int(stats["candidate_incoming_edges"]),
                        "candidate_plastic_edges": int(stats["candidate_plastic_edges"]),
                    }
                )
            normal_step_counter += 1
            return result

    trainer.PopulationNeuralBridgeClient = ProfilingClient

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(Path(trainer.__file__).resolve()),
            "--episodes", str(args.episodes),
            "--population", "1",
            "--snapshot", str(snapshot_dir),
            "--groups", str(snapshot_dir / "embodiment-groups-v0.json"),
            "--retinotopic-map", str(snapshot_dir / "retinotopic-vision-v1.json"),
            "--wing-motor-map", str(snapshot_dir / "wing-motor-neurons-v0.json"),
            "--body-motor-map", str(snapshot_dir / "body-motor-neurons-v0.json"),
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
        raise RuntimeError("production checkpoint changed during frontier profiling")
    if rc != 0:
        raise RuntimeError(f"population trainer returned non-zero status {rc}")
    if not samples:
        raise RuntimeError("frontier profiler produced no samples")

    p = int(P_EXPECTED)
    active_counts = [int(s["active_pre"]) for s in samples]
    frontier_edges = [int(s["frontier_edges"]) for s in samples]
    candidate_posts = [int(s["candidate_posts"]) for s in samples]
    candidate_incoming = [int(s["candidate_incoming_edges"]) for s in samples]
    candidate_plastic = [int(s["candidate_plastic_edges"]) for s in samples]
    propagation_touch = [a + b for a, b in zip(frontier_edges, candidate_incoming)]
    current_work = [touch + p for touch in propagation_touch]
    old_work = e + p

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
        f"- P: {p:,}",
        f"- episodes: {args.episodes}",
        f"- max control steps / episode: {args.max_control_steps}",
        f"- sample stride: {args.sample_stride}",
        f"- samples: {len(samples)}",
        "",
        "## Aggregate frontier density",
        "",
        f"- active pre mean: {mean(active_counts):,.1f} ({pct(mean(active_counts), n):.4f}% of N)",
        f"- active pre median: {median(active_counts):,.1f}",
        f"- active pre max: {max(active_counts):,} ({pct(max(active_counts), n):.4f}% of N)",
        f"- active outgoing F mean: {mean(frontier_edges):,.1f} ({pct(mean(frontier_edges), e):.4f}% of E)",
        f"- candidate posts C mean: {mean(candidate_posts):,.1f} ({pct(mean(candidate_posts), n):.4f}% of N)",
        f"- candidate incoming I(C) mean: {mean(candidate_incoming):,.1f} ({pct(mean(candidate_incoming), e):.4f}% of E)",
        f"- propagation edge-touch F+I(C) mean: {mean(propagation_touch):,.1f} ({pct(mean(propagation_touch), e):.4f}% of E)",
        f"- candidate-post plastic edges mean: {mean(candidate_plastic):,.1f} ({pct(mean(candidate_plastic), p):.4f}% of P)",
        "",
        "## Current total edge-work proxy",
        "",
        f"- previous dense propagation + dense-P plasticity: E+P = {old_work:,}",
        f"- current frontier propagation + dense-P plasticity mean: {mean(current_work):,.1f}",
        f"- current/previous mean edge-work ratio: {mean(current_work) / old_work:.6f}",
        f"- implied reduction: {old_work / mean(current_work):.3f}x",
        "",
        "The candidate-post plastic-edge count is not itself a valid plasticity live set: eligibility and dopamine modulation can remain causally live after the immediate propagation frontier moves on. It is reported only as a structural bound/signal for the next live-plasticity analysis.",
        "",
        "## Samples",
        "",
        "| sample | neural step | normal step | active pre | F outgoing | C posts | I(C) incoming | plastic under C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in samples:
        lines.append(
            "| {sample} | {neural_step} | {normal_step} | {active_pre:,} | {frontier_edges:,} | {candidate_posts:,} | {candidate_incoming_edges:,} | {candidate_plastic_edges:,} |".format(**item)
        )
    lines.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"frontier_activity_profile={report}")
    print("production_checkpoint_modified=false")
    print(f"samples={len(samples)}")
    print(f"mean_propagation_edge_fraction={mean(propagation_touch) / e:.8f}")
    print(f"mean_total_edge_work_ratio={mean(current_work) / old_work:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
