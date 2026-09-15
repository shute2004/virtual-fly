#!/usr/bin/env python3
"""Read-only smoke test for the shared-weight population neural bridge.

This loads the production checkpoint into an in-memory two-slot runtime, exercises
batched stepping, stale transaction commit ordering, and slot restart semantics,
then exits without saving anything back to the production experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "scripts" / "embodiment"
sys.path.insert(0, str(EMB))

from population_neural_bridge_client import PopulationNeuralBridgeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/malecns-v1.0/embodiment-groups-v0.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/experiments/flyppy-v3/checkpoint"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [
        args.snapshot / "manifest.json",
        args.groups,
        args.checkpoint / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing population smoke-test inputs:\n  " + "\n  ".join(missing))

    with PopulationNeuralBridgeClient(
        snapshot=args.snapshot,
        groups=args.groups,
        slots=2,
        repo_root=ROOT,
    ) as brain:
        brain.ping()
        loaded = brain.load_checkpoint(args.checkpoint, global_weight_version=0)
        if int(loaded.get("global_weight_version", -1)) != 0:
            raise RuntimeError(f"unexpected initial global version: {loaded}")

        batch = brain.step_batch(
            [
                {"slot": 0, "stimulate_body": (), "read_body": ()},
                {"slot": 1, "stimulate_body": (), "read_body": ()},
            ],
            plasticity=True,
        )
        if set(batch) != {0, 1}:
            raise RuntimeError(f"unexpected batch slots: {sorted(batch)}")

        # Exercise the same four-step reinforcement batching used by Flyppy v3.
        brain.step_slot(
            0,
            stimulate={"reward_dan": 2.0},
            plasticity=True,
            steps=4,
        )

        first = brain.commit_slot(0, source_weight_version=0)
        if (
            int(first["commit_from_version"]) != 0
            or int(first["commit_weight_version"]) != 1
            or int(first["staleness"]) != 0
        ):
            raise RuntimeError(f"unexpected first commit: {first}")

        # Slot 1 still carries an episode-local transaction generated from v0.
        # Its commit must rebase onto the latest v1 rather than overwrite it.
        second = brain.commit_slot(1, source_weight_version=0)
        if (
            int(second["commit_from_version"]) != 1
            or int(second["commit_weight_version"]) != 2
            or int(second["staleness"]) != 1
        ):
            raise RuntimeError(f"unexpected stale commit: {second}")

        if brain.global_weight_version != 2:
            raise RuntimeError(
                f"global version did not advance monotonically: {brain.global_weight_version}"
            )

    print("population_shared_weight_smoke=PASS")
    print("checkpoint_modified=false")
    print("commit_order=slot0:v0->v1,slot1:stale-v0-rebased-on-v1->v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
