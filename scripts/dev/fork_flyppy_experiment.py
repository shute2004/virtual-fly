#!/usr/bin/env python3
"""Fork a resumable Flyppy experiment for controlled A/B training."""

from __future__ import annotations

import argparse
from pathlib import Path

from virtual_fly.training.experiment_fork import fork_flyppy_experiment


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--launch-mode", choices=("async", "wave"))
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    source = absolute(args.source)
    destination = absolute(args.destination)
    metadata = fork_flyppy_experiment(
        source,
        destination,
        launch_mode=args.launch_mode,
    )
    transfer = metadata["checkpoint_transfer"]
    print(f"source={source}")
    print(f"destination={destination}")
    print(f"checkpoint_neural_step={metadata['checkpoint_neural_step']}")
    print(f"global_weight_version={metadata['global_weight_version']}")
    print(f"next_episode={metadata['next_episode']}")
    print(f"source_launch_mode={metadata['source_launch_mode']}")
    print(f"fork_launch_mode={metadata['fork_launch_mode']}")
    print(f"hardlinked_checkpoint_files={len(transfer['hardlinked_files'])}")
    print(f"copied_checkpoint_files={len(transfer['copied_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
