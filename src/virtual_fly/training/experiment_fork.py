from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from .population_schedule import VALID_LAUNCH_MODES
from .resume import (
    PENDING_CURRICULUM_STATE,
    PENDING_POPULATION_STATE,
    recover_staged_state_files,
    reconcile_experiment_logs,
)

REQUIRED_STATE_FILES = (
    "curriculum-state.json",
    "population-state.json",
    "trajectory.jsonl",
    "commit-log.jsonl",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def fork_flyppy_experiment(
    source: Path,
    destination: Path,
    *,
    launch_mode: str | None = None,
) -> dict[str, Any]:
    """Fork one resumable Flyppy experiment without mutating the source.

    The neural checkpoint writer replaces the entire checkpoint directory on save,
    so hard-linking the immutable source checkpoint files is safe: a later save in
    either experiment unlinks that experiment's directory and writes fresh files.
    Cross-device filesystems transparently fall back to ordinary copies.
    """

    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination experiments must differ")
    if not source.is_dir():
        raise FileNotFoundError(f"source experiment does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"destination experiment already exists: {destination}")

    checkpoint = source / "checkpoint"
    checkpoint_manifest_path = checkpoint / "manifest.json"
    if not checkpoint_manifest_path.is_file():
        temp_checkpoint = source / ".checkpoint.tmp"
        temp_manifest = temp_checkpoint / "manifest.json"
        if temp_manifest.is_file():
            checkpoint = temp_checkpoint
            checkpoint_manifest_path = temp_manifest
        else:
            raise FileNotFoundError(
                "source checkpoint manifest is missing from both active and completed temp paths: "
                f"{source / 'checkpoint' / 'manifest.json'}"
            )
    for name in REQUIRED_STATE_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"source experiment state is missing: {path}")

    checkpoint_manifest = _load_json(checkpoint_manifest_path)
    population_state = _load_json(source / "population-state.json")
    source_launch_mode = str(population_state.get("launch_mode", "async"))
    if launch_mode is not None and launch_mode not in VALID_LAUNCH_MODES:
        raise ValueError("launch_mode must be async, wave, or None")
    fork_launch_mode = source_launch_mode if launch_mode is None else launch_mode
    destination.mkdir(parents=True)
    destination_checkpoint = destination / "checkpoint"
    destination_checkpoint.mkdir()

    transfer_modes: dict[str, str] = {}
    for item in sorted(checkpoint.iterdir()):
        if not item.is_file():
            raise RuntimeError(f"unexpected non-file in checkpoint: {item}")
        transfer_modes[item.name] = _link_or_copy(item, destination_checkpoint / item.name)

    copied_state_files = list(REQUIRED_STATE_FILES)
    for name in REQUIRED_STATE_FILES:
        shutil.copy2(source / name, destination / name)
    for name in (PENDING_CURRICULUM_STATE, PENDING_POPULATION_STATE):
        pending_source = source / name
        if pending_source.is_file():
            shutil.copy2(pending_source, destination / name)
            copied_state_files.append(name)

    staged_recovery = recover_staged_state_files(destination)
    fork_population_state = _load_json(destination / "population-state.json")
    stable_version = int(fork_population_state.get("global_weight_version", 0))
    reconciliation = reconcile_experiment_logs(
        destination,
        stable_global_weight_version=stable_version,
    )
    fork_population_state["launch_mode"] = fork_launch_mode
    (destination / "population-state.json").write_text(
        json.dumps(fork_population_state, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "schema_version": 1,
        "source_experiment": str(source),
        "checkpoint_neural_step": int(checkpoint_manifest.get("step", -1)),
        "global_weight_version": stable_version,
        "next_episode": reconciliation.next_episode,
        "source_launch_mode": source_launch_mode,
        "fork_launch_mode": fork_launch_mode,
        "checkpoint_transfer": {
            "hardlinked_files": sorted(
                name for name, mode in transfer_modes.items() if mode == "hardlink"
            ),
            "copied_files": sorted(
                name for name, mode in transfer_modes.items() if mode == "copy"
            ),
        },
        "copied_state_files": copied_state_files,
        "staged_state_recovery": {
            "action": staged_recovery.action,
            "checkpoint_global_weight_version": staged_recovery.checkpoint_global_weight_version,
            "active_global_weight_version": staged_recovery.active_global_weight_version,
            "pending_global_weight_version": staged_recovery.pending_global_weight_version,
        },
        "resume_reconciliation": {
            "removed_commits": reconciliation.removed_commits,
            "removed_trajectory_lines": reconciliation.removed_trajectory_lines,
            "removed_episode_ids": list(reconciliation.removed_episode_ids),
        },
        "note": (
            "Fork preserves checkpoint, curriculum, population state, trajectory, and commit log. "
            "summary.json and live telemetry are intentionally not copied."
        ),
    }
    (destination / "fork-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
