from __future__ import annotations

from virtual_fly.semantics import (
    POPULATION_CHECKPOINT_SEMANTICS,
    POPULATION_NEURAL_STEP_SEMANTICS,
)

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from .population_schedule import validate_resume_launch_mode
from .resume import (
    CheckpointDirectoryRecovery,
    ResumeReconciliation,
    StagedStateRecovery,
    finalize_staged_state_files,
    recover_checkpoint_directory,
    recover_staged_state_files,
    reconcile_experiment_logs,
    stage_experiment_state_files,
)


COMMIT_SEMANTICS = (
    "episode-local additive+clamp transaction rebased onto latest global weight"
)


@dataclass(frozen=True)
class PopulationStorageRecovery:
    checkpoint_directory: CheckpointDirectoryRecovery
    staged_state: StagedStateRecovery | None

    @property
    def checkpoint_exists(self) -> bool:
        return self.checkpoint_directory.checkpoint_exists


@dataclass(frozen=True)
class PopulationResumePlan:
    population_state: dict[str, Any]
    initial_global_weight_version: int
    start_episode: int
    trajectory_mode: str
    commit_mode: str
    reconciliation: ResumeReconciliation | None


def recover_population_storage(output_dir: Path) -> PopulationStorageRecovery:
    checkpoint_directory = recover_checkpoint_directory(output_dir)
    staged_state = (
        recover_staged_state_files(output_dir)
        if checkpoint_directory.checkpoint_exists
        else None
    )
    return PopulationStorageRecovery(
        checkpoint_directory=checkpoint_directory,
        staged_state=staged_state,
    )


def prepare_population_resume(
    output_dir: Path,
    *,
    requested_launch_mode: str,
    checkpoint_exists: bool,
) -> PopulationResumePlan:
    output_dir = Path(output_dir)
    population_state_path = output_dir / "population-state.json"
    commit_log_path = output_dir / "commit-log.jsonl"

    population_state: dict[str, Any] = {}
    allow_legacy = os.environ.get("VF_ALLOW_LEGACY_POPULATION_CHECKPOINT") == "1"
    if checkpoint_exists and population_state_path.exists():
        payload = json.loads(population_state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("population-state.json must contain a JSON object")
        population_state = payload
        validate_resume_launch_mode(population_state, requested_launch_mode)
        if not allow_legacy:
            if population_state.get("checkpoint_semantics") != POPULATION_CHECKPOINT_SEMANTICS:
                raise RuntimeError(
                    "population checkpoint predates explicit weights-only semantics; refuse silent resume. "
                    "Use VF_ALLOW_LEGACY_POPULATION_CHECKPOINT=1 only for intentional historical continuation."
                )
            if population_state.get("neural_step_semantics") != POPULATION_NEURAL_STEP_SEMANTICS:
                raise RuntimeError("population-state neural step semantics are missing or incompatible")
            checkpoint_manifest_path = output_dir / "checkpoint" / "manifest.json"
            if not checkpoint_manifest_path.exists():
                raise RuntimeError("population checkpoint manifest is missing")
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
            if checkpoint_manifest.get("checkpoint_semantics") != POPULATION_CHECKPOINT_SEMANTICS:
                raise RuntimeError("checkpoint manifest is not an explicit population weights-only checkpoint")
            if checkpoint_manifest.get("step_semantics") != POPULATION_NEURAL_STEP_SEMANTICS:
                raise RuntimeError("checkpoint manifest neural step semantics are missing or incompatible")
    elif checkpoint_exists and commit_log_path.exists():
        raise RuntimeError(
            "population checkpoint has a commit log but no population-state.json; "
            "cannot determine the persisted global weight version safely"
        )

    try:
        initial_global_weight_version = int(
            population_state.get("global_weight_version", 0)
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("population global_weight_version must be an integer") from error
    if initial_global_weight_version < 0:
        raise RuntimeError("population global_weight_version must be >= 0")

    reconciliation: ResumeReconciliation | None = None
    if checkpoint_exists:
        reconciliation = reconcile_experiment_logs(
            output_dir,
            stable_global_weight_version=initial_global_weight_version,
        )
        start_episode = reconciliation.next_episode
    else:
        start_episode = 0

    return PopulationResumePlan(
        population_state=population_state,
        initial_global_weight_version=initial_global_weight_version,
        start_episode=start_episode,
        trajectory_mode="a" if checkpoint_exists else "w",
        commit_mode="a" if checkpoint_exists and commit_log_path.exists() else "w",
        reconciliation=reconciliation,
    )


def population_state_snapshot(
    *,
    population: int,
    global_weight_version: int,
    curriculum_mode: str,
    launch_mode: str,
    body_runtime: str | None = None,
) -> dict[str, object]:
    if population < 1:
        raise ValueError("population must be >= 1")
    if global_weight_version < 0:
        raise ValueError("global_weight_version must be >= 0")
    payload: dict[str, object] = {
        "schema_version": 2,
        "population": int(population),
        "global_weight_version": int(global_weight_version),
        "commit_semantics": COMMIT_SEMANTICS,
        "weight_averaging": False,
        "checkpoint_semantics": POPULATION_CHECKPOINT_SEMANTICS,
        "neural_step_semantics": POPULATION_NEURAL_STEP_SEMANTICS,
        "curriculum_mode": str(curriculum_mode),
        "launch_mode": str(launch_mode),
    }
    if body_runtime is not None:
        payload["body_runtime"] = str(body_runtime)
    return payload


def persist_shared_checkpoint(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    curriculum_state: Mapping[str, Any],
    population_state: Mapping[str, Any],
    save_checkpoint: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist CNS weights and small experiment state as one recoverable transaction."""

    staged_version = stage_experiment_state_files(
        output_dir,
        curriculum_state=curriculum_state,
        population_state=population_state,
    )
    saved = dict(save_checkpoint(checkpoint_dir))
    try:
        saved_version = int(saved.get("global_weight_version", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError("checkpoint save returned an invalid global weight version") from error
    if saved_version != staged_version:
        raise RuntimeError(
            "checkpoint save returned a different global weight version: "
            f"staged={staged_version} saved={saved_version}"
        )
    finalized_version = finalize_staged_state_files(output_dir)
    if finalized_version != staged_version:
        raise RuntimeError(
            "finalized population state version differs from checkpoint: "
            f"staged={staged_version} finalized={finalized_version}"
        )
    return saved
