from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


PENDING_CURRICULUM_STATE = ".curriculum-state.pending.json"
PENDING_POPULATION_STATE = ".population-state.pending.json"


@dataclass(frozen=True)
class CheckpointDirectoryRecovery:
    action: str
    checkpoint_exists: bool


@dataclass(frozen=True)
class StagedStateRecovery:
    checkpoint_global_weight_version: int | None
    active_global_weight_version: int | None
    pending_global_weight_version: int | None
    action: str


@dataclass(frozen=True)
class ResumeReconciliation:
    stable_global_weight_version: int
    retained_commits: int
    removed_commits: int
    removed_trajectory_lines: int
    removed_episode_ids: tuple[int, ...]
    next_episode: int

    @property
    def changed(self) -> bool:
        return self.removed_commits > 0 or self.removed_trajectory_lines > 0


def recover_checkpoint_directory(output_dir: Path) -> CheckpointDirectoryRecovery:
    """Resolve interruption around the final temp-directory checkpoint rename.

    Rust writes the complete temp checkpoint, including its manifest, before it
    removes the previous active checkpoint.  Therefore an absent active
    directory plus a temp directory with a manifest is a completed checkpoint
    that can safely be promoted.  If the active checkpoint still exists, any
    temp directory belongs to a save that never reached replacement and is
    discarded.
    """

    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoint"
    temp_checkpoint = output_dir / ".checkpoint.tmp"
    active_manifest = checkpoint / "manifest.json"
    temp_manifest = temp_checkpoint / "manifest.json"

    if active_manifest.is_file():
        if temp_checkpoint.exists():
            import shutil

            shutil.rmtree(temp_checkpoint)
            return CheckpointDirectoryRecovery("discarded_stale_checkpoint_temp", True)
        return CheckpointDirectoryRecovery("active_checkpoint", True)

    if temp_manifest.is_file():
        if checkpoint.exists():
            import shutil

            shutil.rmtree(checkpoint)
        temp_checkpoint.replace(checkpoint)
        return CheckpointDirectoryRecovery("promoted_completed_checkpoint_temp", True)

    if temp_checkpoint.exists():
        import shutil

        shutil.rmtree(temp_checkpoint)
        return CheckpointDirectoryRecovery("discarded_incomplete_checkpoint_temp", False)

    return CheckpointDirectoryRecovery("no_checkpoint", False)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    temp = path.with_name(f".{path.name}.resume.tmp")
    temp.write_text("".join(lines), encoding="utf-8")
    temp.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def stage_experiment_state_files(
    output_dir: Path,
    *,
    curriculum_state: Mapping[str, Any],
    population_state: Mapping[str, Any],
) -> int:
    """Write the small state files that must commit with the next CNS checkpoint.

    The pending files are written *before* the neural checkpoint.  If the process
    dies after the checkpoint replacement but before the active state files are
    renamed, startup can finish this transaction by matching the checkpoint's
    embedded global weight version to the pending population state.
    """

    output_dir = Path(output_dir)
    try:
        version = int(population_state["global_weight_version"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("population_state requires integer global_weight_version") from error
    if version < 0:
        raise ValueError("global_weight_version must be >= 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / PENDING_CURRICULUM_STATE, curriculum_state)
    _atomic_write_json(output_dir / PENDING_POPULATION_STATE, population_state)
    return version


def finalize_staged_state_files(output_dir: Path) -> int:
    """Commit pending curriculum first and population state last as the marker."""

    output_dir = Path(output_dir)
    pending_curriculum = output_dir / PENDING_CURRICULUM_STATE
    pending_population = output_dir / PENDING_POPULATION_STATE
    if not pending_curriculum.is_file() or not pending_population.is_file():
        raise RuntimeError("cannot finalize incomplete staged experiment state")
    population = _load_json_object(pending_population)
    try:
        version = int(population["global_weight_version"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("pending population state lacks global_weight_version") from error
    # Population state is renamed last.  Its version therefore acts as the
    # small-file commit marker if interruption occurs between the two replaces.
    pending_curriculum.replace(output_dir / "curriculum-state.json")
    pending_population.replace(output_dir / "population-state.json")
    return version


def _discard_pending_state_files(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    for name in (PENDING_CURRICULUM_STATE, PENDING_POPULATION_STATE):
        path = output_dir / name
        if path.exists():
            path.unlink()


def recover_staged_state_files(output_dir: Path) -> StagedStateRecovery:
    """Recover or discard an interrupted two-phase experiment-state save."""

    output_dir = Path(output_dir)
    checkpoint_manifest_path = output_dir / "checkpoint" / "manifest.json"
    active_population_path = output_dir / "population-state.json"
    pending_curriculum_path = output_dir / PENDING_CURRICULUM_STATE
    pending_population_path = output_dir / PENDING_POPULATION_STATE

    active_version: int | None = None
    if active_population_path.is_file():
        active = _load_json_object(active_population_path)
        try:
            active_version = int(active["global_weight_version"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("active population state lacks global_weight_version") from error

    pending_version: int | None = None
    if pending_population_path.is_file():
        pending = _load_json_object(pending_population_path)
        try:
            pending_version = int(pending["global_weight_version"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("pending population state lacks global_weight_version") from error

    if not checkpoint_manifest_path.is_file():
        if pending_curriculum_path.exists() or pending_population_path.exists():
            _discard_pending_state_files(output_dir)
            action = "discarded_pending_without_checkpoint"
        else:
            action = "none"
        return StagedStateRecovery(None, active_version, pending_version, action)

    manifest = _load_json_object(checkpoint_manifest_path)
    raw_checkpoint_version = manifest.get("global_weight_version")
    if raw_checkpoint_version is None:
        # Legacy checkpoints predate version-tagged atomic state saves.  New-code
        # pending files can only belong to a failed save attempt that did not
        # replace the old checkpoint, so keeping them would be unsafe.
        if pending_curriculum_path.exists() or pending_population_path.exists():
            _discard_pending_state_files(output_dir)
            action = "discarded_pending_for_legacy_checkpoint"
        else:
            action = "legacy_checkpoint"
        return StagedStateRecovery(None, active_version, pending_version, action)

    try:
        checkpoint_version = int(raw_checkpoint_version)
    except (TypeError, ValueError) as error:
        raise RuntimeError("checkpoint global_weight_version is invalid") from error

    pending_complete = pending_curriculum_path.is_file() and pending_population_path.is_file()
    pending_any = pending_curriculum_path.exists() or pending_population_path.exists()

    if active_version == checkpoint_version:
        if pending_any:
            # A pending newer save that never reached checkpoint replacement is
            # stale.  If it matches the active checkpoint, finalizing is also safe
            # and handles interruption after curriculum rename but before the
            # population-state commit marker.
            if pending_complete and pending_version == checkpoint_version:
                finalize_staged_state_files(output_dir)
                action = "finalized_matching_pending"
            else:
                _discard_pending_state_files(output_dir)
                action = "discarded_stale_pending"
        else:
            action = "consistent"
        return StagedStateRecovery(
            checkpoint_version,
            active_version,
            pending_version,
            action,
        )

    if pending_complete and pending_version == checkpoint_version:
        finalize_staged_state_files(output_dir)
        return StagedStateRecovery(
            checkpoint_version,
            active_version,
            pending_version,
            "recovered_checkpoint_ahead_of_active_state",
        )

    if (
        pending_population_path.is_file()
        and not pending_curriculum_path.exists()
        and pending_version == checkpoint_version
    ):
        # Finalization intentionally renames curriculum first and population last.
        # Therefore this shape means the process died in the one remaining gap:
        # new curriculum is already active and the matching population marker is
        # still pending.  Completing that final rename is sufficient.
        pending_population_path.replace(active_population_path)
        return StagedStateRecovery(
            checkpoint_version,
            active_version,
            pending_version,
            "recovered_population_marker_after_curriculum_finalize",
        )

    raise RuntimeError(
        "checkpoint/global population state mismatch without a recoverable staged state: "
        f"checkpoint={checkpoint_version} active={active_version} pending={pending_version}"
    )


def _json_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL in {path} at line {line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise RuntimeError(f"expected JSON object in {path} at line {line_number}")
            rows.append((raw if raw.endswith("\n") else raw + "\n", payload))
    return rows


def reconcile_experiment_logs(
    output_dir: Path,
    *,
    stable_global_weight_version: int,
    apply: bool = True,
) -> ResumeReconciliation:
    """Roll generated logs back to the last persisted shared-weight checkpoint.

    A Flyppy population checkpoint persists global neural weights and curriculum
    state, but active episode-local body/CNS state is intentionally not saved. If
    the process is interrupted between periodic checkpoints, trajectory and
    commit-log writes can therefore be ahead of the persisted global weights.

    The persisted population state's ``global_weight_version`` is the recovery
    watermark. Commits newer than that version are orphaned and are discarded.
    Trajectory rows for uncommitted async episodes are also discarded, while the
    historical pre-population trajectory prefix is retained.
    """

    if stable_global_weight_version < 0:
        raise ValueError("stable_global_weight_version must be >= 0")

    output_dir = Path(output_dir)
    commit_path = output_dir / "commit-log.jsonl"
    trajectory_path = output_dir / "trajectory.jsonl"
    if not commit_path.exists():
        # A checkpoint can predate async population logging. Preserve the older
        # serial trajectory numbering until the population commit log exists.
        trajectory_rows = _json_lines(trajectory_path) if trajectory_path.exists() else []
        next_episode = max(
            (int(row["episode"]) for _, row in trajectory_rows),
            default=-1,
        ) + 1
        return ResumeReconciliation(
            stable_global_weight_version=stable_global_weight_version,
            retained_commits=0,
            removed_commits=0,
            removed_trajectory_lines=0,
            removed_episode_ids=(),
            next_episode=next_episode,
        )

    commit_rows = _json_lines(commit_path)
    seen_versions: set[int] = set()
    retained_commit_rows: list[tuple[str, dict[str, Any]]] = []
    removed_commit_rows: list[tuple[str, dict[str, Any]]] = []
    previous_version = -1
    for raw, row in commit_rows:
        try:
            version = int(row["commit_weight_version"])
            episode = int(row["episode"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("commit log row lacks integer commit_weight_version/episode") from error
        if version in seen_versions or version <= previous_version:
            raise RuntimeError("commit log global weight versions are not strictly increasing")
        seen_versions.add(version)
        previous_version = version
        if version <= stable_global_weight_version:
            retained_commit_rows.append((raw, row))
        else:
            removed_commit_rows.append((raw, row))

    if stable_global_weight_version > 0:
        retained_versions = [int(row["commit_weight_version"]) for _, row in retained_commit_rows]
        if not retained_versions or retained_versions[-1] != stable_global_weight_version:
            raise RuntimeError(
                "persisted global weight version is not represented by the commit log: "
                f"stable={stable_global_weight_version} "
                f"last_retained={retained_versions[-1] if retained_versions else None}"
            )

    all_commit_episodes = [int(row["episode"]) for _, row in commit_rows]
    first_population_episode = min(all_commit_episodes) if all_commit_episodes else None
    retained_episode_ids = {int(row["episode"]) for _, row in retained_commit_rows}

    trajectory_rows = _json_lines(trajectory_path) if trajectory_path.exists() else []
    retained_trajectory: list[tuple[str, dict[str, Any]]] = []
    removed_trajectory: list[tuple[str, dict[str, Any]]] = []
    for raw, row in trajectory_rows:
        try:
            episode = int(row["episode"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("trajectory row lacks integer episode") from error
        historical_prefix = first_population_episode is None or episode < first_population_episode
        if historical_prefix or episode in retained_episode_ids:
            retained_trajectory.append((raw, row))
        else:
            removed_trajectory.append((raw, row))

    if apply and removed_commit_rows:
        _atomic_write_lines(commit_path, [raw for raw, _ in retained_commit_rows])
    if apply and removed_trajectory:
        _atomic_write_lines(trajectory_path, [raw for raw, _ in retained_trajectory])

    retained_episode_numbers = [int(row["episode"]) for _, row in retained_trajectory]
    retained_episode_numbers.extend(retained_episode_ids)
    next_episode = max(retained_episode_numbers, default=-1) + 1
    removed_episode_ids = tuple(
        sorted(
            {int(row["episode"]) for _, row in removed_commit_rows}
            | {int(row["episode"]) for _, row in removed_trajectory}
        )
    )
    return ResumeReconciliation(
        stable_global_weight_version=stable_global_weight_version,
        retained_commits=len(retained_commit_rows),
        removed_commits=len(removed_commit_rows),
        removed_trajectory_lines=len(removed_trajectory),
        removed_episode_ids=removed_episode_ids,
        next_episode=next_episode,
    )
