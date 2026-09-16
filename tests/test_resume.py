from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from virtual_fly.training.resume import (
    PENDING_CURRICULUM_STATE,
    PENDING_POPULATION_STATE,
    finalize_staged_state_files,
    recover_checkpoint_directory,
    recover_staged_state_files,
    reconcile_experiment_logs,
    stage_experiment_state_files,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CheckpointDirectoryRecoveryTests(unittest.TestCase):
    def test_promotes_completed_temp_when_active_checkpoint_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temp_checkpoint = root / ".checkpoint.tmp"
            temp_checkpoint.mkdir()
            (temp_checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
            (temp_checkpoint / "weights.f32le").write_bytes(b"weights")

            recovery = recover_checkpoint_directory(root)

            self.assertEqual(recovery.action, "promoted_completed_checkpoint_temp")
            self.assertTrue(recovery.checkpoint_exists)
            self.assertTrue((root / "checkpoint" / "weights.f32le").is_file())
            self.assertFalse(temp_checkpoint.exists())

    def test_active_checkpoint_wins_over_stale_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
            temp_checkpoint = root / ".checkpoint.tmp"
            temp_checkpoint.mkdir()
            (temp_checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")

            recovery = recover_checkpoint_directory(root)

            self.assertEqual(recovery.action, "discarded_stale_checkpoint_temp")
            self.assertTrue((checkpoint / "manifest.json").is_file())
            self.assertFalse(temp_checkpoint.exists())

    def test_incomplete_temp_without_active_checkpoint_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            temp_checkpoint = root / ".checkpoint.tmp"
            temp_checkpoint.mkdir()
            (temp_checkpoint / "weights.f32le").write_bytes(b"partial")

            recovery = recover_checkpoint_directory(root)

            self.assertEqual(recovery.action, "discarded_incomplete_checkpoint_temp")
            self.assertFalse(recovery.checkpoint_exists)
            self.assertFalse(temp_checkpoint.exists())


class StagedStateRecoveryTests(unittest.TestCase):
    def _write_checkpoint_manifest(self, root: Path, version: int | None) -> None:
        checkpoint = root / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"schema_version": 3, "step": 42}
        if version is not None:
            payload["global_weight_version"] = version
        (checkpoint / "manifest.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )

    def _write_active_state(self, root: Path, version: int) -> None:
        (root / "population-state.json").write_text(
            json.dumps({"global_weight_version": version}) + "\n",
            encoding="utf-8",
        )
        (root / "curriculum-state.json").write_text(
            json.dumps({"marker": f"active-{version}"}) + "\n",
            encoding="utf-8",
        )

    def test_checkpoint_ahead_recovers_matching_pending_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_active_state(root, 10)
            stage_experiment_state_files(
                root,
                curriculum_state={"marker": "pending-12"},
                population_state={"global_weight_version": 12},
            )
            self._write_checkpoint_manifest(root, 12)

            recovery = recover_staged_state_files(root)

            self.assertEqual(recovery.action, "recovered_checkpoint_ahead_of_active_state")
            population = json.loads((root / "population-state.json").read_text(encoding="utf-8"))
            curriculum = json.loads((root / "curriculum-state.json").read_text(encoding="utf-8"))
            self.assertEqual(population["global_weight_version"], 12)
            self.assertEqual(curriculum["marker"], "pending-12")
            self.assertFalse((root / PENDING_POPULATION_STATE).exists())
            self.assertFalse((root / PENDING_CURRICULUM_STATE).exists())

    def test_pending_state_is_discarded_when_checkpoint_never_advanced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_active_state(root, 10)
            self._write_checkpoint_manifest(root, 10)
            stage_experiment_state_files(
                root,
                curriculum_state={"marker": "pending-12"},
                population_state={"global_weight_version": 12},
            )

            recovery = recover_staged_state_files(root)

            self.assertEqual(recovery.action, "discarded_stale_pending")
            population = json.loads((root / "population-state.json").read_text(encoding="utf-8"))
            self.assertEqual(population["global_weight_version"], 10)
            self.assertFalse((root / PENDING_POPULATION_STATE).exists())
            self.assertFalse((root / PENDING_CURRICULUM_STATE).exists())

    def test_population_state_is_commit_marker_when_finalization_is_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_active_state(root, 10)
            self._write_checkpoint_manifest(root, 12)
            stage_experiment_state_files(
                root,
                curriculum_state={"marker": "pending-12"},
                population_state={"global_weight_version": 12},
            )
            # Simulate interruption after curriculum rename but before population
            # state becomes the commit marker.
            (root / PENDING_CURRICULUM_STATE).replace(root / "curriculum-state.json")

            recovery = recover_staged_state_files(root)
            self.assertEqual(
                recovery.action,
                "recovered_population_marker_after_curriculum_finalize",
            )
            population = json.loads((root / "population-state.json").read_text(encoding="utf-8"))
            curriculum = json.loads((root / "curriculum-state.json").read_text(encoding="utf-8"))
            self.assertEqual(population["global_weight_version"], 12)
            self.assertEqual(curriculum["marker"], "pending-12")

    def test_finalize_staged_state_writes_population_marker_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_experiment_state_files(
                root,
                curriculum_state={"marker": "new"},
                population_state={"global_weight_version": 4},
            )
            self.assertEqual(finalize_staged_state_files(root), 4)
            self.assertEqual(
                json.loads((root / "population-state.json").read_text(encoding="utf-8"))[
                    "global_weight_version"
                ],
                4,
            )

    def test_legacy_checkpoint_discards_new_pending_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_active_state(root, 10)
            self._write_checkpoint_manifest(root, None)
            stage_experiment_state_files(
                root,
                curriculum_state={"marker": "pending-12"},
                population_state={"global_weight_version": 12},
            )
            recovery = recover_staged_state_files(root)
            self.assertEqual(recovery.action, "discarded_pending_for_legacy_checkpoint")
            self.assertFalse((root / PENDING_POPULATION_STATE).exists())


class ResumeReconciliationTests(unittest.TestCase):
    def test_rolls_commit_and_trajectory_back_to_persisted_weight_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "commit-log.jsonl",
                [
                    {"commit_weight_version": 1, "episode": 10},
                    {"commit_weight_version": 2, "episode": 12},
                    {"commit_weight_version": 3, "episode": 11},
                ],
            )
            write_jsonl(
                root / "trajectory.jsonl",
                [
                    {"episode": 0, "control_step": 0},
                    {"episode": 10, "control_step": 0},
                    {"episode": 11, "control_step": 0},
                    {"episode": 12, "control_step": 0},
                    {"episode": 13, "control_step": 0},
                ],
            )

            result = reconcile_experiment_logs(root, stable_global_weight_version=2)

            self.assertTrue(result.changed)
            self.assertEqual(result.retained_commits, 2)
            self.assertEqual(result.removed_commits, 1)
            self.assertEqual(result.removed_trajectory_lines, 2)
            self.assertEqual(result.removed_episode_ids, (11, 13))
            self.assertEqual(result.next_episode, 13)
            self.assertEqual(
                [int(row["commit_weight_version"]) for row in load_jsonl(root / "commit-log.jsonl")],
                [1, 2],
            )
            self.assertEqual(
                [int(row["episode"]) for row in load_jsonl(root / "trajectory.jsonl")],
                [0, 10, 12],
            )

    def test_removes_partial_trajectory_even_when_no_orphan_commit_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "commit-log.jsonl", [{"commit_weight_version": 1, "episode": 10}])
            write_jsonl(
                root / "trajectory.jsonl",
                [
                    {"episode": 0, "control_step": 0},
                    {"episode": 10, "control_step": 0},
                    {"episode": 11, "control_step": 0},
                ],
            )

            result = reconcile_experiment_logs(root, stable_global_weight_version=1)

            self.assertTrue(result.changed)
            self.assertEqual(result.removed_commits, 0)
            self.assertEqual(result.removed_trajectory_lines, 1)
            self.assertEqual(result.removed_episode_ids, (11,))
            self.assertEqual(result.next_episode, 11)

    def test_dry_run_reports_repair_without_rewriting_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commit_rows = [
                {"commit_weight_version": 1, "episode": 10},
                {"commit_weight_version": 2, "episode": 11},
            ]
            trajectory_rows = [
                {"episode": 10, "control_step": 0},
                {"episode": 11, "control_step": 0},
            ]
            write_jsonl(root / "commit-log.jsonl", commit_rows)
            write_jsonl(root / "trajectory.jsonl", trajectory_rows)

            result = reconcile_experiment_logs(
                root,
                stable_global_weight_version=1,
                apply=False,
            )

            self.assertTrue(result.changed)
            self.assertEqual(result.removed_commits, 1)
            self.assertEqual(result.removed_trajectory_lines, 1)
            self.assertEqual(load_jsonl(root / "commit-log.jsonl"), commit_rows)
            self.assertEqual(load_jsonl(root / "trajectory.jsonl"), trajectory_rows)

    def test_requires_commit_log_to_reach_persisted_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "commit-log.jsonl", [{"commit_weight_version": 1, "episode": 10}])
            write_jsonl(root / "trajectory.jsonl", [{"episode": 10, "control_step": 0}])
            with self.assertRaisesRegex(RuntimeError, "persisted global weight version"):
                reconcile_experiment_logs(root, stable_global_weight_version=2)

    def test_pre_population_trajectory_numbering_is_preserved_without_commit_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "trajectory.jsonl",
                [
                    {"episode": 3, "control_step": 0},
                    {"episode": 7, "control_step": 0},
                ],
            )
            result = reconcile_experiment_logs(root, stable_global_weight_version=0)
            self.assertFalse(result.changed)
            self.assertEqual(result.next_episode, 8)


if __name__ == "__main__":
    unittest.main()
