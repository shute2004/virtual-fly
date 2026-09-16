from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from virtual_fly.training.checkpointing import (
    COMMIT_SEMANTICS,
    persist_shared_checkpoint,
    population_state_snapshot,
    prepare_population_resume,
    recover_population_storage,
)


class PopulationCheckpointingTests(unittest.TestCase):
    def test_recover_population_storage_reports_active_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "manifest.json").write_text(
                json.dumps({"global_weight_version": 0}) + "\n",
                encoding="utf-8",
            )
            (root / "population-state.json").write_text(
                json.dumps({"global_weight_version": 0, "launch_mode": "async"}) + "\n",
                encoding="utf-8",
            )
            recovery = recover_population_storage(root)
            self.assertTrue(recovery.checkpoint_exists)
            self.assertEqual(recovery.checkpoint_directory.action, "active_checkpoint")
            self.assertIsNotNone(recovery.staged_state)

    def test_prepare_population_resume_reconciles_to_persisted_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "population-state.json").write_text(
                json.dumps({"global_weight_version": 1, "launch_mode": "async"}) + "\n",
                encoding="utf-8",
            )
            (root / "commit-log.jsonl").write_text(
                json.dumps({"commit_weight_version": 1, "episode": 10}) + "\n",
                encoding="utf-8",
            )
            (root / "trajectory.jsonl").write_text(
                json.dumps({"episode": 10, "control_step": 0}) + "\n",
                encoding="utf-8",
            )
            plan = prepare_population_resume(
                root,
                requested_launch_mode="async",
                checkpoint_exists=True,
            )
            self.assertEqual(plan.initial_global_weight_version, 1)
            self.assertEqual(plan.start_episode, 11)
            self.assertEqual(plan.trajectory_mode, "a")
            self.assertEqual(plan.commit_mode, "a")
            self.assertIsNotNone(plan.reconciliation)

    def test_prepare_population_resume_rejects_mode_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "population-state.json").write_text(
                json.dumps({"global_weight_version": 0, "launch_mode": "async"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot change launch mode"):
                prepare_population_resume(
                    root,
                    requested_launch_mode="wave",
                    checkpoint_exists=True,
                )

    def test_population_state_snapshot_carries_shared_weight_contract(self) -> None:
        payload = population_state_snapshot(
            population=4,
            global_weight_version=184,
            curriculum_mode="boundary-band",
            launch_mode="wave",
            body_runtime="process-isolated",
        )
        self.assertEqual(payload["population"], 4)
        self.assertEqual(payload["global_weight_version"], 184)
        self.assertEqual(payload["commit_semantics"], COMMIT_SEMANTICS)
        self.assertFalse(payload["weight_averaging"])
        self.assertEqual(payload["launch_mode"], "wave")
        self.assertEqual(payload["body_runtime"], "process-isolated")

    def test_persist_shared_checkpoint_commits_matching_small_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            population_state = population_state_snapshot(
                population=4,
                global_weight_version=12,
                curriculum_mode="boundary-band",
                launch_mode="async",
            )

            def save(path: Path) -> dict[str, object]:
                path.mkdir(parents=True, exist_ok=True)
                (path / "manifest.json").write_text(
                    json.dumps({"global_weight_version": 12}) + "\n",
                    encoding="utf-8",
                )
                return {"event": "checkpoint_saved", "global_weight_version": 12}

            saved = persist_shared_checkpoint(
                output_dir=root,
                checkpoint_dir=checkpoint,
                curriculum_state={"curriculum_episodes": 24},
                population_state=population_state,
                save_checkpoint=save,
            )

            self.assertEqual(saved["global_weight_version"], 12)
            active_population = json.loads(
                (root / "population-state.json").read_text(encoding="utf-8")
            )
            active_curriculum = json.loads(
                (root / "curriculum-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(active_population["global_weight_version"], 12)
            self.assertEqual(active_curriculum["curriculum_episodes"], 24)

    def test_persist_rejects_checkpoint_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            population_state = population_state_snapshot(
                population=4,
                global_weight_version=12,
                curriculum_mode="boundary-band",
                launch_mode="async",
            )

            with self.assertRaisesRegex(RuntimeError, "different global weight version"):
                persist_shared_checkpoint(
                    output_dir=root,
                    checkpoint_dir=root / "checkpoint",
                    curriculum_state={"curriculum_episodes": 24},
                    population_state=population_state,
                    save_checkpoint=lambda _path: {"global_weight_version": 13},
                )

    def test_snapshot_rejects_invalid_sizes(self) -> None:
        with self.assertRaises(ValueError):
            population_state_snapshot(
                population=0,
                global_weight_version=0,
                curriculum_mode="boundary-band",
                launch_mode="async",
            )
        with self.assertRaises(ValueError):
            population_state_snapshot(
                population=1,
                global_weight_version=-1,
                curriculum_mode="boundary-band",
                launch_mode="async",
            )


if __name__ == "__main__":
    unittest.main()
