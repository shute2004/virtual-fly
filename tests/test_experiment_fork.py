from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from virtual_fly.training.experiment_fork import fork_flyppy_experiment
from virtual_fly.training.population_schedule import validate_resume_launch_mode


class FlyppyExperimentForkTests(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        source = root / "source"
        checkpoint = source / "checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text(
            json.dumps({"schema_version": 3, "step": 15781}) + "\n",
            encoding="utf-8",
        )
        (checkpoint / "weights.f32le").write_bytes(b"weights")
        (source / "curriculum-state.json").write_text(
            json.dumps({"curriculum_mode": "boundary-band"}) + "\n",
            encoding="utf-8",
        )
        (source / "population-state.json").write_text(
            json.dumps({"global_weight_version": 184}) + "\n",
            encoding="utf-8",
        )
        (source / "trajectory.jsonl").write_text(
            '\n'.join(
                [
                    json.dumps({"episode": 254}),
                    json.dumps({"episode": 255}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (source / "commit-log.jsonl").write_text(
            '\n'.join(
                [
                    json.dumps({"episode": 254, "commit_seq": 183, "commit_weight_version": 183}),
                    json.dumps({"episode": 255, "commit_seq": 184, "commit_weight_version": 184}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (source / "summary.json").write_text("{}\n", encoding="utf-8")
        live = source / "live"
        live.mkdir()
        (live / "body.json").write_text("{}\n", encoding="utf-8")
        return source

    def test_fork_preserves_resume_state_without_copying_runtime_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            destination = root / "fork"

            metadata = fork_flyppy_experiment(source, destination, launch_mode="wave")

            self.assertEqual(metadata["checkpoint_neural_step"], 15781)
            self.assertEqual(metadata["global_weight_version"], 184)
            self.assertEqual(metadata["next_episode"], 256)
            self.assertEqual(metadata["source_launch_mode"], "async")
            self.assertEqual(metadata["fork_launch_mode"], "wave")
            self.assertTrue((destination / "checkpoint" / "weights.f32le").is_file())
            for name in (
                "curriculum-state.json",
                "population-state.json",
                "trajectory.jsonl",
                "commit-log.jsonl",
                "fork-metadata.json",
            ):
                self.assertTrue((destination / name).is_file(), name)
            self.assertFalse((destination / "summary.json").exists())
            self.assertFalse((destination / "live").exists())
            fork_population = json.loads(
                (destination / "population-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(fork_population["launch_mode"], "wave")

            source_weights = source / "checkpoint" / "weights.f32le"
            destination_weights = destination / "checkpoint" / "weights.f32le"
            if metadata["checkpoint_transfer"]["hardlinked_files"]:
                self.assertEqual(source_weights.stat().st_ino, destination_weights.stat().st_ino)

    def test_fork_discards_uncheckpointed_tail_only_in_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            with (source / "commit-log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"episode": 257, "commit_seq": 185, "commit_weight_version": 185}
                    )
                    + "\n"
                )
            with (source / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode": 256}) + "\n")
                handle.write(json.dumps({"episode": 257}) + "\n")

            destination = root / "fork"
            metadata = fork_flyppy_experiment(source, destination, launch_mode="wave")

            self.assertEqual(metadata["next_episode"], 256)
            self.assertEqual(metadata["resume_reconciliation"]["removed_commits"], 1)
            self.assertEqual(metadata["resume_reconciliation"]["removed_trajectory_lines"], 2)
            source_commits = [
                json.loads(line)
                for line in (source / "commit-log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            fork_commits = [
                json.loads(line)
                for line in (destination / "commit-log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(source_commits[-1]["commit_weight_version"], 185)
            self.assertEqual(fork_commits[-1]["commit_weight_version"], 184)
            fork_episodes = [
                int(json.loads(line)["episode"])
                for line in (destination / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(fork_episodes, [254, 255])

    def test_fork_recovers_interrupted_versioned_checkpoint_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            checkpoint_manifest = json.loads(
                (source / "checkpoint" / "manifest.json").read_text(encoding="utf-8")
            )
            checkpoint_manifest["global_weight_version"] = 186
            (source / "checkpoint" / "manifest.json").write_text(
                json.dumps(checkpoint_manifest) + "\n",
                encoding="utf-8",
            )
            (source / ".curriculum-state.pending.json").write_text(
                json.dumps({"curriculum_mode": "boundary-band", "marker": "v186"}) + "\n",
                encoding="utf-8",
            )
            (source / ".population-state.pending.json").write_text(
                json.dumps({"global_weight_version": 186, "launch_mode": "async"}) + "\n",
                encoding="utf-8",
            )
            (source / "commit-log.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"episode": 255, "commit_weight_version": 184},
                        {"episode": 256, "commit_weight_version": 185},
                        {"episode": 257, "commit_weight_version": 186},
                    )
                ),
                encoding="utf-8",
            )
            with (source / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode": 256}) + "\n")
                handle.write(json.dumps({"episode": 257}) + "\n")

            destination = root / "fork"
            metadata = fork_flyppy_experiment(source, destination, launch_mode="wave")

            self.assertEqual(metadata["global_weight_version"], 186)
            self.assertEqual(
                metadata["staged_state_recovery"]["action"],
                "recovered_checkpoint_ahead_of_active_state",
            )
            fork_population = json.loads(
                (destination / "population-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(fork_population["global_weight_version"], 186)
            self.assertEqual(fork_population["launch_mode"], "wave")
            source_population = json.loads(
                (source / "population-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(source_population["global_weight_version"], 184)
            self.assertTrue((source / ".population-state.pending.json").exists())

    def test_fork_uses_completed_temp_checkpoint_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            active = source / "checkpoint"
            completed_temp = source / ".checkpoint.tmp"
            active.replace(completed_temp)

            destination = root / "fork"
            metadata = fork_flyppy_experiment(source, destination)

            self.assertEqual(metadata["checkpoint_neural_step"], 15781)
            self.assertTrue((destination / "checkpoint" / "manifest.json").is_file())
            self.assertTrue((source / ".checkpoint.tmp" / "manifest.json").is_file())
            self.assertFalse((source / "checkpoint").exists())

    def test_resume_launch_mode_treats_legacy_state_as_async(self) -> None:
        self.assertEqual(validate_resume_launch_mode({}, "async"), "async")
        with self.assertRaisesRegex(ValueError, "fork the experiment first"):
            validate_resume_launch_mode({}, "wave")
        self.assertEqual(validate_resume_launch_mode({"launch_mode": "wave"}, "wave"), "wave")

    def test_fork_rejects_unknown_launch_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            with self.assertRaises(ValueError):
                fork_flyppy_experiment(source, root / "fork", launch_mode="lockstep")

    def test_fork_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            destination = root / "fork"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                fork_flyppy_experiment(source, destination)

    def test_fork_refuses_incomplete_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root)
            (source / "commit-log.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                fork_flyppy_experiment(source, root / "fork")


if __name__ == "__main__":
    unittest.main()
