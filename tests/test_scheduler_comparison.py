from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(ANALYSIS_DIR))
MODULE_PATH = ANALYSIS_DIR / "compare_flyppy_scheduler_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_flyppy_scheduler_runs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
summarize_run = MODULE.summarize_run
validate_pair = MODULE.validate_pair
render_markdown = MODULE.render_markdown
portable = MODULE.portable


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_summary(mode: str, source_versions: list[int]) -> dict[str, object]:
    episode_rows = []
    for index in range(4):
        episode_rows.append(
            {
                "episode": 256 + index,
                "slot": index,
                "version_staleness": index,
                "control_steps": 20 + index,
                "passed_gates": 1 if index % 2 == 0 else 0,
                "collision": True,
                "max_z_mm": 11.0 + index * 0.1,
                "spawn_z_mm": 11.0,
            }
        )
    return {
        "schema_version": 13,
        "experiment": "flyppy_v3_shared_weight_population",
        "population": 4,
        "curriculum_mode": "boundary-band",
        "vision_runtime": "direct-ray",
        "vision_rays_per_ommatidium": 13,
        "launch_mode": mode,
        "episode_start": 256,
        "episode_end": 259,
        "elapsed_seconds": 2.0,
        "aggregate_control_steps": 86,
        "aggregate_control_steps_per_second": 43.0,
        "control_dt_seconds": 0.0005,
        "episode_results": episode_rows,
        "_source_versions": source_versions,
    }


class SchedulerComparisonTests(unittest.TestCase):
    def test_portable_path_is_relative_inside_repository(self) -> None:
        path = MODULE.ROOT / "reports/flyppy/example.json"
        self.assertEqual(portable(path), "reports/flyppy/example.json")

    def _experiment(self, root: Path, name: str, mode: str, source_versions: list[int]) -> tuple[Path, dict[str, object]]:
        directory = root / name
        directory.mkdir()
        summary = make_summary(mode, source_versions)
        summary.pop("_source_versions")
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        write_jsonl(
            directory / "commit-log.jsonl",
            [
                {
                    "commit_seq": 185 + index,
                    "commit_weight_version": 185 + index,
                    "episode": 256 + index,
                    "slot": index,
                    "source_weight_version": source_versions[index],
                    "staleness": index,
                    "reward_events": 1 if index % 2 == 0 else 0,
                    "aversive_events": 1,
                    "control_steps": 20 + index,
                    "boundary_attempt_index": index,
                    "boundary_batch_number": 3,
                    "launch_mode": mode,
                }
                for index in range(4)
            ],
        )
        return summary_path, summary

    def test_async_and_wave_source_spread_are_compared(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            async_path, async_payload = self._experiment(
                root, "async", "async", [184, 185, 186, 187]
            )
            wave_path, wave_payload = self._experiment(
                root, "wave", "wave", [184, 184, 184, 184]
            )
            validate_pair(async_payload, wave_payload)
            left = summarize_run(async_path, async_payload)
            right = summarize_run(wave_path, wave_payload)

            self.assertEqual(left["source_version_rounds"]["max_source_version_spread"], 3)
            self.assertEqual(right["source_version_rounds"]["max_source_version_spread"], 0)
            self.assertEqual(left["first_gate_successes"], 2)
            self.assertEqual(right["first_gate_successes"], 2)
            text = render_markdown({"left": left, "right": right, "left_fixed": None, "right_fixed": None})
            self.assertIn("async", text)
            self.assertIn("wave", text)
            self.assertIn("control steps/s", text)

    def test_pair_validation_rejects_different_episode_ranges(self) -> None:
        left = make_summary("async", [184, 185, 186, 187])
        right = make_summary("wave", [184, 184, 184, 184])
        left.pop("_source_versions")
        right.pop("_source_versions")
        right["episode_start"] = 260
        with self.assertRaisesRegex(ValueError, "episode_start"):
            validate_pair(left, right)


if __name__ == "__main__":
    unittest.main()
