from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/analyze_flyppy_async_bias.py"
SPEC = importlib.util.spec_from_file_location("analyze_flyppy_async_bias", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
annotate_boundary_batches = MODULE.annotate_boundary_batches
resolve_analysis_watermark = MODULE.resolve_analysis_watermark
source_version_round_summary = MODULE.source_version_round_summary


def row(
    slot: int,
    attempt: int,
    source_version: int,
    *,
    batch: int | None = 0,
    episode: int | None = None,
    launch_round: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "slot": slot,
        "episode": attempt if episode is None else episode,
        "boundary_attempt_index": attempt,
        "source_weight_version": source_version,
    }
    if batch is not None:
        payload["boundary_batch_number"] = batch
    if launch_round is not None:
        payload["launch_round"] = launch_round
    return payload


class SourceVersionRoundSummaryTests(unittest.TestCase):
    def test_population_state_supplies_default_persisted_watermark(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commit_log = root / "commit-log.jsonl"
            commit_log.write_text("", encoding="utf-8")
            (root / "population-state.json").write_text(
                json.dumps({"global_weight_version": 184}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(resolve_analysis_watermark(commit_log, None), 184)
            self.assertEqual(resolve_analysis_watermark(commit_log, 160), 160)

    def test_stored_launch_round_takes_priority_over_boundary_grouping(self) -> None:
        rows = [
            row(slot, slot + 8, 100, batch=7, launch_round=0)
            for slot in range(4)
        ] + [
            row(slot, slot, 104, batch=8, launch_round=1)
            for slot in range(4)
        ]
        summary = source_version_round_summary(rows)
        self.assertEqual(summary["complete_rounds"], 2)
        self.assertEqual(summary["mean_source_version_spread"], 0)
        self.assertEqual(summary["max_source_version_spread"], 0)
        self.assertEqual(summary["zero_spread_rounds"], 2)

    def test_partial_launch_round_is_not_counted_as_complete(self) -> None:
        rows = [
            row(slot, slot, 100, batch=7, launch_round=0)
            for slot in range(4)
        ] + [
            row(slot, 4 + slot, 104, batch=7, launch_round=1)
            for slot in range(2)
        ]
        summary = source_version_round_summary(rows)
        self.assertEqual(summary["complete_rounds"], 1)
        self.assertEqual(summary["zero_spread_rounds"], 1)

    def test_wave_rounds_have_zero_source_version_spread(self) -> None:
        rows = [
            *(row(slot, slot, 100, batch=7) for slot in range(4)),
            *(row(slot, 4 + slot, 104, batch=7) for slot in range(4)),
        ]
        summary = source_version_round_summary(rows)
        self.assertEqual(summary["complete_rounds"], 2)
        self.assertEqual(summary["mean_source_version_spread"], 0)
        self.assertEqual(summary["max_source_version_spread"], 0)
        self.assertEqual(summary["zero_spread_rounds"], 2)

    def test_async_rounds_measure_mixed_source_versions(self) -> None:
        versions = [100, 101, 101, 102, 103, 104, 105, 105]
        rows = [
            row(index % 4, index, version, batch=7)
            for index, version in enumerate(versions)
        ]
        summary = source_version_round_summary(rows)
        self.assertEqual(summary["complete_rounds"], 2)
        self.assertEqual(summary["mean_source_version_spread"], 2)
        self.assertEqual(summary["max_source_version_spread"], 2)
        self.assertEqual(summary["zero_spread_rounds"], 0)

    def test_historical_single_batch_without_batch_number_is_supported(self) -> None:
        rows = [row(slot, slot, 100 + slot, batch=None) for slot in range(4)]
        summary = source_version_round_summary(rows)
        self.assertEqual(summary["complete_rounds"], 1)
        self.assertEqual(summary["max_source_version_spread"], 3)

    def test_repeated_historical_attempt_ids_are_split_by_episode_order(self) -> None:
        rows = [
            *(row(slot, slot, 100, batch=None, episode=slot) for slot in range(4)),
            *(row(slot, slot, 104, batch=None, episode=4 + slot) for slot in range(4)),
        ]
        summary = source_version_round_summary(rows)
        self.assertEqual(summary["complete_rounds"], 2)
        self.assertEqual(summary["mean_source_version_spread"], 0)
        self.assertEqual(summary["max_source_version_spread"], 0)

    def test_preannotating_full_history_keeps_batch_identity_in_straddling_window(self) -> None:
        rows = [
            *(row(slot, slot, 100 + slot, batch=None, episode=slot) for slot in range(4)),
            *(row(slot, slot, 200, batch=None, episode=4 + slot) for slot in range(4)),
        ]
        annotated = annotate_boundary_batches(rows)
        summary = source_version_round_summary(annotated[-6:])
        self.assertEqual(summary["complete_rounds"], 1)
        self.assertEqual(summary["mean_source_version_spread"], 0)
        self.assertEqual(summary["max_source_version_spread"], 0)


if __name__ == "__main__":
    unittest.main()
