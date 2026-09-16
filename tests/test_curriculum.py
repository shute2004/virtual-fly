from __future__ import annotations

from collections import Counter
import unittest

from virtual_fly.training.curriculum import (
    BoundaryBandConfig,
    SpawnCondition,
    boundary_condition_for_attempt,
    current_boundary_condition,
    next_boundary_attempt_for_group,
    record_boundary_result,
)


def config(*, seed: int = 0) -> BoundaryBandConfig:
    return BoundaryBandConfig(
        hard=SpawnCondition(10.500, 5.2900, 350.0),
        easy=SpawnCondition(10.625, 5.3525, 356.25),
        target=SpawnCondition(9.0, 5.0, 300.0),
        batch_size=24,
        harden_success_rate=0.80,
        ease_success_rate=0.40,
        harden_step=SpawnCondition(0.125, 0.0625, 6.25),
        ease_step=SpawnCondition(0.0625, 0.03125, 3.125),
        seed=seed,
    )


class BoundaryBandCurriculumTests(unittest.TestCase):
    def test_24_episode_band_has_expected_level_counts(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        levels: list[float] = []
        for episode in range(24):
            _, level = current_boundary_condition(state, config())
            levels.append(level)
            # 12/24 keeps the band in the hold region.
            record_boundary_result(state, config(), success=episode < 12)

        self.assertEqual(
            Counter(levels),
            Counter({0.0: 4, 0.25: 5, 0.5: 6, 0.75: 5, 1.0: 4}),
        )
        band = state["boundary_band"]
        self.assertEqual(band["last_adjustment"], "hold")
        self.assertAlmostEqual(band["last_batch_success_rate"], 0.5)
        self.assertEqual(state["boundary_batch_number"], 1)
        self.assertIsNone(state["boundary_ease_level"])

    def test_async_launch_schedule_is_independent_of_completion_order(self) -> None:
        launched_state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        levels = [
            boundary_condition_for_attempt(launched_state, config(), attempt)[1]
            for attempt in range(24)
        ]
        self.assertEqual(
            Counter(levels),
            Counter({0.0: 4, 0.25: 5, 0.5: 6, 0.75: 5, 1.0: 4}),
        )

        outcomes = [(index, index < 12) for index in range(24)]
        state_a = dict(launched_state)
        state_a["boundary_band"] = dict(launched_state["boundary_band"])
        state_b = dict(launched_state)
        state_b["boundary_band"] = dict(launched_state["boundary_band"])
        for attempt, success in outcomes:
            record_boundary_result(
                state_a,
                config(),
                success=success,
                attempt_index=attempt,
            )
        for attempt, success in reversed(outcomes):
            record_boundary_result(
                state_b,
                config(),
                success=success,
                attempt_index=attempt,
            )

        band_a = state_a["boundary_band"]
        band_b = state_b["boundary_band"]
        self.assertEqual(band_a["last_adjustment"], "hold")
        self.assertEqual(band_b["last_adjustment"], "hold")
        self.assertEqual(band_a["last_batch_success_rate"], 0.5)
        self.assertEqual(band_b["last_batch_success_rate"], 0.5)
        self.assertEqual(band_a["hard"], band_b["hard"])
        self.assertEqual(band_a["easy"], band_b["easy"])
        self.assertEqual(state_a["successful_first_gates"], state_b["successful_first_gates"])
        self.assertEqual(state_a["consecutive_failures"], 0)
        self.assertEqual(state_b["consecutive_failures"], 0)

    def test_async_group_schedule_balances_ease_levels_across_course_seeds(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        by_group: dict[int, list[float]] = {group: [] for group in range(4)}
        all_levels: list[float] = []
        for attempt in range(24):
            _, level = boundary_condition_for_attempt(
                state,
                config(),
                attempt,
                group_count=4,
            )
            all_levels.append(level)
            by_group[attempt % 4].append(level)

        self.assertEqual(
            Counter(all_levels),
            Counter({0.0: 4, 0.25: 5, 0.5: 6, 0.75: 5, 1.0: 4}),
        )
        self.assertEqual({group: len(levels) for group, levels in by_group.items()}, {0: 6, 1: 6, 2: 6, 3: 6})
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            counts = [Counter(by_group[group])[level] for group in range(4)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_async_group_schedule_stays_balanced_across_batches_and_seeds(self) -> None:
        for seed in (0, 1, 7, 42, 104729):
            for batch_number in range(5):
                state: dict[str, object] = {
                    "successful_first_gates": 0,
                    "consecutive_failures": 0,
                    "boundary_band": {
                        "batch_number": batch_number,
                        "attempts_in_batch": 0,
                        "successes_in_batch": 0,
                        "completed_attempt_indices": [],
                        "group_attempts": {},
                        "group_successes": {},
                        "last_batch_success_rate": None,
                        "last_batch_raw_success_rate": None,
                        "last_batch_group_success_rates": {},
                        "last_batch_aggregation": None,
                        "last_adjustment": None,
                        "harder_shifts": 0,
                        "easier_shifts": 0,
                        "hard": {"x_mm": 10.5, "z_mm": 5.29, "speed_mm_s": 350.0},
                        "easy": {"x_mm": 10.625, "z_mm": 5.3525, "speed_mm_s": 356.25},
                    },
                }
                cfg = config(seed=seed)
                by_group: dict[int, Counter[float]] = {
                    group: Counter() for group in range(4)
                }
                for attempt in range(cfg.batch_size):
                    _, level = boundary_condition_for_attempt(
                        state,
                        cfg,
                        attempt,
                        group_count=4,
                    )
                    by_group[attempt % 4][level] += 1
                for level in (0.0, 0.25, 0.5, 0.75, 1.0):
                    counts = [by_group[group][level] for group in range(4)]
                    self.assertLessEqual(max(counts) - min(counts), 1)

    def test_async_group_assignment_is_static_and_resume_safe(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        issued: set[int] = set()
        assigned: dict[int, list[int]] = {group: [] for group in range(4)}
        while len(issued) < 24:
            progressed = False
            for group in range(4):
                attempt = next_boundary_attempt_for_group(
                    state,
                    config(),
                    group_index=group,
                    group_count=4,
                    issued_attempts=issued,
                )
                if attempt is None:
                    continue
                self.assertEqual(attempt % 4, group)
                issued.add(attempt)
                assigned[group].append(attempt)
                progressed = True
            self.assertTrue(progressed)

        self.assertEqual(sorted(issued), list(range(24)))
        self.assertEqual(assigned[0], [0, 4, 8, 12, 16, 20])
        self.assertEqual(assigned[1], [1, 5, 9, 13, 17, 21])
        self.assertEqual(assigned[2], [2, 6, 10, 14, 18, 22])
        self.assertEqual(assigned[3], [3, 7, 11, 15, 19, 23])

        resumed: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        for attempt in range(8):
            record_boundary_result(
                resumed,
                config(),
                success=False,
                attempt_index=attempt,
                group=attempt % 4,
            )
        self.assertEqual(
            [
                next_boundary_attempt_for_group(
                    resumed,
                    config(),
                    group_index=group,
                    group_count=4,
                )
                for group in range(4)
            ],
            [8, 9, 10, 11],
        )

    def test_boundary_attempt_index_must_stay_inside_current_batch(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        with self.assertRaises(ValueError):
            boundary_condition_for_attempt(state, config(), 24)

    def test_explicit_attempt_identity_prevents_duplicate_recording(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        record_boundary_result(
            state,
            config(),
            success=True,
            attempt_index=7,
        )
        band = state["boundary_band"]
        self.assertEqual(band["attempts_in_batch"], 1)
        self.assertEqual(band["completed_attempt_indices"], [7])
        with self.assertRaises(ValueError):
            record_boundary_result(
                state,
                config(),
                success=False,
                attempt_index=7,
            )

    def test_group_weighting_prevents_fast_group_from_dominating_batch(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        completed = None
        # One slow group succeeds once while one fast group contributes 23
        # failures.  Raw episode rate is 1/24, but equal-group aggregation is
        # (1.0 + 0.0) / 2 = 0.5 and therefore holds this band.
        for attempt in range(24):
            completed = record_boundary_result(
                state,
                config(),
                success=attempt == 0,
                attempt_index=attempt,
                group=0 if attempt == 0 else 1,
            )
        self.assertIsNotNone(completed)
        self.assertEqual(completed["aggregation"], "equal_group_mean")
        self.assertAlmostEqual(completed["raw_success_rate"], 1 / 24)
        self.assertAlmostEqual(completed["success_rate"], 0.5)
        self.assertEqual(completed["group_success_rates"], {"0": 1.0, "1": 0.0})
        self.assertEqual(completed["adjustment"], "hold")

    def test_high_success_shifts_whole_band_toward_target(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        completed = None
        for _ in range(24):
            current_boundary_condition(state, config())
            completed = record_boundary_result(state, config(), success=True)

        self.assertIsNotNone(completed)
        self.assertEqual(completed["adjustment"], "harder")
        self.assertAlmostEqual(completed["hard"]["x_mm"], 10.375)
        self.assertAlmostEqual(completed["easy"]["x_mm"], 10.500)
        self.assertAlmostEqual(completed["hard"]["z_mm"], 5.2275)
        self.assertAlmostEqual(completed["easy"]["z_mm"], 5.2900)
        self.assertAlmostEqual(completed["hard"]["speed_mm_s"], 343.75)
        self.assertAlmostEqual(completed["easy"]["speed_mm_s"], 350.0)

    def test_low_success_can_recover_beyond_initial_easy_endpoint(self) -> None:
        state: dict[str, object] = {
            "successful_first_gates": 0,
            "consecutive_failures": 0,
        }
        completed = None
        for _ in range(24):
            current_boundary_condition(state, config())
            completed = record_boundary_result(state, config(), success=False)

        self.assertIsNotNone(completed)
        self.assertEqual(completed["adjustment"], "easier")
        self.assertAlmostEqual(completed["hard"]["x_mm"], 10.5625)
        self.assertAlmostEqual(completed["easy"]["x_mm"], 10.6875)
        self.assertAlmostEqual(completed["hard"]["z_mm"], 5.32125)
        self.assertAlmostEqual(completed["easy"]["z_mm"], 5.38375)
        self.assertAlmostEqual(completed["hard"]["speed_mm_s"], 353.125)
        self.assertAlmostEqual(completed["easy"]["speed_mm_s"], 359.375)


if __name__ == "__main__":
    unittest.main()
