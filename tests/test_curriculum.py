from __future__ import annotations

from collections import Counter
import unittest

from virtual_fly.training.curriculum import (
    BoundaryBandConfig,
    SpawnCondition,
    current_boundary_condition,
    record_boundary_result,
)


def config() -> BoundaryBandConfig:
    return BoundaryBandConfig(
        hard=SpawnCondition(10.500, 5.2900, 350.0),
        easy=SpawnCondition(10.625, 5.3525, 356.25),
        target=SpawnCondition(9.0, 5.0, 300.0),
        batch_size=24,
        harden_success_rate=0.80,
        ease_success_rate=0.40,
        harden_step=SpawnCondition(0.125, 0.0625, 6.25),
        ease_step=SpawnCondition(0.0625, 0.03125, 3.125),
        seed=0,
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
