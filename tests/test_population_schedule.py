from __future__ import annotations

import unittest

from virtual_fly.training.population_schedule import checkpoint_can_flush, launch_round_for_index


class PopulationScheduleTests(unittest.TestCase):
    def test_24_launches_form_six_rounds_at_population_four(self) -> None:
        rounds = [launch_round_for_index(index, 4) for index in range(24)]
        self.assertEqual(rounds, [round for round in range(6) for _ in range(4)])

    def test_partial_final_round_is_numbered_consistently(self) -> None:
        rounds = [launch_round_for_index(index, 4) for index in range(10)]
        self.assertEqual(rounds, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    def test_wave_checkpoint_waits_for_round_boundary(self) -> None:
        self.assertFalse(
            checkpoint_can_flush(
                launch_mode="wave",
                checkpoint_pending=True,
                active_slots=3,
            )
        )
        self.assertTrue(
            checkpoint_can_flush(
                launch_mode="wave",
                checkpoint_pending=True,
                active_slots=0,
            )
        )

    def test_async_checkpoint_may_flush_with_active_slots(self) -> None:
        self.assertTrue(
            checkpoint_can_flush(
                launch_mode="async",
                checkpoint_pending=True,
                active_slots=3,
            )
        )

    def test_no_pending_checkpoint_never_flushes(self) -> None:
        for mode in ("async", "wave"):
            self.assertFalse(
                checkpoint_can_flush(
                    launch_mode=mode,
                    checkpoint_pending=False,
                    active_slots=0,
                )
            )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            launch_round_for_index(-1, 4)
        with self.assertRaises(ValueError):
            launch_round_for_index(0, 0)
        with self.assertRaises(ValueError):
            checkpoint_can_flush(
                launch_mode="lockstep",
                checkpoint_pending=True,
                active_slots=0,
            )
        with self.assertRaises(ValueError):
            checkpoint_can_flush(
                launch_mode="wave",
                checkpoint_pending=True,
                active_slots=-1,
            )


if __name__ == "__main__":
    unittest.main()
