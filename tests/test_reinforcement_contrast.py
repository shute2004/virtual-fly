from __future__ import annotations

import unittest

import numpy as np

from virtual_fly.training.reinforcement_contrast import (
    compare_effects,
    comparison_from_transaction_contrast,
    summarize_effect,
)


class ReinforcementContrastTests(unittest.TestCase):
    def test_summarize_effect_counts_signs(self) -> None:
        summary = summarize_effect(np.array([0.0, 1e-4, -2e-4, 1e-9]), epsilon=1e-7)
        self.assertEqual(summary["changed_edges"], 2)
        self.assertEqual(summary["positive_edges"], 1)
        self.assertEqual(summary["negative_edges"], 1)
        self.assertAlmostEqual(float(summary["max_abs_delta"]), 2e-4)

    def test_compare_effects_reports_overlap_and_sign(self) -> None:
        reward = np.array([1.0, -2.0, 0.0, 4.0, 0.0])
        aversive = np.array([2.0, 3.0, 5.0, 0.0, 0.0])
        result = compare_effects(reward, aversive, epsilon=1e-9)
        self.assertEqual(result["shared_changed_edges"], 2)
        self.assertEqual(result["reward_only_edges"], 1)
        self.assertEqual(result["aversive_only_edges"], 1)
        self.assertAlmostEqual(float(result["changed_jaccard"]), 0.5)
        self.assertEqual(result["shared_same_sign_edges"], 1)
        self.assertEqual(result["shared_opposite_sign_edges"], 1)

    def test_transaction_contrast_normalizes_bridge_summary(self) -> None:
        result = comparison_from_transaction_contrast(
            {
                "plastic_edges": 10,
                "reward": {
                    "changed_edges": 4,
                    "shift_changed_edges": 3,
                    "bound_changed_edges": 1,
                    "positive_shift_edges": 2,
                    "negative_shift_edges": 1,
                    "sum_shift_delta": 0.3,
                    "sum_abs_shift_delta": 0.7,
                    "max_abs_shift_delta": 0.4,
                },
                "aversive": {
                    "changed_edges": 5,
                    "shift_changed_edges": 4,
                    "bound_changed_edges": 2,
                    "positive_shift_edges": 1,
                    "negative_shift_edges": 3,
                    "sum_shift_delta": -0.2,
                    "sum_abs_shift_delta": 0.8,
                    "max_abs_shift_delta": 0.5,
                },
                "shared_changed_edges": 2,
                "reward_only_changed_edges": 2,
                "aversive_only_changed_edges": 3,
                "shared_shift_edges": 2,
                "shared_same_sign_shift_edges": 1,
                "shared_opposite_sign_shift_edges": 1,
                "shift_dot": 0.0,
                "reward_shift_norm_sq": 0.25,
                "aversive_shift_norm_sq": 1.0,
            }
        )
        self.assertEqual(result["union_changed_edges"], 7)
        self.assertAlmostEqual(float(result["changed_jaccard"]), 2 / 7)
        self.assertAlmostEqual(float(result["shared_same_sign_fraction"]), 0.5)
        self.assertAlmostEqual(float(result["cosine_similarity_on_union"]), 0.0)
        self.assertAlmostEqual(float(dict(result["reward"])["l2_delta"]), 0.5)
        self.assertAlmostEqual(float(dict(result["aversive"])["l2_delta"]), 1.0)

    def test_compare_effects_requires_same_shape(self) -> None:
        with self.assertRaises(ValueError):
            compare_effects(np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
