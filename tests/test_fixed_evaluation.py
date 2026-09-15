import unittest

from virtual_fly.training.fixed_evaluation import (
    FIXED_EVAL_SUITE_V1,
    SUITE_VERSION,
    render_markdown,
    summarize_suite,
)


def row(condition: str, *, passed: int, collision: bool, gain: float, loss: float):
    return {
        "condition": condition,
        "course_seed": 0,
        "slot": 0,
        "control_steps": 10,
        "passed_gates": passed,
        "collision": collision,
        "collision_reason": "gate" if collision else None,
        "finished": False,
        "max_x_mm": 12.0,
        "min_z_mm": 8.0,
        "max_z_mm": 10.0,
        "max_altitude_gain_mm": gain,
        "max_altitude_loss_mm": loss,
        "final_vx_mm_s": 250.0,
    }


class FixedEvaluationTests(unittest.TestCase):
    def test_fixed_suite_is_named_and_ordered(self):
        self.assertEqual(SUITE_VERSION, "flyppy-v3-fixed-v1")
        self.assertEqual(
            [condition.name for condition in FIXED_EVAL_SUITE_V1],
            ["part3_frontier", "midpoint", "target"],
        )

    def test_summarize_suite_counts_gate_levels_independently(self):
        rows = []
        for condition in FIXED_EVAL_SUITE_V1:
            rows.extend(
                [
                    row(condition.name, passed=0, collision=True, gain=-0.1, loss=1.0),
                    row(condition.name, passed=1, collision=True, gain=0.2, loss=0.5),
                    row(condition.name, passed=2, collision=False, gain=0.4, loss=0.2),
                    row(condition.name, passed=3, collision=False, gain=0.6, loss=0.1),
                ]
            )

        summaries = summarize_suite(rows)
        self.assertEqual(len(summaries), 3)
        for summary in summaries:
            self.assertEqual(summary["episodes"], 4)
            self.assertEqual(summary["first_gate_passes"], 3)
            self.assertEqual(summary["first_gate_pass_rate"], 0.75)
            self.assertEqual(summary["second_gate_passes"], 2)
            self.assertEqual(summary["second_gate_pass_rate"], 0.5)
            self.assertEqual(summary["collisions"], 2)
            self.assertEqual(summary["collision_rate"], 0.5)
            self.assertEqual(summary["max_passed_gates"], 3)

    def test_render_markdown_records_read_only_contract(self):
        rows = [
            row(condition.name, passed=0, collision=True, gain=-0.1, loss=1.0)
            for condition in FIXED_EVAL_SUITE_V1
        ]
        payload = {
            "suite_version": SUITE_VERSION,
            "checkpoint": "checkpoint",
            "checkpoint_neural_step": 10,
            "global_weight_version": 2,
            "global_weight_version_unchanged": True,
            "backend": "test",
            "population": 1,
            "seed_start": 0,
            "seed_end": 0,
            "vision_runtime": "direct-ray",
            "vision_rays_per_ommatidium": 13,
            "plasticity": False,
            "transaction_dirty_edges_after": {0: 0},
            "elapsed_seconds": 1.0,
            "condition_summaries": summarize_suite(rows),
            "episode_results": rows,
        }
        text = render_markdown(payload)
        self.assertIn("plasticity: `false`", text)
        self.assertIn("global weight version unchanged: `true`", text)
        self.assertIn("PAM刺激", text)
        self.assertIn("PPL刺激", text)


if __name__ == "__main__":
    unittest.main()
