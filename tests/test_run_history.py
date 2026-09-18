from __future__ import annotations

import unittest

from virtual_fly.reporting.history import build_row


def summary(launch_mode: str) -> dict[str, object]:
    return {
        "experiment": "flyppy_v3_shared_weight_population",
        "environment_version": "v3",
        "flight_body_version": "v3",
        "motor_boundary": "whole-body",
        "backend": "gpu-population:test",
        "population": 4,
        "body_runtime": "packed-process",
        "body_processes": 4,
        "vision_runtime": "direct-ray",
        "vision_rays_per_ommatidium": 13,
        "vision_framebuffer": False,
        "episode_start": 256,
        "episode_end": 279,
        "elapsed_seconds": 12.0,
        "control_dt_seconds": 0.0005,
        "checkpoint_neural_step": 17000,
        "curriculum_mode": "boundary-band",
        "launch_mode": launch_mode,
        "checkpoint_semantics": "global-weights-only-v1",
        "checkpoint_neural_step_semantics": "aggregate-slot-neural-step-count-v1",
        "reproducibility": {
            "code": {"git_sha": "abc123", "dirty": False},
            "dependency_lock": {"composite_sha256": "lock123"},
            "snapshot": {"runtime_semantics": "male-cns-v1-class-dan-v1", "snapshot_sha256": "snap123"},
            "semantics": {"neural_runtime": "signed-activity-local-dynamics-v1", "plasticity": "local-three-factor-dopamine-eligibility-v1"},
            "conditions": {
                "haltere": {"enabled": True, "map_kind": "male-cns-haltere-timing-afferents-inferred-v1", "current_gain": 0.05, "transduction": "interaction-load-v2"}
            },
        },
        "curriculum": {"curriculum_episodes": 280},
        "episode_results": [
            {
                "episode": 256,
                "control_steps": 100,
                "passed_gates": 1,
                "collision": True,
                "finished": False,
                "max_x_mm": 20.0,
                "max_z_mm": 11.5,
                "min_z_mm": 9.0,
                "spawn_z_mm": 11.0,
            }
        ],
    }


class RunHistoryTests(unittest.TestCase):
    def test_launch_mode_is_recorded_and_disambiguates_run_key(self) -> None:
        async_row = build_row(summary("async"))
        wave_row = build_row(summary("wave"))
        self.assertEqual(async_row["launch_mode"], "async")
        self.assertEqual(wave_row["launch_mode"], "wave")
        self.assertNotEqual(async_row["run_key"], wave_row["run_key"])

    def test_reproducibility_fields_are_preserved(self) -> None:
        row = build_row(summary("async"))
        self.assertEqual(row["flight_body_version"], "v3")
        self.assertEqual(row["haltere_sensory_kind"], "male-cns-haltere-timing-afferents-inferred-v1")
        self.assertEqual(row["snapshot_semantics"], "male-cns-v1-class-dan-v1")
        self.assertEqual(row["neural_runtime_semantics"], "signed-activity-local-dynamics-v1")
        self.assertEqual(row["git_sha"], "abc123")
        self.assertEqual(row["checkpoint_neural_step_semantics"], "aggregate-slot-neural-step-count-v1")

    def test_legacy_summary_leaves_launch_mode_blank(self) -> None:
        payload = summary("async")
        payload.pop("launch_mode")
        row = build_row(payload)
        self.assertEqual(row["launch_mode"], "")


if __name__ == "__main__":
    unittest.main()
