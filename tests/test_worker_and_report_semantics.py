from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts/embodiment/flyppy_body_worker.py"
SPEC = importlib.util.spec_from_file_location("flyppy_body_worker_for_test", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class WorkerAndReportSemanticsTests(unittest.TestCase):
    def test_process_worker_preserves_nondefault_body_and_periphery_settings(self) -> None:
        config = {
            "snapshot": "/tmp/snapshot",
            "seed": 3,
            "gate_count": 6,
            "environment_version": "v7",
            "flight_body_version": "v7",
            "vertical_steering_gain": 0.8,
            "measured_steering_gain": 0.7,
            "neutral_trim_strength": 0.63,
            "steering_tau_ms": 18.5,
            "steering_spike_increment": 0.42,
            "wing_motor_map": "/tmp/wing.json",
            "body_motor_map": "/tmp/body.json",
            "retinotopic_map": "/tmp/retina.json",
            "haltere_sensory_map": "/tmp/haltere.json",
            "haltere_sensory_kind": "male-cns-haltere-timing-afferents-inferred-v1",
            "photoreceptor_current_gain": 2.0,
            "haltere_current_gain": 0.05,
            "haltere_transduction": "interaction-load-v2",
        }
        args = WORKER._worker_args(config)
        self.assertEqual(args.neutral_trim_strength, 0.63)
        self.assertEqual(args.steering_tau_ms, 18.5)
        self.assertEqual(args.steering_spike_increment, 0.42)
        self.assertEqual(args.haltere_sensory_kind, "male-cns-haltere-timing-afferents-inferred-v1")
        self.assertEqual(args.snapshot, Path("/tmp/snapshot"))

    def test_gate2_height_report_does_not_describe_adaptive_or_boundary_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            output = root / "report"
            summary.write_text(
                json.dumps(
                    {
                        "backend": "test",
                        "curriculum_mode": "gate2-height",
                        "launch_mode": "async",
                        "elapsed_seconds": 1.0,
                        "checkpoint_neural_step": 42,
                        "checkpoint_neural_step_semantics": "aggregate-slot-neural-step-count-v1",
                        "episode_start": 10,
                        "episode_end": 10,
                        "curriculum": {
                            "curriculum_mode": "gate2-height",
                            "curriculum_episodes": 11,
                            "boundary_band": {"batch_number": 99},
                            "gate_height_curriculum": {
                                "gate_index": 1,
                                "frontier_center_z_mm": 13.5,
                                "target_center_z_mm": 14.2,
                                "step_mm": 0.125,
                                "batch_number": 4,
                                "role_attempts": {"current": 16, "review": 8},
                                "role_successes": {"current": 13, "review": 7},
                                "last_role_success_rates": {"current": 0.8125, "review": 0.875},
                                "mastered_centers_z_mm": [12.75, 13.0],
                                "last_adjustment": "advance",
                                "advances": 2,
                                "curriculum_complete": False,
                            },
                        },
                        "episode_results": [
                            {
                                "episode": 10,
                                "spawn_x_mm": 8.8,
                                "spawn_z_mm": 11.2,
                                "initial_speed_mm_s": 400.0,
                                "control_steps": 100,
                                "passed_gates": 2,
                                "collision": False,
                                "finished": True,
                                "max_x_mm": 30.0,
                                "min_z_mm": 11.0,
                                "max_z_mm": 13.8,
                                "final_vx_mm_s": 250.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/analysis/export_flyppy_report.py"),
                    "--summary",
                    str(summary),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            text = (output / "latest.md").read_text(encoding="utf-8")
            self.assertIn("gate2-height criterion", text)
            self.assertIn("## Gate2-heightカリキュラム", text)
            self.assertNotIn("adaptive training criterion", text)
            self.assertNotIn("## 境界帯カリキュラム", text)


if __name__ == "__main__":
    unittest.main()
