#!/usr/bin/env python3
"""Evaluate one learned checkpoint across gate-2 heights without plasticity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

from flyppy_packed_body_worker import spawn_packed_body_processes
from population_neural_bridge_client import PopulationNeuralBridgeClient
from preview_flyppy_best import DEFAULT_CALIBRATION, absolute, ensure_runtime_environment, load_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=Path("artifacts/experiments/flyppy-v7-gate2-height-v2"))
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    p.add_argument("--output", type=Path, default=Path("reports/flyppy/diagnostics/frozen-gate2-heights-v960.json"))
    p.add_argument("--heights", type=float, nargs="+", default=[13.5, 13.625, 13.75, 14.0, 14.282297335224168])
    p.add_argument("--max-control-steps", type=int, default=600)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    production = absolute(a.production)
    snapshot = absolute(a.snapshot)
    output = absolute(a.output)
    ensure_runtime_environment(absolute(DEFAULT_CALIBRATION))
    os.environ["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    os.environ["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"
    gv = int(load_json(production / "population-state.json")["global_weight_version"])
    config = {
        "seed": 0,
        "fixed_course_seed": 0,
        "gate_count": 6,
        "environment_version": "v7",
        "flight_body_version": "v7",
        "wing_motor_map": str(snapshot / "wing-motor-neurons-v0.json"),
        "body_motor_map": str(snapshot / "body-motor-neurons-v0.json"),
        "retinotopic_map": str(snapshot / "retinotopic-vision-v1.json"),
        "haltere_sensory_map": str(snapshot / "haltere-timing-afferents-v1.json"),
        "photoreceptor_current_gain": 2.0,
        "haltere_current_gain": 0.05,
        "haltere_transduction": "interaction-load-v2",
    }
    workers = []
    results = []
    try:
        workers, slotmap = spawn_packed_body_processes(
            population=1, process_count=1, physics_steps=10, timeout_s=120.0, config=config
        )
        worker = slotmap[0]
        motor_ids = tuple(int(v) for v in worker.body_ids[0])
        with PopulationNeuralBridgeClient(
            snapshot=snapshot,
            groups=snapshot / "embodiment-groups-v0.json",
            slots=1,
            repo_root=ROOT,
        ) as brain:
            brain.load_checkpoint(production / "checkpoint", global_weight_version=gv)
            for height in a.heights:
                version = brain.restart_slot(0)
                if version != gv:
                    raise RuntimeError(f"restart version {version} != {gv}")
                worker.request(
                    "reset",
                    {0: {
                        "episode": -1,
                        "source_weight_version": gv,
                        "spawn_x_mm": 8.83575,
                        "spawn_z_mm": 11.2286819148691,
                        "initial_speed_mm_s": 400.0,
                        "initial_vz_mm_s": 0.0,
                        "gate_center_overrides": {1: float(height)},
                    }},
                )
                worker.receive("reset")
                passed = 0
                max_abs_y = 0.0
                max_z = float("-inf")
                terminal = "step_limit"
                collision_reason = None
                controls = 0
                for control in range(a.max_control_steps):
                    obs = worker.call("observe", (0,))[0]
                    spikes = brain.step_batch(
                        [{"slot": 0, "stimulate_body": obs["body_currents"], "read_body": motor_ids}],
                        plasticity=False,
                    ).get(0, {})
                    active = tuple(body_id for body_id in motor_ids if spikes.get(body_id, False))
                    act = worker.call("act", {0: active})[0]
                    controls = control + 1
                    y = float(act["position"][1]); z = float(act["position"][2])
                    max_abs_y = max(max_abs_y, abs(y)); max_z = max(max_z, z)
                    if act["passed_gate"]:
                        passed += 1
                        brain.step_slot(0, stimulate={"reward_dan": 2.0}, plasticity=False, steps=4)
                        if passed >= 2:
                            terminal = "gate2_pass"
                            break
                    if act["collision"]:
                        terminal = "collision"
                        collision_reason = act.get("collision_reason")
                        break
                    if act["finished"]:
                        terminal = "finished"
                        break
                row = {
                    "gate2_center_z_mm": float(height),
                    "passed_gates": passed,
                    "success": passed >= 2,
                    "terminal": terminal,
                    "collision_reason": collision_reason,
                    "control_steps": controls,
                    "max_abs_y_mm": max_abs_y,
                    "max_z_mm": max_z,
                }
                results.append(row)
                print(json.dumps(row, separators=(",", ":")), flush=True)
    finally:
        for worker in workers:
            try:
                worker.close()
            except Exception:
                pass
    payload = {"schema_version": 1, "global_weight_version": gv, "plasticity": False, "haltere_gain": 0.05, "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
