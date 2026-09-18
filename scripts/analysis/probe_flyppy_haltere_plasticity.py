#!/usr/bin/env python3
"""Probe how haltere timing currents change MaleCNS plasticity on one fixed Flyppy trajectory.

The physical trajectory is generated once from the v7 checkpoint with retinal input only and
plasticity disabled. The worker still measures physical haltere load, so the exact same
trajectory can then be replayed neurally as either retina-only or retina+haltere.

All replay slots receive the real first-gate PAM event. Immediately before the second-gate
terminal event, each condition is split into control/PAM/PPL slots and transaction deltas are
compared. No slot is committed and no checkpoint is written.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "scripts" / "embodiment"
if str(EMB) not in sys.path:
    sys.path.insert(0, str(EMB))

from flyppy_packed_body_worker import spawn_packed_body_processes  # noqa: E402
from population_neural_bridge_client import PopulationNeuralBridgeClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--production", type=Path, default=Path("artifacts/experiments/flyppy-v7-gate2-height-v2"))
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/malecns-v1.0"))
    p.add_argument("--gate2-center-z-mm", type=float, default=13.5)
    p.add_argument("--haltere-gain", type=float, default=0.05)
    p.add_argument("--max-control-steps", type=int, default=400)
    p.add_argument("--reinforcement-steps", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("reports/flyppy/diagnostics/haltere-plasticity-v960.json"))
    return p.parse_args()


def abspath(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    a = parse_args()
    prod = abspath(a.production)
    snap = abspath(a.snapshot)
    out = abspath(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads((prod / "population-state.json").read_text())
    gv = int(state["global_weight_version"])
    calibration = json.loads((ROOT / "artifacts/embodiment/neural-runtime-calibration-v1.json").read_text())
    os.environ["VF_NEURAL_SYNAPSE_SCALE"] = str(float(calibration["synapse_scale"]))
    os.environ["VF_FLYPPY_VISION_MODE"] = "direct-ray"
    os.environ["VF_FLYPPY_OMMATIDIA_RAYS"] = "13"

    config = {
        "seed": 0,
        "fixed_course_seed": 0,
        "gate_count": 6,
        "environment_version": "v7",
        "flight_body_version": "v7",
        "neutral_trim_strength": 1.0,
        "steering_tau_ms": 12.0,
        "steering_spike_increment": 0.85,
        "wing_motor_map": str(snap / "wing-motor-neurons-v0.json"),
        "body_motor_map": str(snap / "body-motor-neurons-v0.json"),
        "retinotopic_map": str(snap / "retinotopic-vision-v1.json"),
        "photoreceptor_current_gain": 2.0,
        "haltere_sensory_map": str(snap / "haltere-timing-afferents-v1.json"),
        "haltere_current_gain": a.haltere_gain,
        "haltere_transduction": "interaction-load-v2",
        "capture_physics_trace": False,
    }

    workers = []
    try:
        workers, slotmap = spawn_packed_body_processes(
            population=1,
            process_count=1,
            physics_steps=10,
            timeout_s=120.0,
            config=config,
        )
        worker = slotmap[0]
        motor_ids = tuple(int(x) for x in worker.body_ids[0])
        body_ids_snapshot = np.fromfile(snap / "body_ids.u64le", dtype="<u8")
        neurotransmitters = np.fromfile(snap / "neurotransmitters.u8", dtype=np.uint8)
        if len(body_ids_snapshot) != len(neurotransmitters):
            raise RuntimeError("MaleCNS body/neurotransmitter arrays are misaligned")
        all_dan_ids = tuple(int(v) for v in body_ids_snapshot[neurotransmitters == 4])
        groups_payload = json.loads((snap / "embodiment-groups-v0.json").read_text())
        reward_dan_ids = frozenset(int(v) for v in groups_payload["groups"]["reward_dan"]["body_ids"])
        aversive_dan_ids = frozenset(int(v) for v in groups_payload["groups"]["aversive_dan"]["body_ids"])
        read_ids = tuple(dict.fromkeys((*motor_ids, *all_dan_ids)))
        annotations = pd.read_feather(snap / "annotations.feather")
        dan_type_by_id = {
            int(row.bodyId): ("<untyped>" if pd.isna(row.type) or not str(row.type) else str(row.type))
            for row in annotations.itertuples(index=False)
            if int(row.bodyId) in set(all_dan_ids)
        }
        with PopulationNeuralBridgeClient(
            snapshot=snap,
            groups=snap / "embodiment-groups-v0.json",
            slots=9,
            repo_root=ROOT,
        ) as brain:
            loaded = brain.load_checkpoint(prod / "checkpoint", global_weight_version=gv)
            brain.restart_slot(0)
            reset = {
                "episode": -1,
                "source_weight_version": gv,
                "spawn_x_mm": 8.83575,
                "spawn_z_mm": 11.2286819148691,
                "initial_speed_mm_s": 400.0,
                "initial_vz_mm_s": 0.0,
                "gate_center_overrides": {1: a.gate2_center_z_mm},
            }
            worker.request("reset", {0: reset})
            worker.receive("reset")

            history: list[dict[str, object]] = []
            gate_pass_steps: list[int] = []
            passed = 0
            terminal: dict[str, object] | None = None
            for step in range(a.max_control_steps):
                obs = worker.call("observe", (0,))[0]
                retinal = [[int(i), float(v)] for i, v in obs["retinal_body_currents"]]
                haltere = [[int(i), float(v)] for i, v in obs["haltere_body_currents"]]
                row: dict[str, object] = {
                    "retinal": retinal,
                    "haltere": haltere,
                    "haltere_active_sensilla": int(obs["haltere_active_sensilla"]),
                    "haltere_mean_current": float(obs["haltere_mean_current"]),
                    "haltere_max_current": float(obs["haltere_max_current"]),
                    "event": None,
                }
                history.append(row)
                spikes = brain.step_batch(
                    [{"slot": 0, "stimulate_body": retinal, "read_body": read_ids}],
                    plasticity=False,
                ).get(0, {})
                active_dan = tuple(body_id for body_id in all_dan_ids if spikes.get(body_id, False))
                row["active_dan_count"] = len(active_dan)
                row["active_reward_dan_count"] = sum(body_id in reward_dan_ids for body_id in active_dan)
                row["active_aversive_dan_count"] = sum(body_id in aversive_dan_ids for body_id in active_dan)
                row["active_other_dan_count"] = len(active_dan) - int(row["active_reward_dan_count"]) - int(row["active_aversive_dan_count"])
                active = tuple(body_id for body_id in motor_ids if spikes.get(body_id, False))
                act = worker.call("act", {0: active})[0]
                if act["passed_gate"]:
                    passed += 1
                    gate_pass_steps.append(step)
                    row["event"] = "gate_pass"
                    # Production semantics: the gate pass still drives DAN neural activity;
                    # plasticity is disabled only for trajectory capture.
                    brain.step_slot(
                        0,
                        stimulate={"reward_dan": 2.0},
                        plasticity=False,
                        steps=a.reinforcement_steps,
                    )
                    if passed >= 2:
                        terminal = {
                            "reason": "gate_pass",
                            "step": step + 1,
                            "position": list(act["position"]),
                            "next_gate": int(act["next_gate"]),
                        }
                        break
                if act["collision"] or act["finished"]:
                    row["event"] = "collision" if act["collision"] else "finished"
                    terminal = {
                        "reason": row["event"],
                        "collision_reason": act.get("collision_reason"),
                        "gate_miss_distance_mm": float(act.get("gate_miss_distance_mm") or 0.0),
                        "step": step + 1,
                        "position": list(act["position"]),
                        "next_gate": int(act["next_gate"]),
                    }
                    break
            if not gate_pass_steps:
                raise RuntimeError(f"fixed v960 trajectory never passed gate 1: {terminal}")
            if terminal is None:
                raise RuntimeError("fixed v960 trajectory hit control-step limit")

            slots = tuple(range(1, 9))
            for slot in slots:
                brain.restart_slot(slot)

            # Replay both conditions over the exact same physical trajectory. The first gate
            # reward is part of the common history; the terminal reinforcement is withheld
            # for the final control/PAM/PPL contrast.
            terminal_index = int(terminal["step"]) - 1
            first_gate_index = gate_pass_steps[0]
            timecourse: list[dict[str, object]] = []
            no_reward_dan_type_spikes: dict[str, dict[str, int]] = {"off": {}, "on": {}}
            no_reward_dan_total_spikes = {"off": 0, "on": 0}
            no_reward_dan_active_steps = {"off": 0, "on": 0}

            def record_no_reward_dan(label: str, readout: dict[int, bool]) -> None:
                active = [body_id for body_id in all_dan_ids if bool(readout.get(body_id, False))]
                if active:
                    no_reward_dan_active_steps[label] += 1
                no_reward_dan_total_spikes[label] += len(active)
                counts = no_reward_dan_type_spikes[label]
                for body_id in active:
                    type_name = dan_type_by_id.get(body_id, "<untyped>")
                    counts[type_name] = counts.get(type_name, 0) + 1

            sample_offsets = {0, 1, 5, 10, 25, 50, 100, 150}
            for index, row in enumerate(history):
                off = row["retinal"]
                on = [*row["retinal"], *row["haltere"]]
                replay_spikes = brain.step_batch(
                    [
                        {"slot": 1, "stimulate_body": off},
                        {"slot": 2, "stimulate_body": off},
                        {"slot": 3, "stimulate_body": off},
                        {"slot": 4, "stimulate_body": on},
                        {"slot": 5, "stimulate_body": on},
                        {"slot": 6, "stimulate_body": on},
                        {"slot": 7, "stimulate_body": off, "read_body": all_dan_ids},
                        {"slot": 8, "stimulate_body": on, "read_body": all_dan_ids},
                    ],
                    plasticity=True,
                )
                record_no_reward_dan("off", replay_spikes.get(7, {}))
                record_no_reward_dan("on", replay_spikes.get(8, {}))
                if row["event"] == "gate_pass" and index != terminal_index:
                    timecourse.append({
                        "offset_after_gate1": -1,
                        "off": brain.transaction_stats(1),
                        "on": brain.transaction_stats(4),
                        "no_reward_off": brain.transaction_stats(7),
                        "no_reward_on": brain.transaction_stats(8),
                    })
                    for _ in range(a.reinforcement_steps):
                        reinforcement_spikes = brain.step_batch(
                            [
                                {"slot": 1, "stimulate": {"reward_dan": 2.0}},
                                {"slot": 2, "stimulate": {"reward_dan": 2.0}},
                                {"slot": 3, "stimulate": {"reward_dan": 2.0}},
                                {"slot": 4, "stimulate": {"reward_dan": 2.0}},
                                {"slot": 5, "stimulate": {"reward_dan": 2.0}},
                                {"slot": 6, "stimulate": {"reward_dan": 2.0}},
                                {"slot": 7, "read_body": all_dan_ids},
                                {"slot": 8, "read_body": all_dan_ids},
                            ],
                            plasticity=True,
                        )
                        record_no_reward_dan("off", reinforcement_spikes.get(7, {}))
                        record_no_reward_dan("on", reinforcement_spikes.get(8, {}))
                    timecourse.append({
                        "offset_after_gate1": 0,
                        "off": brain.transaction_stats(1),
                        "on": brain.transaction_stats(4),
                        "no_reward_off": brain.transaction_stats(7),
                        "no_reward_on": brain.transaction_stats(8),
                    })
                offset = index - first_gate_index
                if offset in sample_offsets and offset != 0:
                    timecourse.append({
                        "offset_after_gate1": offset,
                        "off": brain.transaction_stats(1),
                        "on": brain.transaction_stats(4),
                        "no_reward_off": brain.transaction_stats(7),
                        "no_reward_on": brain.transaction_stats(8),
                    })
                if index == terminal_index:
                    break

            before = {f"slot{slot}": brain.transaction_stats(slot) for slot in slots}
            for _ in range(a.reinforcement_steps):
                brain.step_batch(
                    [
                        {"slot": 1},
                        {"slot": 2, "stimulate": {"reward_dan": 2.0}},
                        {"slot": 3, "stimulate": {"aversive_dan": 2.0}},
                        {"slot": 4},
                        {"slot": 5, "stimulate": {"reward_dan": 2.0}},
                        {"slot": 6, "stimulate": {"aversive_dan": 2.0}},
                        {"slot": 7},
                        {"slot": 8},
                    ],
                    plasticity=True,
                )
            after = {f"slot{slot}": brain.transaction_stats(slot) for slot in slots}
            contrast_off = brain.transaction_contrast(
                control_slot=1, reward_slot=2, aversive_slot=3, epsilon=1e-7
            )
            contrast_on = brain.transaction_contrast(
                control_slot=4, reward_slot=5, aversive_slot=6, epsilon=1e-7
            )

            active_steps = sum(1 for row in history if int(row["haltere_active_sensilla"]) > 0)
            count_hist: dict[str, int] = {}
            for row in history:
                key = str(int(row["haltere_active_sensilla"]))
                count_hist[key] = count_hist.get(key, 0) + 1
            nonzero_means = [
                float(row["haltere_mean_current"])
                for row in history
                if float(row["haltere_mean_current"]) > 0.0
            ]
            dan_summary = {
                "all_dan_neurons": len(all_dan_ids),
                "reward_dan_neurons": len(reward_dan_ids),
                "aversive_dan_neurons": len(aversive_dan_ids),
                "steps_with_any_dan_spike": sum(int(row.get("active_dan_count", 0)) > 0 for row in history),
                "total_dan_spikes": sum(int(row.get("active_dan_count", 0)) for row in history),
                "reward_dan_spikes": sum(int(row.get("active_reward_dan_count", 0)) for row in history),
                "aversive_dan_spikes": sum(int(row.get("active_aversive_dan_count", 0)) for row in history),
                "other_dan_spikes": sum(int(row.get("active_other_dan_count", 0)) for row in history),
                "pre_gate1": {
                    "steps": first_gate_index + 1,
                    "total": sum(int(row.get("active_dan_count", 0)) for row in history[: first_gate_index + 1]),
                    "reward": sum(int(row.get("active_reward_dan_count", 0)) for row in history[: first_gate_index + 1]),
                    "aversive": sum(int(row.get("active_aversive_dan_count", 0)) for row in history[: first_gate_index + 1]),
                    "other": sum(int(row.get("active_other_dan_count", 0)) for row in history[: first_gate_index + 1]),
                },
                "post_gate1": {
                    "steps": max(0, len(history) - first_gate_index - 1),
                    "total": sum(int(row.get("active_dan_count", 0)) for row in history[first_gate_index + 1 :]),
                    "reward": sum(int(row.get("active_reward_dan_count", 0)) for row in history[first_gate_index + 1 :]),
                    "aversive": sum(int(row.get("active_aversive_dan_count", 0)) for row in history[first_gate_index + 1 :]),
                    "other": sum(int(row.get("active_other_dan_count", 0)) for row in history[first_gate_index + 1 :]),
                },
            }
            payload = {
                "schema_version": 2,
                "production": str(a.production),
                "checkpoint_step": int(loaded.get("step", -1)),
                "global_weight_version": gv,
                "gate2_center_z_mm": a.gate2_center_z_mm,
                "history_steps": len(history),
                "gate_pass_steps": gate_pass_steps,
                "terminal": terminal,
                "haltere": {
                    "gain": a.haltere_gain,
                    "active_steps": active_steps,
                    "active_fraction": active_steps / len(history),
                    "active_sensilla_histogram": count_hist,
                    "peak_current": max(float(row["haltere_max_current"]) for row in history),
                    "mean_nonzero_current": sum(nonzero_means) / max(len(nonzero_means), 1),
                },
                "dan_activity": dan_summary,
                "no_reward_dan_activity": {
                    label: {
                        "active_steps": no_reward_dan_active_steps[label],
                        "total_spikes": no_reward_dan_total_spikes[label],
                        "type_spikes": dict(
                            sorted(
                                no_reward_dan_type_spikes[label].items(),
                                key=lambda item: (-item[1], item[0]),
                            )
                        ),
                    }
                    for label in ("off", "on")
                },
                "plasticity_timecourse": timecourse,
                "no_reward_terminal": {"off": brain.transaction_stats(7), "on": brain.transaction_stats(8)},
                "before_terminal_reinforcement": before,
                "after_terminal_reinforcement": after,
                "contrast_off": contrast_off,
                "contrast_on": contrast_on,
            }
            tmp = out.with_name("." + out.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, out)
            print(
                "haltere_plasticity_probe=PASS",
                f"steps={len(history)}",
                f"terminal={terminal['reason']}",
                f"haltere_active_fraction={payload['haltere']['active_fraction']:.3f}",
            )
            for name, contrast in (("off", contrast_off), ("on", contrast_on)):
                print(name, "reward", contrast["reward"], "aversive", contrast["aversive"])
            print("output", out)
    finally:
        for item in workers:
            try:
                item.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
