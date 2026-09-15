#!/usr/bin/env python3
"""Find the maximum safe shared-weight Flyppy population on this machine.

The parent process launches every candidate in a fresh child process so GPU and
MuJoCo resources are fully released between attempts.  Each child constructs the
requested number of FlyBody slots, starts the real population neural bridge,
loads the production checkpoint, and executes two batched neural steps with
plasticity enabled.  It never saves a checkpoint and never mutates training
state.

The search is bounded by the GPU runtime's 32-slot active-mask limit and by a
conservative estimate of currently reclaimable unified memory.  Actual child
execution is the final authority: a candidate is accepted only if the real
initialization and batched compute path succeeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EMBODIMENT = ROOT / "scripts" / "embodiment"
if str(EMBODIMENT) not in sys.path:
    sys.path.insert(0, str(EMBODIMENT))

DEFAULT_EXPERIMENT = Path("artifacts/experiments/flyppy-v3")
DEFAULT_SNAPSHOT = Path("artifacts/malecns-v1.0")
DEFAULT_REPORT = Path("reports/flyppy/population_capacity.md")
DEFAULT_JSON = Path("artifacts/profiles/flyppy-population-capacity/latest.json")
HARD_MAX = 32
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--hard-max", type=int, default=HARD_MAX)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--child-population", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object in {path}")
    return payload


def physical_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def reclaimable_memory_bytes() -> int | None:
    """Approximate currently reclaimable macOS memory from vm_stat."""
    try:
        result = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True
        )
    except Exception:
        return None
    page_match = re.search(r"page size of (\d+) bytes", result.stdout)
    if not page_match:
        return None
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        digits = re.sub(r"[^0-9]", "", raw)
        if digits:
            pages[name.strip()] = int(digits)
    reclaimable_names = (
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    )
    return page_size * sum(pages.get(name, 0) for name in reclaimable_names)


def memory_model(snapshot_manifest: dict[str, Any]) -> dict[str, int]:
    n = int(snapshot_manifest["neuron_count"])
    m = int(snapshot_manifest["edge_count"])
    # gpu_population.rs exact dominant allocations:
    # fixed: topology(row offsets + pre/post/count + neurotransmitters) + global weights
    fixed = 4 * ((n + 1) + 3 * m + n) + 4 * m
    # per slot: neuron state(16n), synapse state(8m), transaction(12m),
    # two spike buffers(8n), external current(4n).
    per_slot = 28 * n + 20 * m
    return {"fixed_gpu_bytes": fixed, "per_slot_gpu_bytes": per_slot}


def estimated_upper_bound(
    *, hard_max: int, fixed_bytes: int, per_slot_bytes: int
) -> tuple[int, dict[str, Any]]:
    physical = physical_memory_bytes()
    reclaimable = reclaimable_memory_bytes()
    limits = [hard_max]
    details: dict[str, Any] = {
        "physical_memory_bytes": physical,
        "reclaimable_memory_bytes": reclaimable,
    }
    if physical:
        reserve = max(3 * GIB, int(physical * 0.20))
        physical_budget = max(0, physical - reserve - fixed_bytes)
        physical_slots = max(1, physical_budget // per_slot_bytes)
        limits.append(int(physical_slots))
        details["physical_reserve_bytes"] = reserve
        details["physical_memory_slot_limit"] = int(physical_slots)
    if reclaimable:
        # Do not deliberately consume the final 15% of currently reclaimable memory.
        reclaimable_budget = max(0, int(reclaimable * 0.85) - fixed_bytes)
        reclaimable_slots = max(1, reclaimable_budget // per_slot_bytes)
        limits.append(int(reclaimable_slots))
        details["reclaimable_memory_slot_limit"] = int(reclaimable_slots)
    upper = max(1, min(limits))
    details["estimated_safe_upper_bound"] = upper
    return upper, details


def child_run(population: int, experiment: Path, snapshot: Path) -> int:
    from population_neural_bridge_client import PopulationNeuralBridgeClient
    from train_flyppy_population import make_slot

    groups = snapshot / "embodiment-groups-v0.json"
    checkpoint = experiment / "checkpoint"
    population_state_path = experiment / "population-state.json"

    class SlotArgs:
        gate_count = 6
        seed = 0
        wing_motor_map = snapshot / "wing-motor-neurons-v0.json"
        body_motor_map = snapshot / "body-motor-neurons-v0.json"
        retinotopic_map = snapshot / "retinotopic-vision-v1.json"
        photoreceptor_current_gain = 2.0

    started = time.perf_counter()
    slots = [make_slot(SlotArgs(), index) for index in range(population)]
    if len(slots) != population:
        raise RuntimeError("slot construction count mismatch")

    population_state = (
        read_json(population_state_path) if population_state_path.exists() else {}
    )
    version = int(population_state.get("global_weight_version", 0))
    groups_payload = read_json(groups)
    reward_ids = groups_payload.get("groups", {}).get("reward_dan", {}).get("body_ids", [])
    if not reward_ids:
        raise RuntimeError("reward_dan group contains no body IDs")
    read_id = int(reward_ids[0])

    with PopulationNeuralBridgeClient(
        snapshot=snapshot,
        groups=groups,
        slots=population,
    ) as brain:
        brain.ping()
        brain.load_checkpoint(checkpoint, global_weight_version=version)
        requests = [
            {"slot": slot, "read_body": [read_id]}
            for slot in range(population)
        ]
        # Two real batched steps exercise both spike-buffer directions and the
        # production plasticity kernel without persisting any state.
        brain.step_batch(requests, plasticity=True)
        brain.step_batch(requests, plasticity=True)
        backend = str(brain.ready.get("backend", "unknown"))

    print(
        json.dumps(
            {
                "ok": True,
                "population": population,
                "backend": backend,
                "elapsed_seconds": time.perf_counter() - started,
            },
            separators=(",", ":"),
        )
    )
    return 0


def run_candidate(
    population: int,
    *,
    args: argparse.Namespace,
    experiment: Path,
    snapshot: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--experiment",
        str(experiment),
        "--snapshot",
        str(snapshot),
        "--child-population",
        str(population),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        ok = proc.returncode == 0
        return {
            "population": population,
            "ok": ok,
            "returncode": proc.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "population": population,
            "ok": False,
            "returncode": None,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": "timeout",
        }


def search_capacity(
    upper: int,
    *,
    args: argparse.Namespace,
    experiment: Path,
    snapshot: Path,
) -> tuple[int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    cache: dict[int, bool] = {}

    def test(n: int) -> bool:
        if n in cache:
            return cache[n]
        result = run_candidate(n, args=args, experiment=experiment, snapshot=snapshot)
        attempts.append(result)
        cache[n] = bool(result["ok"])
        return cache[n]

    # Exponential growth finds the useful region quickly without immediately
    # jumping to the largest high-memory candidate.
    last_success = 0
    n = 1
    first_failure: int | None = None
    while n <= upper:
        if test(n):
            last_success = n
            if n == upper:
                return n, attempts
            n = min(upper, n * 2)
            if n == last_success:
                break
        else:
            first_failure = n
            break

    if first_failure is None:
        if last_success < upper:
            if test(upper):
                return upper, attempts
            first_failure = upper
        else:
            return last_success, attempts

    low = last_success + 1
    high = first_failure - 1
    while low <= high:
        mid = (low + high) // 2
        if test(mid):
            last_success = mid
            low = mid + 1
        else:
            high = mid - 1
    return last_success, attempts


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Flyppy population capacity",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        "- mutates training artifacts: false",
        f"- hard runtime limit: {payload['hard_max']}",
        f"- estimated safe upper bound: {payload['estimated_safe_upper_bound']}",
        f"- maximum verified population: **{payload['max_supported_population']}**",
        f"- estimated GPU bytes / slot: {payload['per_slot_gpu_bytes']}",
        f"- estimated fixed GPU bytes: {payload['fixed_gpu_bytes']}",
        "",
        "## Candidate results",
        "",
        "| population | result | seconds | returncode |",
        "|---:|---|---:|---:|",
    ]
    for item in payload["attempts"]:
        lines.append(
            f"| {item['population']} | {'PASS' if item['ok'] else 'FAIL'} | "
            f"{item['elapsed_seconds']:.3f} | {item['returncode'] if item['returncode'] is not None else '-'} |"
        )
    lines.extend(["", "## Memory context", "", "```json"])
    memory_keys = [
        "physical_memory_bytes",
        "reclaimable_memory_bytes",
        "physical_reserve_bytes",
        "physical_memory_slot_limit",
        "reclaimable_memory_slot_limit",
    ]
    lines.append(json.dumps({k: payload.get(k) for k in memory_keys}, indent=2))
    lines.extend(["```", ""])
    failed = [item for item in payload["attempts"] if not item["ok"]]
    if failed:
        lines.extend(["## Failure tails", ""])
        for item in failed:
            lines.extend(
                [
                    f"### population={item['population']}",
                    "",
                    "```text",
                    (item.get("stderr_tail") or item.get("stdout_tail") or "no output").rstrip(),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    experiment = absolute(args.experiment)
    snapshot = absolute(args.snapshot)

    if args.child_population is not None:
        if not 1 <= args.child_population <= HARD_MAX:
            raise SystemExit(f"child population must be 1..{HARD_MAX}")
        return child_run(args.child_population, experiment, snapshot)

    hard_max = max(1, min(int(args.hard_max), HARD_MAX))
    manifest = read_json(experiment / "checkpoint" / "manifest.json")
    model = memory_model(manifest)
    upper, memory_details = estimated_upper_bound(
        hard_max=hard_max,
        fixed_bytes=model["fixed_gpu_bytes"],
        per_slot_bytes=model["per_slot_gpu_bytes"],
    )
    maximum, attempts = search_capacity(
        upper, args=args, experiment=experiment, snapshot=snapshot
    )
    if maximum < 1:
        raise SystemExit("population capacity probe could not run even one slot")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hard_max": hard_max,
        "max_supported_population": maximum,
        **model,
        **memory_details,
        "attempts": attempts,
    }
    report = absolute(args.report)
    json_out = absolute(args.json_out)
    report.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(payload), encoding="utf-8")
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"population_capacity_report={report}")
    print(f"population_capacity_json={json_out}")
    print(f"max_supported_population={maximum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
