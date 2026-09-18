#!/usr/bin/env python3
"""Prepare generated inputs required by Flyppy shared-weight population training.

This script may create missing generated MaleCNS boundary artifacts, but it never
modifies the Flyppy experiment checkpoint, curriculum, trajectory, or summary.
A compact report is always written under reports/flyppy/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/flyppy/population_input_prepare.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/malecns-v1.0")
    p.add_argument("--groups", type=Path)
    p.add_argument("--retinotopic-map", type=Path)
    p.add_argument("--haltere-sensory-map", type=Path)
    p.add_argument("--wing-motor-map", type=Path)
    p.add_argument("--body-motor-map", type=Path)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def abspath(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def run_stage(name: str, command: list[str], stages: list[dict[str, object]]) -> None:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    stages.append({
        "stage": name,
        "ok": proc.returncode == 0,
        "command": command,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    })
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def main() -> int:
    args = parse_args()
    snapshot = abspath(args.snapshot)
    groups = abspath(args.groups or snapshot / "embodiment-groups-v0.json")
    retina = abspath(args.retinotopic_map or snapshot / "retinotopic-vision-v1.json")
    haltere = abspath(
        args.haltere_sensory_map or snapshot / "haltere-campaniform-sensory-v1.json"
    )
    wing = abspath(args.wing_motor_map or snapshot / "wing-motor-neurons-v0.json")
    body = abspath(args.body_motor_map or snapshot / "body-motor-neurons-v0.json")
    report = abspath(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, object]] = []
    failure = ""
    overall = True
    python = sys.executable

    try:
        required_snapshot = [
            snapshot / "manifest.json",
            snapshot / "annotations.feather",
            snapshot / "body_ids.u64le",
            snapshot / "neurotransmitters.u8",
        ]
        missing_snapshot = [str(p) for p in required_snapshot if not p.exists()]
        if missing_snapshot:
            raise FileNotFoundError("missing snapshot files: " + ", ".join(missing_snapshot))
        stages.append({"stage": "snapshot", "ok": True, "detail": "required snapshot files present"})

        if not groups.exists():
            run_stage(
                "generate_groups",
                [python, "scripts/data/make_embodiment_groups.py", "--snapshot", str(snapshot), "--output", str(groups)],
                stages,
            )
        else:
            stages.append({"stage": "generate_groups", "ok": True, "detail": "already present"})

        if not wing.exists():
            run_stage(
                "generate_wing_motor_map",
                [python, "scripts/data/prepare_wing_motor_map.py", "--snapshot", str(snapshot), "--output", str(wing)],
                stages,
            )
        else:
            stages.append({"stage": "generate_wing_motor_map", "ok": True, "detail": "already present"})

        if not body.exists():
            run_stage(
                "generate_body_motor_map",
                [python, "scripts/data/prepare_body_motor_map.py", "--snapshot", str(snapshot), "--output", str(body)],
                stages,
            )
        else:
            stages.append({"stage": "generate_body_motor_map", "ok": True, "detail": "already present"})

        if not retina.exists():
            run_stage(
                "generate_retinotopic_map",
                [python, "scripts/data/prepare_retinotopic_vision.py", "--snapshot", str(snapshot), "--output", str(retina), "--download"],
                stages,
            )
        else:
            stages.append({"stage": "generate_retinotopic_map", "ok": True, "detail": "already present"})

        if not haltere.exists():
            run_stage(
                "generate_haltere_sensory_map",
                [
                    python,
                    "scripts/data/prepare_haltere_sensory_map.py",
                    "--annotations",
                    str(snapshot / "annotations.feather"),
                    "--output",
                    str(haltere),
                ],
                stages,
            )
        else:
            stages.append({
                "stage": "generate_haltere_sensory_map",
                "ok": True,
                "detail": "already present",
            })

        for path in (groups, wing, body, retina, haltere):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"generated input missing or empty: {path}")

        group_payload = json.loads(groups.read_text(encoding="utf-8"))
        group_names = set(group_payload.get("groups", {}))
        required_groups = {"reward_dan", "aversive_dan"}
        missing_groups = sorted(required_groups - group_names)
        if missing_groups:
            raise RuntimeError("required reinforcement groups missing: " + ", ".join(missing_groups))
        stages.append({
            "stage": "validate_generated_inputs",
            "ok": True,
            "detail": f"groups={len(group_names)} reward_dan+aversive_dan present",
        })
    except Exception:
        overall = False
        failure = traceback.format_exc()

    lines = [
        "# Flyppy population input preparation",
        "",
        f"- generated_at_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- overall: {'PASS' if overall else 'FAIL'}",
        "- mutates Flyppy training state: false",
        "",
        "## Inputs",
        "",
        f"- snapshot: `{snapshot}`",
        f"- groups: `{groups}`",
        f"- wing motor map: `{wing}`",
        f"- body motor map: `{body}`",
        f"- retinotopic map: `{retina}`",
        f"- haltere sensory map: `{haltere}`",
        "",
        "## Stages",
        "",
    ]
    for stage in stages:
        lines.append(f"### {stage.get('stage')} — {'PASS' if stage.get('ok') else 'FAIL'}")
        lines.append("")
        if "detail" in stage:
            lines.append(str(stage["detail"]))
            lines.append("")
        if "returncode" in stage:
            lines.append(f"- returncode: {stage['returncode']}")
            lines.append("")
            if stage.get("stdout"):
                lines += ["stdout:", "```text", str(stage["stdout"]), "```", ""]
            if stage.get("stderr"):
                lines += ["stderr:", "```text", str(stage["stderr"]), "```", ""]
    if failure:
        lines += ["## Failure", "", "```text", failure.rstrip(), "```", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={report}")
    print(f"flyppy_population_input_prepare={'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
