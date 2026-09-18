from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
import subprocess
from typing import Any

MALECNS_DATASET = "male-cns:v1.0"
MALECNS_RUNTIME_SEMANTICS = "male-cns-v1-class-dan-v1"
MODULATOR_ROLE_DEFINITION = "consensus_nt=dopamine AND released annotation class=DAN"
NEURAL_RUNTIME_SEMANTICS = "signed-activity-local-dynamics-v1"
PLASTICITY_SEMANTICS = "local-three-factor-dopamine-eligibility-v1"
POPULATION_CHECKPOINT_SEMANTICS = "global-weights-only-v1"
POPULATION_NEURAL_STEP_SEMANTICS = "aggregate-slot-neural-step-count-v1"
HALTERE_FULL_KIND = "male-cns-haltere-campaniform-afferents"
HALTERE_TIMING_KIND = "male-cns-haltere-timing-afferents-inferred-v1"
HALTERE_EXPECTED_COUNTS = {HALTERE_FULL_KIND: 195, HALTERE_TIMING_KIND: 97}
NEUTRAL_TRIM_KIND = "flybody-task-independent-neutral-wing-trim"
NEUTRAL_TRIM_SEMANTICS = "flybody-neutral-trim-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    )
    return proc.stdout.strip()


def git_provenance(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    sha = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "git_sha": sha,
        "dirty": bool(status),
        "dirty_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def dependency_lock_provenance(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    files: dict[str, str] = {}
    for name in ("uv.lock", "Cargo.lock"):
        path = root / name
        if path.exists():
            files[name] = sha256_file(path)
    return {"files": files, "composite_sha256": sha256_json(files)}


def artifact_record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    path = Path(path).resolve()
    display = str(path)
    if root is not None:
        try:
            display = str(path.relative_to(Path(root).resolve()))
        except ValueError:
            pass
    record: dict[str, object] = {
        "path": display,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("schema_version", "kind", "dataset", "runtime_semantics"):
                if key in payload:
                    record[key] = payload[key]
    return record


def validate_production_snapshot(snapshot: Path) -> dict[str, object]:
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != MALECNS_DATASET:
        raise RuntimeError(
            f"production Flyppy requires dataset {MALECNS_DATASET!r}; got {payload.get('dataset')!r}"
        )
    if payload.get("runtime_semantics") != MALECNS_RUNTIME_SEMANTICS:
        raise RuntimeError(
            "MaleCNS snapshot does not declare the current production runtime semantics; "
            f"expected {MALECNS_RUNTIME_SEMANTICS!r}. Rebuild the snapshot with current prepare_malecns.py."
        )
    if payload.get("modulator_role_definition") != MODULATOR_ROLE_DEFINITION:
        raise RuntimeError(
            "MaleCNS snapshot modulator role definition is incompatible with production class-DAN semantics"
        )
    role_file = payload.get("modulator_roles_file")
    if not isinstance(role_file, str) or not role_file:
        raise RuntimeError("MaleCNS production snapshot must declare modulator_roles_file")
    role_path = snapshot / role_file
    if not role_path.exists():
        raise RuntimeError(f"MaleCNS production snapshot is missing {role_file}")
    generator_provenance = payload.get("generator_provenance")
    if not isinstance(generator_provenance, dict) or not generator_provenance.get("generator_git"):
        raise RuntimeError(
            "MaleCNS production snapshot lacks generator Git provenance; rebuild with current prepare_malecns.py"
        )
    source_hashes = payload.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("MaleCNS production snapshot must record source_sha256")
    generated_hashes = payload.get("generated_sha256")
    if not isinstance(generated_hashes, dict) or not generated_hashes:
        raise RuntimeError(
            "MaleCNS production snapshot must record generated_sha256; rebuild with current prepare_malecns.py"
        )
    for name, expected in generated_hashes.items():
        path = snapshot / str(name)
        if not path.exists():
            raise RuntimeError(f"MaleCNS production snapshot is missing generated file {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"MaleCNS generated file hash mismatch for {name}")
    return {
        "dataset": payload["dataset"],
        "format_version": payload.get("format_version"),
        "runtime_semantics": payload["runtime_semantics"],
        "modulator_role_definition": payload["modulator_role_definition"],
        "generator_provenance": generator_provenance,
        "manifest_sha256": sha256_file(manifest_path),
        "snapshot_sha256": sha256_json(
            {
                "dataset": payload["dataset"],
                "runtime_semantics": payload["runtime_semantics"],
                "source_sha256": source_hashes,
                "generated_sha256": generated_hashes,
            }
        ),
    }


def validate_haltere_map(
    path: Path, expected_kind: str, *, snapshot: Path | None = None
) -> dict[str, object]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError("haltere sensory map schema_version=1 is required")
    actual_kind = str(payload.get("kind", ""))
    if actual_kind != expected_kind:
        raise RuntimeError(
            f"haltere sensory map kind mismatch: expected {expected_kind!r}, got {actual_kind!r}"
        )
    expected_count = HALTERE_EXPECTED_COUNTS.get(expected_kind)
    count = int(payload.get("count", -1))
    if expected_count is not None and count != expected_count:
        raise RuntimeError(
            f"haltere sensory map count mismatch for {expected_kind}: expected {expected_count}, got {count}"
        )
    by_side = payload.get("body_ids_by_side") or {}
    if not by_side.get("left") or not by_side.get("right"):
        raise RuntimeError("haltere sensory map must contain both sides")
    provenance = payload.get("provenance")
    allow_legacy = os.environ.get("VF_ALLOW_LEGACY_HALTERE_ARTIFACT") == "1"
    if not allow_legacy and (
        not isinstance(provenance, dict)
        or not provenance.get("generator_git")
        or not (provenance.get("annotations_sha256") or provenance.get("snapshot_manifest_sha256"))
    ):
        raise RuntimeError(
            "haltere sensory map lacks current generator/source provenance; regenerate it with the matching current generator. Set VF_ALLOW_LEGACY_HALTERE_ARTIFACT=1 only for explicit historical replay."
        )
    if snapshot is not None and not allow_legacy:
        snapshot = Path(snapshot).resolve()
        manifest_path = snapshot / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_kind == HALTERE_FULL_KIND:
            generated = manifest.get("generated_sha256")
            expected_annotations = (
                generated.get("annotations.feather")
                if isinstance(generated, dict)
                else None
            )
            if not expected_annotations or provenance.get("annotations_sha256") != expected_annotations:
                raise RuntimeError("haltere full map was not generated from this snapshot annotations artifact")
        elif expected_kind == HALTERE_TIMING_KIND:
            if provenance.get("snapshot_manifest_sha256") != sha256_file(manifest_path):
                raise RuntimeError("haltere timing map was not generated from this snapshot manifest")
    return {
        "kind": actual_kind,
        "count": count,
        "sha256": sha256_file(path),
        "path": str(path),
    }



def validate_derived_artifact(path: Path, snapshot: Path) -> dict[str, object]:
    path = Path(path).resolve()
    snapshot = Path(snapshot).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"derived artifact must contain a JSON object: {path}")
    provenance = payload.get("artifact_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(
            f"derived artifact lacks generator provenance: {path}. Regenerate it with the current generator."
        )
    if not provenance.get("generator_git"):
        raise RuntimeError(f"derived artifact lacks generator Git provenance: {path}")
    manifest_hash = sha256_file(snapshot / "manifest.json")
    if provenance.get("snapshot_manifest_sha256") != manifest_hash:
        raise RuntimeError(
            f"derived artifact was not generated from the current snapshot manifest: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "generator": provenance.get("generator"),
        "generator_git": provenance.get("generator_git"),
        "snapshot_manifest_sha256": manifest_hash,
    }


def validate_neutral_trim(pattern: Path, metadata: Path | None = None) -> dict[str, object]:
    pattern = Path(pattern).resolve()
    metadata = (
        Path(metadata).resolve()
        if metadata is not None
        else pattern.with_suffix(".json")
    )
    if not pattern.exists() or not metadata.exists():
        raise RuntimeError(
            "FlyBody v7/v8 requires the neutral-trim pattern and metadata; run scripts/data/prepare_flybody_neutral_trim.py"
        )
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("kind") != NEUTRAL_TRIM_KIND:
        raise RuntimeError("neutral-trim metadata kind is incompatible")
    allow_legacy = os.environ.get("VF_ALLOW_LEGACY_FLYBODY_ARTIFACTS") == "1"
    if payload.get("runtime_semantics") != NEUTRAL_TRIM_SEMANTICS:
        if allow_legacy:
            return {
                "kind": payload.get("kind"),
                "runtime_semantics": "legacy/unspecified",
                "pattern_sha256": sha256_file(pattern),
                "metadata_sha256": sha256_file(metadata),
            }
        raise RuntimeError(
            "neutral-trim metadata predates current provenance semantics; regenerate it with current prepare_flybody_neutral_trim.py. Set VF_ALLOW_LEGACY_FLYBODY_ARTIFACTS=1 only for explicit historical replay."
        )
    actual = sha256_file(pattern)
    if payload.get("output_sha256") != actual:
        raise RuntimeError("neutral-trim pattern hash does not match metadata")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("generator_git"):
        raise RuntimeError("neutral-trim metadata is missing generator provenance")
    return {
        "kind": payload["kind"],
        "runtime_semantics": payload["runtime_semantics"],
        "pattern_sha256": actual,
        "metadata_sha256": sha256_file(metadata),
    }


def compatibility_warnings(
    *,
    body_version: str,
    environment_version: str,
    vision_mode: str,
    haltere_gain: float,
    haltere_kind: str,
) -> list[str]:
    warnings: list[str] = []
    if body_version in {"v3", "v4", "v5", "v6", "v7"} and body_version != environment_version:
        warnings.append(
            f"body={body_version} and environment={environment_version} are different version axes; verify this mixed-generation experiment is intentional"
        )
    if body_version == "v8" and environment_version != "v7":
        warnings.append(
            f"body=v8 has no matching environment v8; environment={environment_version} is a deliberate mixed-generation configuration"
        )
    if vision_mode not in {"direct-ray", "raster"}:
        warnings.append(f"unrecognized vision mode in provenance: {vision_mode}")
    if haltere_gain > 0.0 and haltere_kind not in {HALTERE_FULL_KIND, HALTERE_TIMING_KIND}:
        warnings.append("enabled haltere feedback uses an unknown sensory-map kind")
    return warnings


def build_run_provenance(
    *,
    root: Path,
    snapshot: Path,
    artifacts: dict[str, Path],
    conditions: dict[str, object],
) -> dict[str, object]:
    root = Path(root).resolve()
    snapshot_info = validate_production_snapshot(snapshot)
    records: dict[str, object] = {}
    for name, path in artifacts.items():
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = root / resolved
        if resolved.exists():
            records[name] = artifact_record(resolved, root=root)
    return {
        "schema_version": 1,
        "code": git_provenance(root),
        "dependency_lock": dependency_lock_provenance(root),
        "snapshot": snapshot_info,
        "semantics": run_semantics(),
        "artifacts": records,
        "conditions": conditions,
    }

def run_semantics() -> dict[str, str]:
    return {
        "snapshot": MALECNS_RUNTIME_SEMANTICS,
        "neural_runtime": NEURAL_RUNTIME_SEMANTICS,
        "plasticity": PLASTICITY_SEMANTICS,
        "population_checkpoint": POPULATION_CHECKPOINT_SEMANTICS,
        "checkpoint_neural_step": POPULATION_NEURAL_STEP_SEMANTICS,
    }
