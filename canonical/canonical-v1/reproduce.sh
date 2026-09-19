#!/usr/bin/env bash
# End-to-end reproducer for virtual-fly canonical experiment v1.
#
# This script does NOT run scientific code from the caller's current HEAD. It
# creates an isolated detached worktree at the canonical commit and runs every
# generator/trainer/evaluator there. Output always goes to a new directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONFIG="$SCRIPT_DIR/config.json"
REFERENCE_MANIFEST="$SCRIPT_DIR/reference-manifest.json"
ORIGINAL_EXPERIMENT_SHA="7fa464aad7269d34f46f1171080b51e095d1d811"
PUBLIC_EQUIVALENT_SHA="f9c86c904d67ff974f3c43d37aab3619bc93fc1b"
EXPECTED_SHA="$PUBLIC_EQUIVALENT_SHA"
EXPECTED_CONFIG_SHA256="91f5e02cad1a3603b31f23f6fb95e5c6792fe3079c531e247d529b455bdc3ea2"
EXPECTED_REFERENCE_MANIFEST_SHA256="89309f3c7c0968b750b36c1cd8eb75a550a7b5add0fefd45c292f36387cb9f59"
STOP_AFTER_STATIC="${VF_CANONICAL_STOP_AFTER_STATIC:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_OUTPUT_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/reproductions"
OUTPUT_ROOT="${VF_CANONICAL_OUTPUT_ROOT:-$DEFAULT_OUTPUT_BASE/canonical-v1-$STAMP}"
WORKTREE="${VF_CANONICAL_WORKTREE:-/tmp/virtual-fly-canonical-v1-reproduce-$STAMP-$$}"
KEEP_WORKTREE="${VF_CANONICAL_KEEP_WORKTREE:-0}"

if [ ! -f "$CONFIG" ] || [ ! -f "$REFERENCE_MANIFEST" ]; then
  echo "canonical config/reference manifest missing under $SCRIPT_DIR" >&2
  exit 2
fi
sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "neither shasum nor sha256sum is available" >&2
    return 127
  fi
}
ACTUAL_CONFIG_SHA256="$(sha256_file "$CONFIG")"
ACTUAL_REFERENCE_MANIFEST_SHA256="$(sha256_file "$REFERENCE_MANIFEST")"
if [ "$ACTUAL_CONFIG_SHA256" != "$EXPECTED_CONFIG_SHA256" ]; then
  echo "canonical config hash mismatch: expected $EXPECTED_CONFIG_SHA256, got $ACTUAL_CONFIG_SHA256" >&2
  exit 2
fi
if [ "$ACTUAL_REFERENCE_MANIFEST_SHA256" != "$EXPECTED_REFERENCE_MANIFEST_SHA256" ]; then
  echo "canonical reference manifest hash mismatch: expected $EXPECTED_REFERENCE_MANIFEST_SHA256, got $ACTUAL_REFERENCE_MANIFEST_SHA256" >&2
  exit 2
fi
if ! git -C "$REPO_ROOT" cat-file -e "$EXPECTED_SHA^{commit}" 2>/dev/null; then
  echo "canonical commit $EXPECTED_SHA is not available in this repository" >&2
  exit 2
fi
if [ -e "$OUTPUT_ROOT" ]; then
  echo "refusing to overwrite existing reproduction output: $OUTPUT_ROOT" >&2
  exit 2
fi
if [ -e "$WORKTREE" ]; then
  echo "refusing to reuse existing worktree path: $WORKTREE" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/report" "$OUTPUT_ROOT/inputs" "$OUTPUT_ROOT/source"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

cleanup() {
  if [ "$KEEP_WORKTREE" = "1" ]; then
    echo "canonical_worktree_preserved=$WORKTREE"
  else
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "canonical_original_experiment_sha=$ORIGINAL_EXPERIMENT_SHA"
echo "canonical_public_equivalent_sha=$PUBLIC_EQUIVALENT_SHA"
echo "canonical_reproduction_sha=$EXPECTED_SHA"
echo "canonical_reproduction_output=$OUTPUT_ROOT"
echo "canonical_worktree=$WORKTREE"

git -C "$REPO_ROOT" worktree add --detach "$WORKTREE" "$EXPECTED_SHA"
cd "$WORKTREE"
ACTUAL_SHA="$(git rev-parse HEAD)"
[ "$ACTUAL_SHA" = "$EXPECTED_SHA" ] || { echo "unexpected worktree SHA: $ACTUAL_SHA" >&2; exit 3; }
[ -z "$(git status --porcelain=v1 --untracked-files=all)" ] || { echo "canonical worktree is not clean" >&2; git status --short >&2; exit 3; }

# Fresh dependency environment from the pinned lockfile.
uv sync --frozen 2>&1 | tee "$OUTPUT_ROOT/logs/uv-sync.log"
PY="$WORKTREE/.venv/bin/python"
"$PY" - <<'PY' > "$OUTPUT_ROOT/runtime-versions.json"
import importlib.metadata as md, json, platform, subprocess, sys
pkgs = ["virtual-fly-tools", "flygym", "mujoco", "numpy", "scipy", "pyarrow", "pandas"]
def cmd(*args):
    return subprocess.check_output(args, text=True).strip()
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "rustc": cmd("rustc", "--version"),
    "cargo": cmd("cargo", "--version"),
    "uv": cmd("uv", "--version"),
    "packages": {name: md.version(name) for name in pkgs},
}, indent=2))
PY

RAW="$OUTPUT_ROOT/raw/male-cns-v1.0"
SNAP="$OUTPUT_ROOT/snapshot/male-cns-v1.0"
DERIVED="$OUTPUT_ROOT/derived"
TRAIN="$OUTPUT_ROOT/training"
INITIAL="$OUTPUT_ROOT/initial"
EVAL="$OUTPUT_ROOT/evaluation"
REPORT="$OUTPUT_ROOT/report"
INPUTS="$OUTPUT_ROOT/inputs"
mkdir -p "$RAW" "$SNAP" "$DERIVED" "$INITIAL" "$EVAL/initial" "$EVAL/final" "$REPORT" "$INPUTS"

# Give generators the same canonical repository-relative names used by the
# recorded reference while storing bytes outside the disposable worktree. Some
# metadata intentionally records these logical paths, so stable relative names
# are part of byte-level static reproducibility.
mkdir -p artifacts artifacts/derived
ln -s "$SNAP" artifacts/canonical-malecns-v1.0
ln -s "$DERIVED" artifacts/embodiment
CANONICAL_SNAPSHOT="artifacts/canonical-malecns-v1.0"
CANONICAL_DERIVED="artifacts/derived"
CANONICAL_EMBODIMENT="artifacts/embodiment"

# 1) Fresh official source acquisition and MaleCNS snapshot.
"$PY" scripts/data/prepare_malecns.py \
  --download \
  --raw-dir "$RAW" \
  --output "$SNAP" \
  2>&1 | tee "$OUTPUT_ROOT/logs/prepare-malecns.log"

# FlyBody official measured wingbeat source. This downloader has an environment
# output override; keep a persistent copy under OUTPUT_ROOT before worktree cleanup.
export VF_FLYBODY_WING_PATTERN="$WORKTREE/artifacts/flybody-data/wing_pattern_fmech.npy"
"$PY" scripts/dev/prefetch_flybody_flight_data.py \
  2>&1 | tee "$OUTPUT_ROOT/logs/prefetch-flybody.log"
cp "$VF_FLYBODY_WING_PATTERN" "$OUTPUT_ROOT/source/wing_pattern_fmech.npy"

# 2) Fresh snapshot-derived boundaries/maps.
"$PY" scripts/dev/prepare_flyppy_population_inputs.py \
  --snapshot "$CANONICAL_SNAPSHOT" \
  --groups "$CANONICAL_SNAPSHOT/embodiment-groups-v0.json" \
  --retinotopic-map "$CANONICAL_SNAPSHOT/retinotopic-vision-v1.json" \
  --haltere-sensory-map "$CANONICAL_SNAPSHOT/haltere-campaniform-sensory-v1.json" \
  --haltere-sensory-kind male-cns-haltere-campaniform-afferents \
  --wing-motor-map "$CANONICAL_SNAPSHOT/wing-motor-neurons-v0.json" \
  --body-motor-map "$CANONICAL_SNAPSHOT/body-motor-neurons-v0.json" \
  --report "$REPORT/population-input-prepare.md" \
  2>&1 | tee "$OUTPUT_ROOT/logs/prepare-population-inputs.log"

"$PY" scripts/data/prepare_haltere_timing_sensory_map.py \
  --snapshot "$CANONICAL_SNAPSHOT" \
  --base-map "$CANONICAL_SNAPSHOT/haltere-campaniform-sensory-v1.json" \
  --output "$CANONICAL_SNAPSHOT/haltere-timing-afferents-v1.json" \
  2>&1 | tee "$OUTPUT_ROOT/logs/prepare-haltere-timing.log"

"$PY" scripts/data/prepare_neural_viewer_graph.py \
  --snapshot "$CANONICAL_SNAPSHOT" \
  --groups "$CANONICAL_SNAPSHOT/embodiment-groups-v0.json" \
  --motor-map "$CANONICAL_SNAPSHOT/wing-motor-neurons-v0.json" \
  --output "$CANONICAL_EMBODIMENT/neural-viewer-graph-v1.json" \
  2>&1 | tee "$OUTPUT_ROOT/logs/prepare-viewer-graph.log"

"$PY" scripts/data/prepare_flybody_neutral_trim.py \
  --output-pattern "$CANONICAL_DERIVED/wing-pattern-neutral-trim-v1.npy" \
  --output-metadata "$CANONICAL_DERIVED/wing-pattern-neutral-trim-v1.json" \
  2>&1 | tee "$OUTPUT_ROOT/logs/prepare-neutral-trim.log"

# Preserve the canonical logical path embedded in metadata, then copy the bytes
# out of the disposable worktree for the reproduction artifact bundle.
cp "$CANONICAL_DERIVED/wing-pattern-neutral-trim-v1.npy" "$DERIVED/wing-pattern-neutral-trim-v1.npy"
cp "$CANONICAL_DERIVED/wing-pattern-neutral-trim-v1.json" "$DERIVED/wing-pattern-neutral-trim-v1.json"

"$PY" scripts/embodiment/calibrate_neural_runtime.py \
  --snapshot "$CANONICAL_SNAPSHOT" \
  --mapping "$CANONICAL_SNAPSHOT/retinotopic-vision-v1.json" \
  --backend gpu \
  --zero-input-steps 24 \
  --output "$CANONICAL_EMBODIMENT/neural-runtime-calibration-v1.json" \
  2>&1 | tee "$OUTPUT_ROOT/logs/calibrate-neural-runtime.log"

"$PY" scripts/data/prepare_plastic_fast_graph.py \
  --snapshot "$SNAP" \
  --graph "$INPUTS/plastic-fast-graph-v1.bin" \
  --report "$REPORT/plastic-fast-graph-v1.md" \
  2>&1 | tee "$OUTPUT_ROOT/logs/prepare-plastic-fast-graph.log"

# 3) Static reproducibility checks. The privacy rewrite changed the Git commit
# identity embedded in JSON provenance, but not scientific source, conditions,
# upstream bytes, or generated scientific content. Binary artifacts therefore
# remain byte-identical. For provenance-bearing JSON, validate both the actual
# public bytes and a reference-equivalent byte stream obtained by replacing only
# the public Git identity and the resulting chained artifact hashes with their
# recorded original-experiment values. Every artifact is checked individually,
# so a scientific-content change still fails at its source artifact.
"$PY" - "$CONFIG" "$SNAP" "$DERIVED" "$INPUTS" "$OUTPUT_ROOT/source/wing_pattern_fmech.npy" "$OUTPUT_ROOT/static-hash-check.json" "$ORIGINAL_EXPERIMENT_SHA" "$PUBLIC_EQUIVALENT_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
config=Path(sys.argv[1]); snap=Path(sys.argv[2]); derived=Path(sys.argv[3]); inputs=Path(sys.argv[4]); wing=Path(sys.argv[5]); out=Path(sys.argv[6])
original_sha=sys.argv[7]; public_sha=sys.argv[8]
cfg=json.loads(config.read_text())
expected=cfg["static_reference"]
def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
paths={
 "snapshot_manifest": snap/"manifest.json",
 "embodiment_groups": snap/"embodiment-groups-v0.json",
 "retina_map": snap/"retinotopic-vision-v1.json",
 "wing_motor_map": snap/"wing-motor-neurons-v0.json",
 "body_motor_map": snap/"body-motor-neurons-v0.json",
 "haltere_full_map": snap/"haltere-campaniform-sensory-v1.json",
 "haltere_timing_map": snap/"haltere-timing-afferents-v1.json",
 "viewer_graph": derived/"neural-viewer-graph-v1.json",
 "neural_calibration": derived/"neural-runtime-calibration-v1.json",
 "measured_wingbeat": wing,
 "neutral_trim_pattern": derived/"wing-pattern-neutral-trim-v1.npy",
 "neutral_trim_metadata": derived/"wing-pattern-neutral-trim-v1.json",
}
actual_hashes={name:sha(path) for name,path in paths.items()}
# All substitutions are fixed-length SHA-256/Git-SHA identity strings. Replacing
# them in raw bytes preserves each generator's original JSON formatting exactly.
replacements={public_sha:original_sha}
for name, actual in actual_hashes.items():
    replacements[actual]=expected["artifacts"][name]
def reference_equivalent_sha(path):
    data=path.read_bytes()
    for current, recorded in replacements.items():
        data=data.replace(current.encode("ascii"), recorded.encode("ascii"))
    return sha_bytes(data)
checks={}
for name,p in paths.items():
    actual=actual_hashes[name]; exp=expected["artifacts"][name]
    equivalent=reference_equivalent_sha(p) if p.suffix.lower()==".json" else actual
    match=actual==exp or equivalent==exp
    checks[name]={
        "path":str(p),
        "expected_sha256":exp,
        "actual_sha256":actual,
        "reference_equivalent_sha256":equivalent,
        "comparison":"byte-identical" if actual==exp else "privacy-provenance-equivalent",
        "match":match,
    }
for name, meta in expected["plastic_fast_graph_components"].items():
    p=inputs/name; actual=sha(p); exp=meta["sha256"]
    checks[name]={"path":str(p),"expected_sha256":exp,"actual_sha256":actual,"reference_equivalent_sha256":actual,"comparison":"byte-identical","match":actual==exp,"size_bytes":p.stat().st_size}
from virtual_fly.reproducibility import validate_production_snapshot
snapshot_actual=validate_production_snapshot(snap)["snapshot_sha256"]
snapshot_match=snapshot_actual==expected["snapshot_sha256"]
result={
    "original_experiment_git_sha":original_sha,
    "public_equivalent_git_sha":public_sha,
    "snapshot_sha256_expected":expected["snapshot_sha256"],
    "snapshot_sha256_actual":snapshot_actual,
    "snapshot_match":snapshot_match,
    "artifacts":checks,
    "all_match":snapshot_match and all(x["match"] for x in checks.values()),
}
out.write_text(json.dumps(result,indent=2)+'\n')
if not result["all_match"]:
    print(json.dumps(result,indent=2), file=sys.stderr)
    raise SystemExit("static canonical artifact hash mismatch")
print("static_canonical_hashes=PASS")
PY

if [ "$STOP_AFTER_STATIC" = "1" ]; then
  echo "canonical_stop_after_static=PASS"
  exit 0
fi

SCALE="$($PY -c 'import json,sys; print(float(json.load(open(sys.argv[1]))["synapse_scale"]))' "$DERIVED/neural-runtime-calibration-v1.json")"
[ "$SCALE" = "0.005" ] || { echo "unexpected neural synapse scale: $SCALE" >&2; exit 4; }
export VF_NEURAL_SYNAPSE_SCALE="$SCALE"
export VF_FLYPPY_VISION_MODE=direct-ray
export VF_FLYPPY_OMMATIDIA_RAYS=13

# 4) Explicit global-v0 initial weights-only checkpoint.
"$PY" - "$SNAP" "$SNAP/embodiment-groups-v0.json" "$INITIAL/checkpoint" <<'PY' 2>&1 | tee "$OUTPUT_ROOT/logs/save-initial-checkpoint.log"
from pathlib import Path
import json, sys
from virtual_fly.runtime.neural_bridge import PopulationNeuralBridgeClient
snapshot, groups, output = map(Path, sys.argv[1:4])
with PopulationNeuralBridgeClient(snapshot=snapshot, groups=groups, slots=1) as brain:
    brain.ping()
    if brain.global_weight_version != 0:
        raise RuntimeError(f"fresh bridge started at v{brain.global_weight_version}")
    result=brain.save_checkpoint(output)
Path(str(output.parent / 'save-result.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY

# 5) Exact canonical training conditions. Defaults that affect scientific
# semantics are deliberately repeated as explicit arguments.
"$PY" scripts/embodiment/train_flyppy_population_packed.py \
  --episodes 6 \
  --population 2 \
  --snapshot "$SNAP" \
  --groups "$SNAP/embodiment-groups-v0.json" \
  --retinotopic-map "$SNAP/retinotopic-vision-v1.json" \
  --haltere-sensory-map "$SNAP/haltere-timing-afferents-v1.json" \
  --haltere-sensory-kind male-cns-haltere-timing-afferents-inferred-v1 \
  --haltere-current-gain 0.05 \
  --haltere-transduction interaction-load-v2 \
  --wing-motor-map "$SNAP/wing-motor-neurons-v0.json" \
  --body-motor-map "$SNAP/body-motor-neurons-v0.json" \
  --viewer-graph "$DERIVED/neural-viewer-graph-v1.json" \
  --neutral-trim-pattern "$DERIVED/wing-pattern-neutral-trim-v1.npy" \
  --neutral-trim-strength 1.0 \
  --vertical-steering-gain 1.0 \
  --measured-steering-gain 1.0 \
  --steering-tau-ms 12.0 \
  --steering-spike-increment 0.85 \
  --photoreceptor-current-gain 2.0 \
  --reward-current 2.0 \
  --aversive-current 2.0 \
  --reinforcement-steps 4 \
  --gate-near-miss-aversive-floor-fraction 0.25 \
  --gate-near-miss-distance-mm 2.0 \
  --output-dir "$TRAIN" \
  --fresh \
  --seed 0 \
  --fixed-course-seed 0 \
  --gate-count 6 \
  --environment-version v7 \
  --flight-body-version v7 \
  --max-control-steps 96 \
  --physics-steps 10 \
  --checkpoint-every 2 \
  --curriculum-mode boundary-band \
  --boundary-batch-size 6 \
  --launch-mode async \
  --trajectory-stride 8 \
  2>&1 | tee "$OUTPUT_ROOT/logs/training.log"

"$PY" - "$TRAIN/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s["episodes_this_run"] == 6, s["episodes_this_run"]
assert len(s["episode_results"]) == 6, len(s["episode_results"])
assert s["global_weight_version_start"] == 0
assert s["global_weight_version_end"] == 6
assert s["checkpoint_semantics"] == "global-weights-only-v1"
assert s["checkpoint_neural_step_semantics"] == "aggregate-slot-neural-step-count-v1"
print("canonical_training_contract=PASS")
PY

# 6) Full-edge initial/final stored-weight delta statistics.
"$PY" - "$INITIAL/checkpoint/weights.f32le" "$TRAIN/checkpoint/weights.f32le" "$OUTPUT_ROOT/weight-delta.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
initial=Path(sys.argv[1]); final=Path(sys.argv[2]); out=Path(sys.argv[3])
a=np.memmap(initial,dtype='<f4',mode='r'); b=np.memmap(final,dtype='<f4',mode='r')
if a.shape != b.shape: raise SystemExit(f"weight shape mismatch: {a.shape} vs {b.shape}")
thresholds=[0.0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
counts={str(t):0 for t in thresholds}; strengthened=weakened=0; max_abs=0.0; sum_abs=sum_sq=sum_delta=0.0; nonzero=[]
chunk=1_000_000
for lo in range(0,a.size,chunk):
    d=b[lo:lo+chunk].astype(np.float64)-a[lo:lo+chunk].astype(np.float64)
    ad=np.abs(d); max_abs=max(max_abs,float(ad.max(initial=0))); sum_abs+=float(ad.sum()); sum_sq+=float(np.square(d).sum()); sum_delta+=float(d.sum())
    strengthened += int(np.count_nonzero(d>0)); weakened += int(np.count_nonzero(d<0))
    for t in thresholds: counts[str(t)] += int(np.count_nonzero(ad>t))
    nz=ad[ad>0]
    if nz.size: nonzero.append(nz.astype(np.float32))
allnz=np.concatenate(nonzero) if nonzero else np.empty(0,dtype=np.float32)
q={str(x):float(np.quantile(allnz,x)) for x in (0.5,0.9,0.99,0.999)} if allnz.size else {}
result={"edge_count":int(a.size),"changed_exact":counts["0.0"],"strengthened":strengthened,"weakened":weakened,"max_abs_delta":max_abs,"mean_abs_delta":sum_abs/a.size,"rms_delta":(sum_sq/a.size)**0.5,"l1_abs_delta":sum_abs,"delta_sum":sum_delta,"threshold_counts":counts,"abs_delta_quantiles_nonzero":q}
out.write_text(json.dumps(result,indent=2)+'\n')
if result["changed_exact"] <= 0: raise SystemExit("canonical training changed zero weights")
print(f"weight_delta=PASS changed={result['changed_exact']}")
PY

# 7) Checkpoint load validation in a new bridge process.
"$PY" - "$SNAP" "$SNAP/embodiment-groups-v0.json" "$INITIAL/checkpoint" "$TRAIN/checkpoint" "$OUTPUT_ROOT/checkpoint-validation.json" <<'PY' 2>&1 | tee "$OUTPUT_ROOT/logs/checkpoint-validation.log"
import json,sys
from pathlib import Path
from virtual_fly.runtime.neural_bridge import PopulationNeuralBridgeClient
snap,groups,initial,final,out=map(Path,sys.argv[1:6]); result={}
for name,path,gv in (("initial",initial,0),("final",final,6)):
    with PopulationNeuralBridgeClient(snapshot=snap,groups=groups,slots=1) as brain:
        brain.ping(); loaded=brain.load_checkpoint(path,global_weight_version=gv)
        result[name]={"loaded_step":int(loaded["step"]),"global_weight_version":int(brain.global_weight_version),"path":str(path)}
        if int(brain.global_weight_version)!=gv: raise RuntimeError(f"{name} loaded v{brain.global_weight_version}, expected v{gv}")
out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
PY

# 8) Frozen evaluation: plasticity OFF (no --plasticity) and DAN stimulation
# OFF (both current amplitudes exactly 0). Detection of pass/collision remains on.
EVAL_COMMON=(
  --snapshot "$SNAP"
  --viewer-graph "$DERIVED/neural-viewer-graph-v1.json"
  --calibration "$DERIVED/neural-runtime-calibration-v1.json"
  --haltere-sensory-map "$SNAP/haltere-timing-afferents-v1.json"
  --haltere-sensory-kind male-cns-haltere-timing-afferents-inferred-v1
  --haltere-current-gain 0.05
  --haltere-transduction interaction-load-v2
  --environment-version v7
  --flight-body-version v7
  --neutral-trim-pattern "$DERIVED/wing-pattern-neutral-trim-v1.npy"
  --neutral-trim-strength 1.0
  --vertical-steering-gain 1.0
  --measured-steering-gain 1.0
  --steering-tau-ms 12.0
  --steering-spike-increment 0.85
  --course-seed 0
  --gate-count 6
  --spawn-x-mm 8.91
  --spawn-z-mm 11.302931914869099
  --initial-speed-mm-s 400.0
  --initial-vz-mm-s 0.0
  --physics-steps 10
  --physics-substep-stride 3
  --max-control-steps 128
  --photoreceptor-current-gain 2.0
  --reward-current 0.0
  --aversive-current 0.0
  --reinforcement-steps 4
  --neural-telemetry-stride 5
  --playback-fps 60
  --reward-from-gate 1
  --timeout-s 120
)

"$PY" scripts/embodiment/capture_flyppy_preview_playback.py \
  --fresh-brain \
  --output "$EVAL/initial/playback.json" \
  "${EVAL_COMMON[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/logs/evaluation-initial.log"

"$PY" scripts/embodiment/capture_flyppy_preview_playback.py \
  --production "$TRAIN" \
  --output "$EVAL/final/playback.json" \
  "${EVAL_COMMON[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/logs/evaluation-final.log"

"$PY" - "$EVAL/initial/playback.json" "$EVAL/final/playback.json" <<'PY'
import json,sys
for p in sys.argv[1:]:
    x=json.load(open(p))
    assert x["plasticity"] is False
    assert x["vision_runtime"] == "direct-ray"
    assert x["vision_rays_per_ommatidium"] == 13
print("frozen_evaluation_contract=PASS")
PY

# 9) Existing report/history generators.
"$PY" scripts/analysis/export_flyppy_report.py --summary "$TRAIN/summary.json" --output-dir "$REPORT" \
  2>&1 | tee "$OUTPUT_ROOT/logs/export-report.log"
"$PY" scripts/analysis/export_flyppy_population_report.py --summary "$TRAIN/summary.json" --output "$REPORT/population.md" \
  2>&1 | tee "$OUTPUT_ROOT/logs/export-population-report.log"
"$PY" scripts/analysis/append_flyppy_run_history.py --summary "$TRAIN/summary.json" --output "$REPORT/history.csv" \
  2>&1 | tee "$OUTPUT_ROOT/logs/export-history.log"

# 10) Reproduction manifest/report. Async GPU/MuJoCo results are compared but
# are intentionally not required to be bit-identical to the reference run.
"$PY" - "$CONFIG" "$REFERENCE_MANIFEST" "$OUTPUT_ROOT" "$EXPECTED_SHA" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
config=Path(sys.argv[1]); reference=Path(sys.argv[2]); root=Path(sys.argv[3]); expected_sha=sys.argv[4]
cfg=json.loads(config.read_text()); ref=json.loads(reference.read_text())
train=json.loads((root/'training/summary.json').read_text()); weight=json.loads((root/'weight-delta.json').read_text()); checks=json.loads((root/'static-hash-check.json').read_text()); ckpt=json.loads((root/'checkpoint-validation.json').read_text())
initial=json.loads((root/'evaluation/initial/playback.json').read_text()); final=json.loads((root/'evaluation/final/playback.json').read_text()); versions=json.loads((root/'runtime-versions.json').read_text())
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
reference_result=cfg['reference_result']
repro={
 'schema_version':1,'kind':'virtual-fly-canonical-v1-reproduction-run','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'canonical_reference':{'original_experiment_git_sha':'7fa464aad7269d34f46f1171080b51e095d1d811','public_equivalent_git_sha':expected_sha,'config_sha256':sha(config),'reference_manifest_sha256':sha(reference),'reference_result':reference_result},
 'code':train['reproducibility']['code'],'dependencies':train['reproducibility']['dependency_lock'],'runtime_versions':versions,
 'static_reproducibility':checks,
 'training':{
   'conditions':cfg['training'],'episodes':train['episodes_this_run'],'population':train['population'],'launch_mode':train['launch_mode'],'curriculum_mode':train['curriculum_mode'],
   'global_weight_version_start':train['global_weight_version_start'],'global_weight_version_end':train['global_weight_version_end'],'aggregate_neural_step':train['checkpoint_neural_step'],
   'checkpoint_semantics':train['checkpoint_semantics'],'step_semantics':train['checkpoint_neural_step_semantics'],'episode_results':train['episode_results'],'curriculum':train['curriculum'],
 },
 'weight_change':weight,'checkpoint_validation':ckpt,
 'frozen_evaluation':{
   'conditions':cfg['evaluation'],
   'initial':{'passed_gates':initial['passed_gates'],'terminal_reason':initial['terminal_reason'],'control_steps':initial['control_steps'],'global_weight_version':initial['global_weight_version'],'checkpoint_neural_step':initial['checkpoint_neural_step']},
   'final':{'passed_gates':final['passed_gates'],'terminal_reason':final['terminal_reason'],'control_steps':final['control_steps'],'global_weight_version':final['global_weight_version'],'checkpoint_neural_step':final['checkpoint_neural_step']},
 },
 'reference_comparison':{
   'final_global_v_match':train['global_weight_version_end']==reference_result['final_global_weight_version'],
   'weight_changed_reference':reference_result['changed_edges_exact'],'weight_changed_reproduction':weight['changed_exact'],
   'initial_outcome_reference':reference_result['initial_frozen'],'final_outcome_reference':reference_result['final_frozen'],
 },
 'reproducibility_scope':{
   'deterministic_hash_required':['scientific snapshot composite hash','static JSON content modulo privacy-rewrite Git provenance identity','measured wingbeat bytes','neutral trim pattern bytes','PlasticFastGraph binary arrays'],
   'not_bitwise_required':['async slot completion/commit ordering beyond contract','GPU floating-point weight trajectory','MuJoCo/GPU trajectory across hardware/platforms','final learned weight file hash','frozen trajectory coordinates/outcome across different hardware'],
   'required_semantic_invariants':['clean pinned scientific SHA','6 completed training episodes','global weight v0 -> v6','weights changed','weights-only checkpoint loads','frozen evaluation completes with plasticity OFF and DAN current amplitudes zero'],
 }
}
if repro['code']['git_sha'] != expected_sha or repro['code']['dirty'] is not False: raise SystemExit('reproduction scientific worktree was not clean pinned SHA')
if train['episodes_this_run'] != 6 or train['global_weight_version_end'] != 6: raise SystemExit('training semantic invariant failed')
if weight['changed_exact'] <= 0: raise SystemExit('weight update semantic invariant failed')
(root/'manifest.json').write_text(json.dumps(repro,indent=2)+'\n')
lines=[
 '# Canonical v1 reproduction run','',f'- scientific Git SHA: `{expected_sha}`',f'- dirty: `{str(repro["code"]["dirty"]).lower()}`',f'- static artifact hashes: **{"PASS" if checks["all_match"] else "FAIL"}**',f'- training: {train["episodes_this_run"]} episodes, global v{train["global_weight_version_start"]} -> v{train["global_weight_version_end"]}',f'- changed stored weights: {weight["changed_exact"]:,}',f'- aggregate neural step: {train["checkpoint_neural_step"]}','', '## Frozen evaluation','',f'- initial: {initial["passed_gates"]} gate(s), terminal `{initial["terminal_reason"]}`, control {initial["control_steps"]}',f'- final: {final["passed_gates"]} gate(s), terminal `{final["terminal_reason"]}`, control {final["control_steps"]}','', 'The recorded canonical result remains the reference. This directory only verifies the end-to-end reproduction path; it does not replace the reference result.','', '## Reproducibility level','', '- Static binary artifacts are required to match the recorded SHA-256 hashes byte-for-byte. Provenance-bearing JSON must match after normalizing only the privacy-rewrite Git identity and its chained artifact hashes back to the recorded original-experiment values.', '- Async/GPU/MuJoCo training and trajectories are not required to match bit-for-bit across hardware. The semantic invariants listed in `manifest.json` are required.',
]
(root/'report/reproduction.md').write_text('\n'.join(lines)+'\n')
print(f"reproduction_manifest={root/'manifest.json'}")
print(f"reproduction_report={root/'report/reproduction.md'}")
PY

# Preserve exact reproduction inputs alongside the run.
cp "$CONFIG" "$OUTPUT_ROOT/canonical-config.json"
cp "$REFERENCE_MANIFEST" "$OUTPUT_ROOT/reference-manifest.json"
cp "$SCRIPT_DIR/privacy-redaction-provenance.json" "$OUTPUT_ROOT/privacy-redaction-provenance.json"

# Final scientific SHA/clean-state check. Generated files live outside the
# worktree (except ignored build/source caches), so Git remains clean.
[ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]
[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]
echo "canonical_reproduction=PASS"
echo "canonical_reproduction_output=$OUTPUT_ROOT"
