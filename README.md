# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly` is a research project that starts from the released adult male *Drosophila melanogaster* MaleCNS connectome and evolves neural activity, neuromodulation, and local synaptic plasticity in closed loop with a FlyBody / MuJoCo body and a physical environment.

It is **not** an artificial neural network trained on the fly connectome. The production path does not use backpropagation, gradient descent, Q-learning, policy gradients, an external neural-network controller, or a hand-written obstacle policy. Learning-related weight changes are produced inside the simulated nervous system by local activity, eligibility, and dopaminergic modulation.

> **Project status:** canonical experiment v1 and its end-to-end reproducer are complete. The canonical run demonstrates a fully traceable current-code execution chain and measurable local-plasticity-driven changes to stored MaleCNS weights. It did **not** show categorical behavioral improvement in the short canonical run.

**Scientific references:** [`docs/en/references.md`](docs/en/references.md) collects the major primary papers and official upstream resources used by the project, with an evidence map showing which implementation decisions each source supports.

## Historical visualization

![Historical v240 to v966 Before/After visualization](docs/assets/historical-v240-v966-before-after.gif)

This animated comparison is derived from a README-specific source rendering of the historical `v240 → v966` Before/After comparison and is shown at **1.5× speed**. It is intentionally placed near the top because it provides the clearest visual overview of the project, but it is **not canonical evidence**. The Before state (v240) was already a historical checkpoint after 240 training episodes, not an untrained MaleCNS initial state. Historical playback disabled plasticity, but task-triggered PAM stimulation remained enabled, and the comparison was not a held-out evaluation. The v966 state also belongs to an older mixed lineage with semantics that differ from canonical v1.

The X-post video, the source rendering used to make this README GIF, and the GitHub Release MP4 are separate renderings of the same historical v240/v966 comparison; they are not identical files. Their roles, dimensions, and hashes are recorded separately in [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json).

## Canonical result

The publication-facing reference is [`canonical/canonical-v1/`](canonical/canonical-v1/README.md). It is intentionally separate from older development lineages.

| Item | Canonical v1 |
|---|---|
| Public scientific code equivalent | `f9c86c904d67ff974f3c43d37aab3619bc93fc1b` (privacy-redacted equivalent of the clean experiment commit) |
| MaleCNS snapshot | 166,700 neurons / 25,582,938 directed edges |
| DAN semantics | released `class=DAN` + dopamine consensus; 338 modulators |
| Body / environment | v7 / v7 |
| Vision | direct-ray, K=13 rays/ommatidium |
| Haltere input | 97-neuron timing subset, gain `0.05`, `interaction-load-v2` |
| Training | 6 episodes, population 2, async shared weights, boundary-band |
| Global weight version | v0 → v6 |
| Aggregate neural step | 484 |
| Exactly changed stored edges | 2,163,179 |
| Frozen initial evaluation | 1 gate, then gate collision at control step 120 |
| Frozen final evaluation | 1 gate, then gate collision at control step 120 |

Frozen evaluation disables both plasticity and task-triggered DAN stimulation (`reward_current=0`, `aversive_current=0`). The short canonical run therefore establishes **weight change under the current local-plasticity semantics**, not improved behavior, generalization, or long-term learning stability.

The recorded canonical experiment and the 2026-09-19 end-to-end verification were actually run from clean commit `7fa464aad7269d34f46f1171080b51e095d1d811`. Before public release, Git history was rewritten only to replace the local OS username inside historical absolute paths. The corresponding public scientific commit is `f9c86c904d67ff974f3c43d37aab3619bc93fc1b`. At that commit, the only tree differences are 14 historical report/provenance files containing the privacy redaction; executable source, canonical conditions, and scientific artifact-generation logic are unchanged. Full training was therefore not rerun solely for this privacy rewrite. The exact mapping is recorded in [`canonical/canonical-v1/privacy-redaction-provenance.json`](canonical/canonical-v1/privacy-redaction-provenance.json).

See:

- [`canonical/canonical-v1/reference-report.md`](canonical/canonical-v1/reference-report.md) — compact canonical result
- [`canonical/canonical-v1/reference-manifest.json`](canonical/canonical-v1/reference-manifest.json) — machine-readable reference provenance; exact reproduction commands are in `reproduce.sh`
- [`docs/en/results.md`](docs/en/results.md) — canonical vs. historical result boundary
- [`docs/en/reproducibility-fixes-2026-09-19.md`](docs/en/reproducibility-fixes-2026-09-19.md) — provenance/semantics audit and fixes

## What is implemented

The current production path is:

```text
Flyppy physical world
        ↓
FlyBody compound-eye geometry + local direct rays
        ↓
released MaleCNS R1-R6 body-ID currents
        ↓
whole-MaleCNS neural dynamics
        ↓
local eligibility + class-DAN-mediated plasticity
        ↓
individual released motor-neuron spikes
        ↓
whole-body peripheral muscle state
        ↓
FlyBody / MuJoCo physical actuation
        ↓
Flyppy physical world
```

Task events do not directly write weights. A gate passage or collision can trigger current injection into selected real DAN populations; subsequent synaptic changes are computed locally inside the neural runtime.

Important current boundaries:

- the MaleCNS source snapshot is immutable;
- visual input preserves local retinotopic R1-R6 identity rather than injecting an external obstacle classifier;
- motor output stays at individual released motor-neuron IDs rather than a population-average action decoder;
- the viewer is observer-only and does not feed state back into training;
- population training shares one global weight state, but episode-local membrane/spike/refractory/trace/modulation/eligibility state is reset between episodes;
- current population checkpoints persist global weights, not a complete persistent biological individual state.

For implementation details, see [`docs/en/architecture.md`](docs/en/architecture.md) and [`docs/en/code-structure.md`](docs/en/code-structure.md).

## What this project does not claim

`virtual-fly` is a biologically grounded computational reconstruction, not a claim that the current simulator is already a complete biological replica of a fly.

In particular, canonical v1 does **not** establish:

- categorical behavioral improvement after the six training episodes;
- task generalization;
- long-term learning stability;
- complete biophysical fidelity of every neuron, synapse, sensory organ, or flight muscle;
- equivalence between historical v240/v960/v966/v1704 results and the current canonical semantics.

Observed upstream data, literature-derived choices, inferred mappings, engineering assumptions, and calibrated parameters are kept distinct where possible.

## Installation

### Requirements

- Python `>=3.12,<3.15`
- [`uv`](https://docs.astral.sh/uv/)
- Rust `>=1.87` with Cargo
- MuJoCo-compatible local graphics support
- for the **full canonical reproduction**: a wgpu-compatible GPU backend for the neural runtime

Basic installation and the validation commands below do not run the canonical experiment. The full `canonical/canonical-v1/reproduce.sh` path does: its neural calibration explicitly uses the GPU backend, and the recorded canonical reference was produced on macOS / Apple Metal GPU. Static source-derived artifacts are hash-checked; async GPU/MuJoCo trajectories are not promised to be bit-identical across hardware.

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
```

Large upstream datasets and generated experiment artifacts are intentionally not stored in Git.

### Basic validation

After installation, a third party can verify the source checkout without downloading MaleCNS data or running the canonical experiment:

```bash
uv run python -m unittest discover -s tests -v
cargo test --workspace
bash -n canonical/canonical-v1/reproduce.sh
```

The first two commands exercise the Python semantic/orchestration tests and Rust neural-runtime tests. The final command checks the canonical reproducer shell entry point without executing the experiment.

## Checkpoint distribution

Checkpoint payloads are intentionally kept out of the GitHub source repository. The Hugging Face project overview is available here:

- [Hugging Face: `shute2004/virtual-fly`](https://huggingface.co/shute2004/virtual-fly)

The four verified checkpoint repositories are published separately on Hugging Face:

- [`virtual-fly-initial-20260920`](https://huggingface.co/shute2004/virtual-fly-initial-20260920) — canonical v1 initial checkpoint;
- [`virtual-fly-trained-20260920`](https://huggingface.co/shute2004/virtual-fly-trained-20260920) — canonical v1 checkpoint after the six canonical training episodes;
- [`virtual-fly-video-before`](https://huggingface.co/shute2004/virtual-fly-video-before) — historical checkpoint actually used for the Before side of the published comparison video;
- [`virtual-fly-video-after`](https://huggingface.co/shute2004/virtual-fly-video-after) — historical checkpoint actually used for the After side of the published comparison video.

The canonical repository date is 2026-09-20. The historical video pair is **not** the canonical initial/trained pair; internal versions such as v240/v966 are retained only in artifact provenance. Raw MaleCNS data, FlyBody assets, trajectories, frame directories, and unrelated development artifacts are not mirrored merely for completeness.

The exact publication structure, model-card templates, required file lists, attribution, and published checkpoint hashes are tracked in [`release/huggingface/`](release/huggingface/README.md).

## Reproduce canonical v1

The canonical reproducer acquires fresh official sources, reconstructs the snapshot and derived artifacts, runs the six-episode canonical training, validates the initial/final checkpoints, performs frozen evaluation, and emits a provenance manifest.

```bash
bash canonical/canonical-v1/reproduce.sh
```

By default, its large output bundle is written **outside the repository**:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/reproductions/
```

Use `VF_CANONICAL_OUTPUT_ROOT=/path/to/output` to choose another location. The public reproducer executes the scientific workload in an isolated clean worktree pinned to the privacy-redacted public equivalent `f9c86c9`. The recorded 2026-09-19 experiment remains attributed to its original pre-rewrite commit `7fa464a`; the project does not claim that historical run was executed at the rewritten SHA.

## Development training path

For current development runs, the generic production entry point is:

```bash
bash scripts/dev/train_flyppy_population.sh
```

`scripts/dev/train_flyppy_v3_population.sh` remains as a compatibility wrapper. Development runs and their continuously updated reports are **not** automatically canonical results.

## Repository layout

```text
canonical/      pinned canonical experiment package and reference provenance
crates/         Rust neural runtime and runners
docs/           multilingual public docs (`en/`, `ja/`, `zh-CN/`), internal notes, and archive
reports/        small tracked diagnostics and historical development reports
scripts/        data preparation, analysis, compatibility CLIs and launchers
src/            current Python package (`virtual_fly`)
tests/          semantic, scheduling, runtime and reproducibility tests
visualization/  observer-only neural viewers
artifacts/      local large data/checkpoints/videos; gitignored
release/        release-asset manifests; large release files remain outside Git
```

See [`docs/README.md`](docs/README.md) for the documentation map.

## Results, provenance, and large data

Historical development results remain available because they are useful provenance, but they must not be silently relabeled as canonical. The distinction is documented in [`docs/en/results.md`](docs/en/results.md).

Large files such as MaleCNS raw downloads, normalized snapshots, checkpoints, trajectories, rendered videos, and build caches are excluded from Git. Small manifests, hashes, and reports are tracked instead. See [`docs/en/data-and-reproducibility.md`](docs/en/data-and-reproducibility.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The initial v0.1.0 publication uses GitHub and Hugging Face; Zenodo/DOI archival is intentionally deferred and may be considered later if a fixed scholarly archive is needed. See [`docs/internal/en/release-and-zenodo.md`](docs/internal/en/release-and-zenodo.md).

## License

Original `virtual-fly` source code and documentation are released under the [MIT License](LICENSE). Upstream datasets, software, model assets, and generated media that incorporate third-party material remain subject to their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
