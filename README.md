# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly` is a research project that starts from the released adult male *Drosophila melanogaster* MaleCNS connectome and evolves neural activity, neuromodulation, and local synaptic plasticity in closed loop with a FlyBody / MuJoCo body and a physical environment.

It is **not** an artificial neural network trained on the fly connectome. The production path does not use backpropagation, gradient descent, Q-learning, policy gradients, an external neural-network controller, or a hand-written obstacle policy. Learning-related weight changes are produced inside the simulated nervous system by local activity, eligibility, and dopaminergic modulation.

> **Project status:** canonical experiment v1 and its end-to-end reproducer are complete. The canonical run demonstrates a fully traceable current-code execution chain and measurable local-plasticity-driven changes to stored MaleCNS weights. It did **not** show categorical behavioral improvement in the short canonical run.

**Scientific references:** [`docs/en/references.md`](docs/en/references.md) collects the major primary papers and official upstream resources used by the project, with an evidence map showing which implementation decisions each source supports.

## Historical visualization

![Historical v240 to v966 Before/After visualization](docs/assets/historical-v240-v966-before-after.gif)

This animated comparison is derived from the publication master `before-after-neural.mp4` and is shown at **1.5× speed** for README viewing. It visualizes the historical `v240 → v966` Before/After lineage and the body + neural viewer together. It is intentionally placed near the top because it provides the clearest visual overview of the project, but it is **not canonical evidence**: v240/v960/v966/v1704 belong to older lineages with different provenance and, in some cases, different semantics. Release asset provenance and hashes are recorded in [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json).

## Canonical result

The publication-facing reference is [`canonical/canonical-v1/`](canonical/canonical-v1/README.md). It is intentionally separate from older development lineages.

| Item | Canonical v1 |
|---|---|
| Scientific code | `7fa464aad7269d34f46f1171080b51e095d1d811` (clean) |
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

The end-to-end reproducer was independently rerun from a clean `7fa464a` checkout on 2026-09-19. All source-derived static artifact hashes matched the recorded reference, training reached global v6, exactly 2,163,179 stored edges changed, both checkpoints reloaded successfully, and both frozen outcomes matched the recorded reference.

See:

- [`canonical/canonical-v1/reference-report.md`](canonical/canonical-v1/reference-report.md) — compact canonical result
- [`canonical/canonical-v1/reference-manifest.json`](canonical/canonical-v1/reference-manifest.json) — complete provenance
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
- a Rust toolchain with Cargo
- MuJoCo-compatible local graphics/compute support

The current canonical reference was produced on macOS / Apple Metal GPU. Static source-derived artifacts are hash-checked; async GPU/MuJoCo trajectories are not promised to be bit-identical across hardware.

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

## Checkpoints and large artifacts

Large checkpoints are distributed separately from the GitHub source repository through the Hugging Face project hub:

- [Hugging Face: `shute2004/virtual-fly`](https://huggingface.co/shute2004/virtual-fly)

The publication layout separates four checkpoint repositories:

- `virtual-fly-initial-YYYYMMDD` — canonical v1 initial checkpoint;
- `virtual-fly-trained-YYYYMMDD` — canonical v1 checkpoint after the six canonical training episodes;
- `virtual-fly-video-before` — historical checkpoint actually used for the Before side of the published comparison video;
- `virtual-fly-video-after` — historical checkpoint actually used for the After side of the published comparison video.

The two dated canonical names are finalized only on the actual publication day. The historical video pair is **not** the canonical initial/trained pair; internal versions such as v240/v966 are retained only in artifact provenance. Raw MaleCNS data, FlyBody assets, trajectories, frame directories, and unrelated development artifacts are not mirrored merely for completeness.

The exact publication structure, model-card templates, required file lists, attribution, and prepared checkpoint hashes are tracked in [`release/huggingface/`](release/huggingface/README.md).

## Reproduce canonical v1

The canonical reproducer acquires fresh official sources, reconstructs the snapshot and derived artifacts, runs the six-episode canonical training, validates the initial/final checkpoints, performs frozen evaluation, and emits a provenance manifest.

```bash
bash canonical/canonical-v1/reproduce.sh
```

By default, its large output bundle is written **outside the repository**:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/reproductions/
```

Use `VF_CANONICAL_OUTPUT_ROOT=/path/to/output` to choose another location. The script always executes the scientific workload in an isolated clean worktree pinned to `7fa464a`.

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

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). A GitHub Release intended for archival should be tagged from the publication branch and archived with Zenodo; see [`docs/internal/en/release-and-zenodo.md`](docs/internal/en/release-and-zenodo.md).

## License

Original `virtual-fly` source code and documentation are released under the [MIT License](LICENSE). Upstream datasets, software, model assets, and generated media that incorporate third-party material remain subject to their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
