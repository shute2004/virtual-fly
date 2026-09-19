# Data provenance and reproducibility

[English](data-and-reproducibility.md) · [日本語](../ja/data-and-reproducibility.md) · [简体中文](../zh-CN/data-and-reproducibility.md)

## 1. Reference dataset

The initial reference dataset is the adult male Drosophila whole-CNS connectome `male-cns:v1.0`, released by HHMI Janelia and collaborators.

The official distribution provides both neuPrint API access and bulk downloads. The dataset itself is distributed under CC-BY.

At acquisition time, record at least:

- dataset name;
- dataset version;
- source URL;
- acquisition date;
- upstream license;
- upstream citation;
- file hashes;
- conversion-tool version;
- conversion configuration.

## 2. Do not edit source data in place

External data are treated as a read-only source snapshot.

```text
upstream source
      ↓
verified local snapshot
      ↓
normalized static internal data
      ↓
mutable state owned by each simulated individual
```

Learning and plasticity must never be written back into the source connectome.

## 3. Data layers

### Layer A: upstream data

The original downloaded files or official API responses.

### Layer B: normalized static data

Read-only data converted into the project's internal representation, such as:

- neuron tables;
- synapses or aggregated connections;
- annotations;
- morphology references;
- provenance information.

### Layer C: simulated-individual state

State that may change independently for each virtual fly:

- neural state;
- functional synaptic state;
- structural deltas;
- neuromodulation state;
- body state;
- experiment history.

## 4. Provenance categories

Values, parameters, and connections should be classified where practical.

| Category | Meaning |
|---|---|
| `observed` | directly measured or present in upstream data |
| `literature` | adopted from published literature |
| `inferred` | inferred from measured information |
| `assumed` | assumed to fill an unknown part of the model |
| `calibrated` | numerically or experimentally calibrated |

Public documentation, papers, and visualizations must not blur these categories.

## 5. Per-experiment provenance record

Each experiment should emit a machine-readable provenance record. For example:

```yaml
experiment_id: exp-...
fly_id: fly-...
parent_checkpoint: ...
code_commit: ...
connectome:
  dataset: male-cns:v1.0
  normalized_snapshot_hash: ...
neural_model:
  name: ...
  version: ...
plasticity_model:
  name: ...
  version: ...
body:
  model: flybody
  version: ...
environment:
  name: flyppy
  config_hash: ...
rng:
  seed: ...
```

These identifiers are machine-readable field names. Human-facing prose does not need to imitate them.

## 6. Reproducibility levels

### R0: condition identification

The input data, configuration, and code revision can be identified exactly.

### R1: deterministic replay in the same environment

The same execution backend and environment reproduce the same result.

### R2: cross-backend agreement

Different execution backends such as CPU, GPU, and WebGPU produce equivalent results within a defined tolerance.

### R3: statistical reproducibility

When nondeterminism remains, repeated runs support the same statistical conclusion.

Not every experiment must satisfy R2, but each experiment should state which level it reaches.

## 7. External assets

External assets are not vendored into the repository by default. Important examples include:

- MaleCNS;
- FlyBody;
- FlyGym / NeuroMechFly;
- MuJoCo.

When required, retrieval scripts and fixed versions should be provided.

## 8. Large generated artifacts

Large files are not tracked in Git, including:

- raw connectome dumps;
- large EM datasets;
- large meshes;
- checkpoints;
- trajectory data;
- rendered videos;
- detailed profiling traces.

Instead, Git tracks the small provenance records, configuration, hashes, and summary reports required to reacquire or regenerate them.

## 9. Hugging Face checkpoint distribution

Checkpoint payloads intended for third-party download are kept separate from the GitHub source repository. The `shute2004/virtual-fly` Hugging Face project overview is prepared; publication contents for the four checkpoint repositories below are prepared locally, but the repositories themselves are not yet created or published.

The planned publication structure deliberately separates:

- canonical v1 initial checkpoint: `virtual-fly-initial-YYYYMMDD`;
- canonical v1 trained checkpoint: `virtual-fly-trained-YYYYMMDD`;
- historical Before-video checkpoint: `virtual-fly-video-before`;
- historical After-video checkpoint: `virtual-fly-video-after`.

The `YYYYMMDD` suffix is selected only on the actual publication day. Historical video checkpoints are not canonical initial/trained checkpoints, even when the same runtime can inspect their files.

When published, each checkpoint repository will contain the native checkpoint directory plus lightweight `README.md`, attribution, provenance, and SHA-256 records. Full trajectories, frame directories, raw source datasets, unrelated checkpoints, and build artifacts are excluded unless they become directly necessary for using or verifying that checkpoint.

The checkpoint weights derive from MaleCNS `male-cns:v1.0`, licensed by the official source under CC BY 4.0. The raw MaleCNS dataset is not mirrored on Hugging Face; users obtain it from the official source or through the canonical acquisition/reproduction path. FlyBody, FlyGym, and MuJoCo assets are likewise not copied into checkpoint repositories merely for convenience.

The prepared repository cards, exact checkpoint hashes, and required file lists are tracked in [`../../release/huggingface/`](../../release/huggingface/README.md).

## 10. Historical provenance paths

A small number of archived development notes and tracked historical reports retain the structure of absolute local filesystem paths from the machine on which the original run, traceback, or report was recorded. Before the first public release, the local OS username in exactly 14 such files was replaced with `<local-user>` for privacy; the surrounding paths, historical result values, and scientific content were otherwise preserved. The original-to-public blob mapping is recorded in [`../../canonical/canonical-v1/privacy-redaction-provenance.json`](../../canonical/canonical-v1/privacy-redaction-provenance.json) without reproducing the redacted username. These paths are not installation requirements or dependencies of the current code. Current public documentation and commands use repository-relative or portable paths.

## 11. Public browser experiments

Results returned by browsers are treated as outside the trusted execution boundary.

At minimum, each result should identify:

- experiment package version;
- initial-state hash;
- configuration hash;
- execution backend;
- result hash;
- summary metrics.

Important findings should be rerun on the server side or across multiple independent clients.

## 12. Personal data

Public experiments should not collect personal information or browser identifiers that are unnecessary for the neuroscience experiment.

If execution-environment information is required for distributed computation, its purpose and retention scope must be explicit.
