# Hugging Face publication layout

This directory records the Hugging Face checkpoint publication structure for `virtual-fly`. The tracked files mirror the metadata used for the published repositories; checkpoint binaries remain outside Git.

## Publication boundary

GitHub remains the source of truth for:

- source code and tests;
- public documentation;
- canonical manifests and lightweight reports;
- the canonical reproduction script;
- architecture and scientific descriptions;
- citation and release metadata.

Hugging Face is used to distribute checkpoint payloads separately from the GitHub source repository.

The existing overview repository is:

- https://huggingface.co/shute2004/virtual-fly

Its English, Japanese, and Simplified Chinese overview-card sources are tracked under [`overview/`](overview/README.md).

## Checkpoint repositories

The canonical repository date is 2026-09-20.

| Repository | Lineage | Meaning |
|---|---|---|
| [`shute2004/virtual-fly-initial-20260920`](https://huggingface.co/shute2004/virtual-fly-initial-20260920) | canonical v1 | Initial checkpoint of the current canonical experiment |
| [`shute2004/virtual-fly-trained-20260920`](https://huggingface.co/shute2004/virtual-fly-trained-20260920) | canonical v1 | Checkpoint after the six canonical training episodes |
| [`shute2004/virtual-fly-video-before`](https://huggingface.co/shute2004/virtual-fly-video-before) | historical | Checkpoint actually used for the Before side of the published Before/After video |
| [`shute2004/virtual-fly-video-after`](https://huggingface.co/shute2004/virtual-fly-video-after) | historical | Checkpoint actually used for the After side of the published Before/After video |

The historical video repositories must never be presented as the initial/trained pair of canonical v1. Their internal global-weight versions are provenance only and intentionally do not appear in the public repository names.

## Published repository metadata

```text
release/huggingface/
├── overview/README.md
├── checkpoint-artifacts-v1.json
└── repositories/
    ├── virtual-fly-initial-20260920/
    │   ├── README.md
    │   ├── ATTRIBUTION.md
    │   ├── provenance.json
    │   └── SHA256SUMS
    ├── virtual-fly-trained-20260920/
    │   ├── README.md
    │   ├── ATTRIBUTION.md
    │   ├── provenance.json
    │   └── SHA256SUMS
    ├── virtual-fly-video-before/
    │   ├── README.md
    │   ├── ATTRIBUTION.md
    │   ├── provenance.json
    │   └── SHA256SUMS
    └── virtual-fly-video-after/
        ├── README.md
        ├── ATTRIBUTION.md
        ├── provenance.json
        └── SHA256SUMS
```

The binary `checkpoint/` directory is deliberately absent from Git. For the published repositories it was copied from the verified local source identified in `checkpoint-artifacts-v1.json` and checked against the recorded SHA-256 list before upload.

## Files to upload to each checkpoint repository

Each checkpoint repository should contain only:

```text
README.md
ATTRIBUTION.md
provenance.json
SHA256SUMS
checkpoint/
  manifest.json
  membrane.f32le
  spikes.u32le
  refractory.u32le
  activity-trace.f32le
  modulation.f32le
  weights.f32le
  eligibility.f32le
```

Do not add full trajectories, frame directories, raw MaleCNS downloads, normalized MaleCNS snapshots, FlyBody assets, build caches, or unrelated experiment artifacts merely because they exist locally.

## Checkpoint semantics

Canonical population checkpoints are serialized in the full checkpoint file layout for loader compatibility, but declare `checkpoint_semantics=global-weights-only-v1`. The persistent state between canonical episodes is the global synaptic weight vector. Membrane, spike, refractory, activity-trace, modulation, eligibility, body, and peripheral state are reset between episodes and must not be interpreted as one continuously persistent biological individual.

The historical video checkpoints predate that explicit manifest contract. They are published for intentional historical replay and video provenance only, not as canonical training checkpoints.

## Upstream data and licensing

The checkpoint weights derive from the MaleCNS `male-cns:v1.0` connectome. The official MaleCNS download page licenses the dataset under CC BY 4.0. Checkpoint repositories therefore use `cc-by-4.0` metadata and include explicit upstream attribution.

The raw MaleCNS dataset is **not** mirrored to these Hugging Face repositories. Users should obtain it from the official source or let the canonical reproducer acquire and validate it.

FlyBody, FlyGym, and MuJoCo are not bundled into the checkpoint repositories. Their code/assets remain under their respective upstream terms and should be obtained from the official projects. The checkpoint payload itself does not need those body assets embedded.

See [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) and [`../../docs/en/data-and-reproducibility.md`](../../docs/en/data-and-reproducibility.md).

## Checkpoint-file transport

Hugging Face model repositories use Xet-backed large-file storage. Checkpoint binaries are uploaded through the Hugging Face CLI / Xet workflow rather than committed to the GitHub source repository.

Official Hugging Face documentation:

- model cards: https://huggingface.co/docs/hub/model-cards
- repository setup: https://huggingface.co/docs/hub/repositories-getting-started
- Xet large-file storage: https://huggingface.co/docs/hub/xet/index
- repository licenses: https://huggingface.co/docs/hub/repositories-licenses

## Publication record

The four checkpoint repositories were created privately on 2026-09-20, populated from the sources recorded in `checkpoint-artifacts-v1.json`, and checked against the recorded `SHA256SUMS` before public release. The overview and source repository use the final dated links above.

Zenodo / DOI is not part of v0.1.0.
