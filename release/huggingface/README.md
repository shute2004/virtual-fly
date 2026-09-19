# Hugging Face publication layout

This directory prepares the Hugging Face checkpoint publication structure for `virtual-fly`. It does **not** upload anything to Hugging Face by itself.

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

Its English, Japanese, and Simplified Chinese overview-card sources are prepared under [`overview/`](overview/README.md).

## Planned checkpoint repositories

The canonical repository date is chosen only on the actual publication day. Do **not** replace `YYYYMMDD` before that day.

| Repository | Lineage | Meaning |
|---|---|---|
| `shute2004/virtual-fly-initial-YYYYMMDD` | canonical v1 | Initial checkpoint of the current canonical experiment |
| `shute2004/virtual-fly-trained-YYYYMMDD` | canonical v1 | Checkpoint after the six canonical training episodes |
| `shute2004/virtual-fly-video-before` | historical | Checkpoint actually used for the Before side of the published Before/After video |
| `shute2004/virtual-fly-video-after` | historical | Checkpoint actually used for the After side of the published Before/After video |

The historical video repositories must never be presented as the initial/trained pair of canonical v1. Their internal global-weight versions are provenance only and intentionally do not appear in the public repository names.

## Prepared repository templates

```text
release/huggingface/
├── overview/README.md
├── checkpoint-artifacts-v1.json
└── repositories/
    ├── virtual-fly-initial-YYYYMMDD/
    │   ├── README.md
    │   ├── ATTRIBUTION.md
    │   ├── provenance.json
    │   └── SHA256SUMS
    ├── virtual-fly-trained-YYYYMMDD/
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

The binary `checkpoint/` directory is deliberately absent from Git. At publication time it is copied from the verified local source identified in `checkpoint-artifacts-v1.json`, then checked against the prepared SHA-256 list before upload.

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

The historical video checkpoints predate that explicit manifest contract. When published, they are for intentional historical replay and video provenance only, not canonical training checkpoints.

## Upstream data and licensing

The checkpoint weights derive from the MaleCNS `male-cns:v1.0` connectome. The official MaleCNS download page licenses the dataset under CC BY 4.0. Checkpoint repositories therefore use `cc-by-4.0` metadata and include explicit upstream attribution.

The raw MaleCNS dataset is **not** mirrored to these Hugging Face repositories. Users should obtain it from the official source or let the canonical reproducer acquire and validate it.

FlyBody, FlyGym, and MuJoCo are not bundled into the checkpoint repositories. Their code/assets remain under their respective upstream terms and should be obtained from the official projects. The checkpoint payload itself does not need those body assets embedded.

See [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) and [`../../docs/en/data-and-reproducibility.md`](../../docs/en/data-and-reproducibility.md).

## Checkpoint-file transport

Hugging Face model repositories use Xet-backed large-file storage. At upload time use the current Hugging Face CLI / Xet workflow rather than committing these binaries to the GitHub source repository.

Official Hugging Face documentation:

- model cards: https://huggingface.co/docs/hub/model-cards
- repository setup: https://huggingface.co/docs/hub/repositories-getting-started
- Xet large-file storage: https://huggingface.co/docs/hub/xet/index
- repository licenses: https://huggingface.co/docs/hub/repositories-licenses

## Publication-day procedure

1. Choose the real publication date and replace `YYYYMMDD` in the two canonical repository names and cards.
2. Create the two dated canonical model repositories and the two fixed historical-video model repositories.
3. Copy each verified checkpoint directory from the source described in `checkpoint-artifacts-v1.json`.
4. Add the prepared card, attribution, provenance, and SHA-256 files.
5. Verify every checkpoint file against `SHA256SUMS` before upload.
6. Upload with Hugging Face's large-file storage path.
7. Update `overview/README.md` with the final dated canonical links and publish it to `shute2004/virtual-fly`.
8. Replace the placeholder canonical links in the GitHub README only after those repositories exist.

This preparation directory does not create the four checkpoint repositories, upload checkpoint payloads, change repository visibility, create a GitHub tag/Release, or create a DOI. The already-private `shute2004/virtual-fly` overview repository may be updated during preparation.
