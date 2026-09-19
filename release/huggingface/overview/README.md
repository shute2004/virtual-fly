---
license: mit
language:
  - en
tags:
  - drosophila
  - connectome
  - computational-neuroscience
  - neuromodulation
  - synaptic-plasticity
  - embodied-simulation
  - reproducible-research
---

# virtual-fly

`virtual-fly` is a research project that starts from the released adult male Drosophila MaleCNS connectome and evolves neural activity, neuromodulation, and local synaptic plasticity in closed loop with a FlyBody / MuJoCo body and a physical environment.

This Hugging Face repository is the **artifact-distribution overview** for the project. Source code, scientific documentation, canonical manifests, and the end-to-end reproducer live on GitHub:

- GitHub: https://github.com/shute2004/virtual-fly
- English documentation: https://github.com/shute2004/virtual-fly/tree/main/docs/en
- 日本語: https://github.com/shute2004/virtual-fly/blob/main/README.ja.md
- 简体中文: https://github.com/shute2004/virtual-fly/blob/main/README.zh-CN.md

## Checkpoint repositories

Two distinct lineages are published. **Do not treat the historical video pair as the canonical initial/trained pair.**

### Canonical v1

The canonical repository names receive their date only on the actual publication day.

| Artifact | Repository | Meaning |
|---|---|---|
| Initial | `shute2004/virtual-fly-initial-YYYYMMDD` | Initial checkpoint of canonical v1, global weight version 0 |
| Trained | `shute2004/virtual-fly-trained-YYYYMMDD` | Canonical v1 after six training episodes, global weight version 6 |

Canonical v1 is pinned to scientific Git commit `7fa464aad7269d34f46f1171080b51e095d1d811`. The short canonical run changed exactly 2,163,179 stored edges, but its frozen initial and final evaluations had the same categorical outcome. It therefore demonstrates current-semantics local-plasticity weight change, not established behavioral improvement.

### Historical Before / After video

| Artifact | Repository | Provenance role |
|---|---|---|
| Before | `shute2004/virtual-fly-video-before` | Exact historical checkpoint used for the Before side of the published comparison video |
| After | `shute2004/virtual-fly-video-after` | Exact historical checkpoint used for the After side of the published comparison video |

The historical cards record their internal development versions (`v240` and `v966`) as provenance, but those numbers are intentionally not used in the public repository names. These checkpoints come from older experiment lineages and are **not canonical v1 evidence**.

## What is hosted here

The artifact repositories contain checkpoint files plus lightweight provenance, hashes, attribution, and a model card. They do not mirror the full research workspace.

Not hosted here by default:

- raw MaleCNS downloads;
- normalized MaleCNS snapshots;
- FlyBody / FlyGym / MuJoCo source or body assets;
- full trajectory directories;
- rendered frame directories;
- unrelated development checkpoints;
- build caches or virtual environments.

The official MaleCNS source and the GitHub canonical reproducer remain the acquisition/reconstruction path for upstream data and derived static artifacts.

## Licensing and attribution

The original `virtual-fly` source code and documentation are MIT licensed. The checkpoint weights derive from the MaleCNS `male-cns:v1.0` dataset, which the official MaleCNS site distributes under CC BY 4.0. The checkpoint repositories are therefore marked `cc-by-4.0` and carry explicit MaleCNS attribution.

FlyBody, FlyGym, and MuJoCo remain under their upstream licenses and are not copied into the checkpoint repositories merely for convenience.

See the GitHub repository's `THIRD_PARTY_NOTICES.md` for the detailed boundary.

## Reproduction

For the current canonical experiment and full source-level provenance, use:

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
bash canonical/canonical-v1/reproduce.sh
```

The Hugging Face checkpoint repositories are a distribution convenience. The GitHub canonical package remains the scientific definition of canonical v1.
