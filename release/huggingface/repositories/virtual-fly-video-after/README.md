---
license: cc-by-4.0
language:
  - en
tags:
  - drosophila
  - connectome
  - computational-neuroscience
  - neuromodulation
  - synaptic-plasticity
  - virtual-fly
  - checkpoint
  - reproducible-research
---

# virtual-fly historical video After checkpoint

> **Lineage:** Historical After-video state

This repository contains the exact historical checkpoint used for the **After** side of the published `virtual-fly` Before/After comparison video. It is intentionally separated from the current canonical trained checkpoint.

Internal provenance: global weight version **966**, checkpoint neural step **164,754**, source experiment `flyppy-v7h-online-video-v966`. The public repository name deliberately omits `v966`; the version is retained here only as historical provenance.

## Historical semantics

This checkpoint predates the explicit current population-checkpoint contract: its original manifest does **not** declare `global-weights-only-v1` or current class-DAN provenance. It must not be silently relabeled as a canonical checkpoint. Current population code intentionally rejects legacy population manifests unless historical replay is explicitly enabled with `VF_ALLOW_LEGACY_POPULATION_CHECKPOINT=1`. That compatibility opt-in is for intentional historical replay only.

## Video relationship

The recorded After playback used body v7 / environment v7, plasticity disabled during playback, and passed two gates under its historical evaluation condition. The playback was not a held-out evaluation, and task-triggered PAM stimulation remained enabled even though plasticity was off. The v966 checkpoint belongs to a mixed historical lineage assembled across more than one earlier development condition.

This is a historical visualization result, not canonical v1 evidence. Canonical v1 frozen evaluation disables both plasticity and task-triggered DAN stimulation.

The paired historical Before repository is [`shute2004/virtual-fly-video-before`](https://huggingface.co/shute2004/virtual-fly-video-before).

## Source code and reproducibility

- Project overview: https://huggingface.co/shute2004/virtual-fly
- GitHub source: https://github.com/shute2004/virtual-fly
- Canonical package: https://github.com/shute2004/virtual-fly/tree/v0.1.0/canonical/canonical-v1
- Full artifact hash manifest: https://github.com/shute2004/virtual-fly/blob/v0.1.0/release/huggingface/checkpoint-artifacts-v1.json

The checkpoint is not self-contained without a compatible MaleCNS snapshot and the `virtual-fly` runtime. Raw MaleCNS is intentionally not mirrored here; obtain it from the official MaleCNS source or use the GitHub reproduction path.

## Repository contents

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

The checkpoint directory is kept in the native `virtual-fly` checkpoint layout so that the source repository can consume it without repacking. Raw MaleCNS source data and body assets are intentionally not included.

## Integrity

`SHA256SUMS` records the exact hashes of the original checkpoint files selected for publication. Verify the checkpoint before use.

## License and upstream attribution

The checkpoint artifact is distributed as **CC BY 4.0** because its weights derive from the CC BY 4.0 MaleCNS dataset. See `ATTRIBUTION.md`. Original `virtual-fly` source code and documentation remain MIT licensed.
