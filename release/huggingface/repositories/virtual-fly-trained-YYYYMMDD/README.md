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

# virtual-fly canonical trained checkpoint

> **Lineage:** Canonical v1 post-training state

This is the checkpoint produced after the six training episodes of the current `virtual-fly` canonical experiment. It is part of the same canonical lineage as `virtual-fly-initial-YYYYMMDD`, not the historical Before/After video lineage.

This checkpoint was produced by the recorded canonical v1 experiment at original Git commit `7fa464aad7269d34f46f1171080b51e095d1d811`. Its privacy-redacted public scientific equivalent is `f9c86c904d67ff974f3c43d37aab3619bc93fc1b`; the rewrite changed only historical local-path usernames, not executable source or canonical conditions. This checkpoint is global weight version **6** at aggregate neural step **484**. Exactly **2,163,179 / 25,582,938** stored edges differ from the canonical initial checkpoint.

## Checkpoint semantics

The manifest declares `checkpoint_semantics=global-weights-only-v1` and `step_semantics=aggregate-slot-neural-step-count-v1`. The checkpoint is serialized in the full native file layout for loader compatibility, but only global weights are persistent population state. Other neural-dynamic and embodied state is reset between episodes.

## Result boundary

Frozen evaluation disables both plasticity and task-triggered DAN stimulation. The recorded canonical initial and final evaluations both passed one gate and then collided at control step 120. Therefore this checkpoint is evidence of local-plasticity-driven stored-weight change under current semantics, **not** a claim that canonical v1 established categorical behavioral improvement, generalization, or long-term learning stability.

## Source code and reproducibility

- Project overview: https://huggingface.co/shute2004/virtual-fly
- GitHub source: https://github.com/shute2004/virtual-fly
- Canonical package: https://github.com/shute2004/virtual-fly/tree/main/canonical/canonical-v1
- Full artifact hash manifest: https://github.com/shute2004/virtual-fly/blob/main/release/huggingface/checkpoint-artifacts-v1.json

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
