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

# virtual-fly canonical initial checkpoint

> **Lineage:** Canonical v1 initial state

This is the initial checkpoint of the current `virtual-fly` canonical experiment. **Initial does not mean a randomly initialized artificial neural network.** It means the initial neural checkpoint used by canonical v1 after constructing the simulation state from the released MaleCNS connectome and the canonical runtime assumptions.

This checkpoint was produced by the recorded canonical v1 experiment at original Git commit `7fa464aad7269d34f46f1171080b51e095d1d811`. Its privacy-redacted public scientific equivalent is `f9c86c904d67ff974f3c43d37aab3619bc93fc1b`; the rewrite changed only historical local-path usernames, not executable source or canonical conditions. This checkpoint is global weight version **0** at aggregate neural step **0**.

## Checkpoint semantics

The manifest declares `checkpoint_semantics=global-weights-only-v1` and `step_semantics=aggregate-slot-neural-step-count-v1`. The on-disk layout contains the full set of checkpoint arrays for loader compatibility, but canonical population training persists only the **global synaptic weights** across episodes. Membrane, spike, refractory, activity-trace, modulation, eligibility, body, and peripheral state are reset between episodes. Do not interpret the directory as a continuously persistent whole-fly biological state.

## Relationship to the trained checkpoint

The corresponding canonical trained checkpoint will be published as `shute2004/virtual-fly-trained-YYYYMMDD` using the same actual publication date. The short canonical run changed exactly **2,163,179** stored edges. Its frozen initial and final evaluations had the same categorical outcome, so the pair demonstrates current-semantics weight change rather than established behavioral improvement.

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
