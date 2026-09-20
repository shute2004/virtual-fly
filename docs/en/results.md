# Results and provenance boundaries

[English](results.md) · [日本語](../ja/results.md) · [简体中文](../zh-CN/results.md)

This document defines which results may be described as current canonical results and which are retained only as historical development evidence.

## 1. Canonical experiment v1

Canonical v1 is the publication-facing reference experiment. The recorded experiment was actually run from clean Git commit `7fa464aad7269d34f46f1171080b51e095d1d811`. After a pre-publication privacy-only history rewrite, the content-equivalent public scientific commit is:

```text
f9c86c904d67ff974f3c43d37aab3619bc93fc1b
```

The rewrite changed only the local OS username inside 14 historical report/provenance files at the canonical commit; executable source, canonical conditions, and scientific artifact-generation logic are unchanged. The mapping is recorded in [`../../canonical/canonical-v1/privacy-redaction-provenance.json`](../../canonical/canonical-v1/privacy-redaction-provenance.json).

The compact result is in [`../../canonical/canonical-v1/reference-report.md`](../../canonical/canonical-v1/reference-report.md); machine-readable reference provenance is in [`../../canonical/canonical-v1/reference-manifest.json`](../../canonical/canonical-v1/reference-manifest.json). Its `derived_generation.commands` entries are summaries; [`../../canonical/canonical-v1/reproduce.sh`](../../canonical/canonical-v1/reproduce.sh) is the authoritative exact invocation source.

### Conditions

- fresh official MaleCNS v1.0 source acquisition;
- fresh current-semantics MaleCNS snapshot;
- 166,700 neurons and 25,582,938 directed edges;
- class-DAN semantics: released `class=DAN` annotation plus dopamine consensus;
- body v7 / environment v7;
- direct-ray vision, K=13 rays/ommatidium;
- 97-neuron haltere timing subset, gain 0.05, `interaction-load-v2`;
- population 2, asynchronous shared weights;
- boundary-band curriculum;
- 6 training episodes;
- global-weights-only checkpoint semantics.

### Observed canonical result

- global weight version: v0 → v6;
- aggregate neural step: 484;
- exactly changed stored edges: **2,163,179 / 25,582,938**;
- edges with `|Δw| > 1e-7`: 1,461,896;
- edges with `|Δw| > 1e-6`: 536,060;
- strengthened edges: 961,611;
- weakened edges: 1,201,568;
- maximum `|Δw|`: 0.0018288875.

Frozen evaluation disables plasticity and task-triggered DAN currents. Both the initial v0 state and final v6 state passed one gate and then collided with the next gate at control step 120.

**Interpretation:** canonical v1 demonstrates a traceable closed-loop run in which the current local plasticity implementation changed stored MaleCNS weights. The short canonical experiment did **not** show categorical behavioral improvement. It does not establish generalization or long-term learning stability.

## 2. End-to-end reproduction verification

On 2026-09-19, `canonical/canonical-v1/reproduce.sh` was run from beginning to end using an isolated clean checkout of the original experiment commit `7fa464a`. This verification predates the privacy rewrite and is not represented as a run at `f9c86c9`. Because the public-equivalent commit changes no executable/configuration/scientific-generation logic, full training was not rerun solely for the privacy redaction.

On 2026-09-20, the static reconstruction path was rerun from the public-equivalent `f9c86c9` code using fresh official MaleCNS source data. The semantic snapshot hash and all binary static artifacts matched the original reference. Every JSON artifact that differed in raw SHA-256 became byte-identical to the original reference after normalizing only the rewritten Git identity and the chained hashes caused by that identity change. This is the validation rule now implemented by the public reproducer; no training was started for this check.

### Full end-to-end verification — original experiment commit `7fa464a` — 2026-09-19

Verified:

- fresh MaleCNS source download;
- fresh snapshot construction;
- fresh derived artifact construction;
- all recorded static artifact hashes matched byte-for-byte;
- neural calibration selected synapse scale 0.005 from stable scales 0.004–0.007;
- canonical six-episode training completed;
- global v0 → v6;
- exactly 2,163,179 stored edges changed, matching the reference;
- initial/final checkpoints reloaded successfully in new bridge processes;
- frozen initial/final evaluations both matched the reference outcome;
- final pinned scientific worktree remained clean.

### Public-equivalent static verification — `f9c86c9` — 2026-09-20

Verified without starting training:

- fresh official MaleCNS source download;
- fresh snapshot and derived-artifact construction;
- semantic snapshot hash matched the reference exactly;
- binary static artifacts matched the reference byte-for-byte;
- provenance-bearing JSON matched the reference after normalizing only the privacy-rewrite Git identity and the chained hashes caused by that identity change.

The large verification bundle was temporary. Lightweight manifests and reports were retained outside the Git repository after validation. The reproducibility script now writes large verification runs outside the repository by default.

## 3. Historical development results

Results involving v240, v960, v966, v1704, and related posting/diagnostic runs are retained as **historical results**.

They are scientifically useful for understanding the development process, but they are not interchangeable with canonical v1. The provenance audit in [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) records important lineage differences, including older dopamine-source semantics, pre-commit provenance, mixed v966 lineage, and evaluation conditions that differ from the current frozen canonical comparison.

Do not describe historical behavior as if it had been generated by the current class-DAN canonical semantics.

## 4. Historical Before/After media

The publication-oriented historical comparison uses:

- **before:** global weight version v240;
- **after:** global weight version v966;
- body/environment: v7/v7;
- gate 2 center: 13.703125 mm;
- evaluation plasticity: off;
- task-triggered PAM stimulation during playback: on;
- held-out evaluation: no;
- before outcome: 1 gate;
- after outcome: 2 gates.

The Before state is **not an untrained MaleCNS initial state**: v240 had already undergone 240 historical training episodes. The lineage through v960 used the older dopamine-source definition, and v966 combines conditions from more than one historical stage. This media is therefore useful as a visualization of a historical learned-state comparison, not as the canonical behavioral result or as evidence for a simple untrained-to-trained improvement.

Three separate video renderings are associated with this comparison: the video posted to X, the higher-resolution source used to generate the README GIF, and the GitHub Release MP4. They share the historical v240/v966 comparison but are not identical files. The Git repository tracks the README GIF and a manifest that records all three video roles and hashes; the MP4s and raw playback JSON remain outside Git. See [`../../release/release-assets-v0.1.0.json`](../../release/release-assets-v0.1.0.json).

## 5. Claiming rules for README, releases, papers, and posts

Safe claims supported by canonical v1 include:

- the released MaleCNS connectome is used as the initial neural structure;
- closed-loop visual/mechanosensory input reaches the neural runtime;
- individual released motor-neuron outputs drive the physical body through a peripheral model;
- task outcomes stimulate selected real DAN populations rather than directly supplying a scalar weight-update target;
- the current local plasticity implementation changes stored CNS weights;
- the complete canonical data/provenance path can be reproduced end-to-end.

Claims **not** supported by canonical v1 include:

- "canonical v1 learned to solve Flyppy";
- "canonical v1 improved behavior";
- "the simulator reproduces a complete real fly";
- "historical v966 behavior was reproduced under current canonical semantics";
- "the current model establishes general learning or long-term stability".

When showing historical media, label it as historical and link back to this document or the canonical reference report.
