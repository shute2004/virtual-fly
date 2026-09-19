# Contributing to virtual-fly

[English](CONTRIBUTING.md) · [日本語](CONTRIBUTING.ja.md) · [简体中文](CONTRIBUTING.zh-CN.md)

`virtual-fly` is a research software project. Contributions are welcome when they preserve the distinction between measured biology, literature-derived choices, inference, engineering assumptions, and calibration.

Before making a scientific/runtime change, read:

- [`README.md`](README.md)
- [`AGENTS.md`](AGENTS.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/results.md`](docs/results.md)
- [`docs/requirements.md`](docs/requirements.md)

## Branch policy

`main` is the stable/publication branch. Development work may continue on dedicated branches such as `dev/flyppy-v3`, but publication milestones should be integrated back into `main` rather than leaving the default branch stale.

A milestone is ready to integrate when:

- relevant tests pass;
- provenance/semantic contracts are documented;
- no known blocking regression is hidden;
- temporary generated artifacts are not mixed into source control;
- publication claims match the evidence actually produced by that lineage.

## Scientific invariants

Do not introduce, without an explicit project-level design change:

- backpropagation or gradient descent as the nervous-system learner;
- Q-learning, policy gradient, actor-critic, or an external learned policy;
- a hand-written Flyppy obstacle policy;
- direct scalar reward-to-weight updates;
- an external feature extractor/classifier that replaces a known local sensory circuit while only preserving its final answer.

Learning in the canonical architecture follows:

```text
sensory stimulation
    → neural dynamics
    → behavior
    → neuromodulatory stimulation
    → local plasticity
    → changed nervous system
```

## Provenance labels

Where practical, distinguish:

- `observed`
- `literature`
- `inferred`
- `assumed`
- `calibrated`

Do not present an inferred or calibrated boundary as a directly observed biological fact.

## Historical vs. canonical results

Historical v240/v960/v966/v1704 artifacts and diagnostics are retained for provenance. Do not rewrite or relabel them as current canonical results.

Publication-facing result claims must follow [`docs/results.md`](docs/results.md) and the canonical package under [`canonical/canonical-v1/`](canonical/canonical-v1/).

## Change scope

Prefer one verifiable purpose per change, for example:

- MaleCNS data semantics;
- neural state stepping;
- plasticity;
- checkpoint format;
- sensory transduction;
- motor/peripheral mapping;
- body physics;
- reproducibility/reporting.

When changing neural dynamics, plasticity, neuromodulation, sensory transduction, CNS→body mapping, reinforcement stimulation targets, or checkpoint semantics, update the relevant documentation in the same change.

## Testing

For new computational core changes, add the smallest useful combination of:

- unit tests;
- deterministic/reference tests;
- checkpoint round-trip tests;
- semantic contract tests;
- backend parity checks where relevant.

Do not optimize a backend by silently changing scientific semantics.

## Data and generated artifacts

Do not commit large external datasets or generated experiment artifacts to Git.

Examples that belong outside Git:

- raw MaleCNS downloads;
- normalized large snapshots;
- checkpoints;
- trajectories;
- rendered MP4s;
- build caches/profiling traces.

Track small manifests, hashes, configs, and reports instead. See [`docs/data-and-reproducibility.md`](docs/data-and-reproducibility.md).

## Documentation language

Publication-facing documentation is English-first. Japanese and Simplified Chinese versions are desirable for major public entry points.

Internal research/development notes may remain Japanese when that best preserves the original context. Do not mass-translate historical records merely for cosmetic consistency.

## Pull requests / commits

For scientifically meaningful changes, describe:

- what changed;
- why;
- whether the change is observed/literature/inferred/assumed/calibrated;
- which previous experiments become incompatible or historical;
- how the change was tested.

## License

By contributing to this repository, you agree that your original contributions may be distributed under the repository's [MIT License](LICENSE). Do not contribute third-party code/data/assets unless their terms permit the intended use and the necessary attribution/notice is included.
