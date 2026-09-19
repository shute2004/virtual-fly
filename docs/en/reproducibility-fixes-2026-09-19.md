# Reproducibility and semantic-contract fixes — 2026-09-19

[English](reproducibility-fixes-2026-09-19.md) · [日本語](../ja/reproducibility-fixes-2026-09-19.md) · [简体中文](../zh-CN/reproducibility-fixes-2026-09-19.md)

This change set addresses forward-going reproducibility and semantic-contract failures found during the pre-OSS audit. It intentionally does **not** rewrite historical checkpoints or claim that old artifacts were generated under current semantics.

## Scientific-semantics boundary

This change set does not change:

- MaleCNS topology or released connection counts;
- neural activity dynamics or neurotransmitter fast-sign assumptions;
- the local three-factor plasticity equation, learning rate, eligibility dynamics, or weight commit equation;
- gate-pass PAM / collision PPL stimulation behavior;
- individual motor-neuron boundary or peripheral motor mapping;
- retinal transduction semantics;
- haltere physical transduction equations;
- shared-weight additive+clamp rebase semantics.

The changes are validation, provenance, configuration propagation, reporting, and checkpoint-contract changes only.

## 1. MaleCNS class-DAN snapshot contract

**Problem.** A pre-class-DAN MaleCNS snapshot could be silently loaded by current code. Missing `modulator_roles_file` fell back to `consensus neurotransmitter == dopamine`, restoring the old 392-source semantics.

**Fix.** Current production MaleCNS snapshots must declare `runtime_semantics=male-cns-v1-class-dan-v1`, the exact class-DAN role definition, a modulator-role file, source hashes, generated-file hashes, and generator Git provenance. Rust and Python production entry points fail closed when this contract is absent. Explicit historical replay remains possible only through `VF_ALLOW_LEGACY_MALECNS_SNAPSHOT=1` at the Rust snapshot boundary.

**Semantic effect.** No DAN identity was changed relative to the already-adopted class-DAN rule. The fix prevents accidental regression to the older rule.

**Verification.** Python reproducibility tests cover rejection of old semantics and acceptance of a correctly hashed class-DAN snapshot; `vf-neural` tests cover normal snapshot/runtime behavior.

## 2. Haltere full-map versus timing-subset identity

**Problem.** `haltere-campaniform-sensory-v1` (195 candidates) and `haltere-timing-afferents-v1` (97 inferred timing afferents) had compatible JSON shapes. A missing timing map could be generated with the full-map generator and silently used under a timing-map filename.

**Fix.** Both artifacts now carry distinct semantic `kind` values and generator/source provenance. Launchers, input preparation, child workers, and the runtime sensor require the requested kind and expected count. Timing-map preparation invokes the timing-map generator and requires a validated full map as its source. Map provenance is tied to the current MaleCNS snapshot.

**Semantic effect.** The 195-cell and 97-cell definitions are unchanged. The fix only prevents substitution between them.

**Verification.** Unit tests assert that the two kinds are not interchangeable. Existing runtime transduction equations are untouched.

## 3. FlyBody v7/v8 neutral-trim reproducibility

**Problem.** Body v7/v8 required `wing-pattern-neutral-trim-v1.npy`, but a fresh checkout did not reliably generate it. Existing trim artifacts also lacked enough provenance to distinguish current generated output from historical local artifacts.

**Fix.** The production launcher generates the trim when both trim files are absent, then validates metadata semantics, pattern hash, and generator provenance. The body v7 adapter validates the trim before use. Historical trim replay requires explicit `VF_ALLOW_LEGACY_FLYBODY_ARTIFACTS=1`.

**Semantic effect.** Neutral-trim coefficients and the v7 lineage (`v4 + neutral trim`) are unchanged.

**Verification.** Python compile/tests and launcher syntax checks cover the new contract; the generator still uses the pre-existing fixed coefficients.

## 4. Run provenance

**Problem.** Experiment summaries did not record enough information to reconstruct the exact code and generated inputs used by a run.

**Fix.** New population runs write a per-run provenance JSON and embed the same record in the summary. It records at least:

- Git SHA, dirty state, and a hash of dirty status;
- `uv.lock` / `Cargo.lock` hashes;
- MaleCNS snapshot semantic identity and hashes;
- hashes of derived maps and calibration artifacts used by the run;
- neural-runtime, plasticity, DAN/snapshot, population-checkpoint, and neural-step semantics;
- body/environment versions and relevant body parameters;
- vision mode, ray count, and photoreceptor gain;
- haltere kind, gain, and transduction;
- reinforcement currents, duration, and gate-miss grading parameters;
- RNG seed and fixed course seed;
- population launch/curriculum/commit conditions.

Packed-process execution patches the recorded runtime/vision fields after the worker reports its resolved mode.

**Semantic effect.** Metadata only.

**Verification.** Run-history tests assert preservation of the new fields; all existing training tests remain green.

## 5. Derived-artifact provenance

**Problem.** Runtime-affecting derived artifacts could be reused solely because a path existed, even if generated by different code or from another snapshot.

**Fix.** Current generators record generator Git provenance and source snapshot hashes. Production input preparation validates provenance for embodiment groups, retina map, wing/body motor maps, haltere maps, viewer graph, neural calibration, and neutral trim as applicable. Stale artifacts fail closed instead of silently becoming current artifacts.

**Semantic effect.** Generated scientific content is not changed by the validation layer.

**Verification.** Provenance validators are covered by unit tests; generator scripts pass Python compilation.

## 6. Population checkpoint contract

**Problem.** Population checkpoints used the full `NeuralState` file layout, but resume restored only global weights. `checkpoint_neural_step` was also easy to interpret as one organism's elapsed neural time even though it is an aggregate slot-neural-step counter.

**Fix.** New population checkpoint manifests explicitly declare:

- `checkpoint_semantics=global-weights-only-v1`;
- `step_semantics=aggregate-slot-neural-step-count-v1`.

Population state and summaries repeat this contract and describe the reset-on-resume behavior. Python resume and both Rust population bridges reject checkpoints without the explicit contract unless historical replay is deliberately enabled with `VF_ALLOW_LEGACY_POPULATION_CHECKPOINT=1`.

**Semantic effect.** Resume behavior remains weights-only; it is now explicit and fail-closed rather than implicit.

**Verification.** Python checkpoint tests and Rust checkpoint tests cover the contract and rejection path.

## 7. Body/environment/sensory compatibility visibility

**Problem.** Body and environment versions are independent axes, but arbitrary mixed-generation combinations could be launched without an explicit trace in provenance.

**Fix.** Version axes and sensory conditions are recorded separately. Mixed body/environment generations produce compatibility warnings rather than being silently treated as one version. This is a warning, not a prohibition, because historical development intentionally used mixed combinations.

**Semantic effect.** No combinations are redefined or forbidden.

**Verification.** Conditions are emitted through the same tested provenance path.

## 8. Process-worker configuration propagation

**Problem.** The process worker silently dropped non-default `neutral_trim_strength`, `steering_tau_ms`, and `steering_spike_increment` values supplied by its parent.

**Fix.** These fields, along with snapshot and haltere-kind identity, are propagated into the child worker's argument namespace.

**Semantic effect.** Default production behavior is unchanged. Non-default experiments now execute the values they requested.

**Verification.** A dedicated unit test supplies non-default values and asserts exact propagation.

## 9. Gate2-height report semantics

**Problem.** `latest.md` could describe a gate2-height run using stale adaptive/boundary-band language and stale boundary-band state inherited in the curriculum JSON.

**Fix.** Report rendering branches on `curriculum_mode`. Gate2-height gets its own criterion/state section and does not render adaptive or boundary-band claims as current behavior.

**Semantic effect.** Reporting only.

**Verification.** A dedicated report test asserts absence of adaptive/boundary-band claims for gate2-height summaries.

## 10. Run-history schema

**Problem.** `history.csv` omitted body version, haltere conditions, neural/snapshot semantics, code identity, dependency identity, and checkpoint semantics. Different experiments could therefore collapse into an insufficiently discriminating run key.

**Fix.** New history rows include these fields and the run key includes the principal semantic/version axes. Historical rows remain blank where their original summaries did not contain the information; they are not backfilled by inference.

**Semantic effect.** Reporting/provenance only.

**Verification.** Run-history unit tests cover semantic fields and key disambiguation.

## 11. Posting capture defaults and provenance

**Problem.** The generic capture script defaulted to the unnatural combination `body=v3`, `environment=v7`, `haltere=0`, making accidental non-reproduction of the posting comparison easy.

**Fix.** Body and environment versions are now required arguments. The capture also validates snapshot/haltere semantic identity and records code, dependency, vision, body/environment, haltere, and runtime semantic provenance.

**Semantic effect.** Existing posting capture launcher explicitly supplies v7/v7/timing-haltere conditions, so its intended behavior is unchanged.

**Verification.** Python compilation and shell syntax checks cover the changed CLI/launcher integration.

## Historical results deliberately not rewritten

This fix set does not alter or relabel the historical facts established by the audit, including pre-commit v240/v960 provenance, old dopamine-source learning through v960, mixed v966 lineage, v240 already having 240 training episodes, non-held-out v240/v966 evaluation, evaluation-time PAM stimulation, graded PPL task-error magnitude, historical mixed v3 lineage, async-versus-serial algorithmic differences, or the actual v7 inheritance tree.
