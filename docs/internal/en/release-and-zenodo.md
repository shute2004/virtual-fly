# GitHub Release record and optional future Zenodo archival

This document records the publication path used for the first research-OSS release. The initial v0.1.0 publication uses GitHub and Hugging Face; Zenodo/DOI is deliberately deferred and is not part of the initial release checklist.

## 1. Publication branch and tag

For the first public archival release, the publication commit was integrated into `main`, the release worktree and canonical v1 package were checked, lightweight documentation/metadata validation was run, and annotated tag `v0.1.0` was used for the GitHub Release.

Do not tag an older `main` commit while the canonical/publication work exists only on a development branch.

The long-lived development checkout may contain unrelated modified historical reports or untracked diagnostics. Publication-day edits, tagging, and release assembly must therefore be done from a clean isolated worktree or dedicated clean release checkout. Do not use broad staging such as `git add -A` from a dirty development checkout.

## 2. What belongs in Git

Keep in the source repository:

- source code;
- tests;
- public documentation;
- `LICENSE` and `THIRD_PARTY_NOTICES.md`;
- `CITATION.cff` and `.zenodo.json`;
- canonical v1 config/reference manifest/reference report/reproducer;
- small representative figures;
- small historical/provenance reports;
- release asset manifests and hashes.

Do not add large raw scientific artifacts to Git merely to make a release self-contained.

## 3. GitHub Release assets

The GitHub source archive already contains the canonical package, so canonical raw source/snapshots/checkpoints are not duplicated as GitHub Release attachments. Checkpoint distribution is handled separately through Hugging Face as described below.

For `v0.1.0`, attach the historical presentation media only if it remains useful for communicating the project:

- `historical-before-after-v240-v966.mp4`
- `historical-before-after-v240-v966-metadata.json`
- `historical-before-v240-playback.json`
- `historical-after-v966-playback.json`

These assets are **historical**, not canonical results. Their exact local source paths, sizes, and SHA-256 hashes are recorded in [`../../../release/release-assets-v0.1.0.json`](../../../release/release-assets-v0.1.0.json).

The release notes must say explicitly that canonical v1 did not show categorical behavioral improvement and that the v240→v966 media belongs to an older development lineage.

## 4. What should not be attached by default

Do not attach these simply for completeness:

- the ~1 GB raw MaleCNS download;
- normalized MaleCNS snapshots;
- large training checkpoints;
- full development trajectory directories;
- `.venv`, Cargo `target`, or build caches;
- temporary end-to-end reproduction bundles.

The canonical reproducer can reacquire and validate source data. Keeping these large files out of both Git and routine releases reduces duplication while preserving scientific provenance through hashes and manifests.

If a future archival requirement needs a large immutable non-checkpoint data bundle, publish it as a separately identified research-data record rather than silently embedding it in the software release.

## 5. Hugging Face checkpoint distribution

Checkpoint payloads are kept separate from GitHub. The Hugging Face project overview repository is:

- https://huggingface.co/shute2004/virtual-fly

The four checkpoint repositories are:

- https://huggingface.co/shute2004/virtual-fly-initial-20260920 — canonical v1 initial checkpoint;
- https://huggingface.co/shute2004/virtual-fly-trained-20260920 — canonical v1 trained checkpoint;
- https://huggingface.co/shute2004/virtual-fly-video-before — exact historical checkpoint used for the Before side of the published comparison video;
- https://huggingface.co/shute2004/virtual-fly-video-after — exact historical checkpoint used for the After side of the published comparison video.

The canonical repository date is 2026-09-20. The historical video repositories are never described as the canonical initial/trained pair. Internal versions v240/v966 remain provenance fields only.

Each checkpoint repository contains the native checkpoint files plus a model card, attribution notice, machine-readable provenance, and SHA-256 list. Full trajectories, frame directories, raw MaleCNS data, normalized snapshots, FlyBody assets, unrelated development checkpoints, and build artifacts are excluded unless they are directly necessary to use the checkpoint.

The official MaleCNS download page licenses `male-cns:v1.0` under CC BY 4.0. Because the published weights are derived numerical states based on that connectome, the published checkpoint repositories use `cc-by-4.0` metadata and explicit MaleCNS attribution. Raw MaleCNS is not mirrored; users obtain it from the official source or the canonical acquisition/reproduction path. FlyBody, FlyGym, and MuJoCo assets are likewise not copied into checkpoint repositories.

The published layout, model-card sources, exact checkpoint file sizes/hashes, and publication record are tracked in [`../../../release/huggingface/`](../../../release/huggingface/README.md).

The four repositories were created privately on 2026-09-20, uploaded, and checked against the recorded checkpoint hashes before public release.

## 6. Canonical reproduction statement for release notes

Recommended factual summary:

> Canonical experiment v1 was actually run from clean scientific commit `7fa464aad7269d34f46f1171080b51e095d1d811`. A full end-to-end rerun at that original commit on 2026-09-19 reproduced all recorded static artifact hashes, completed six canonical training episodes from global v0 to v6, changed exactly 2,163,179 stored edges, reloaded both checkpoints successfully, and reproduced the recorded frozen initial/final outcomes. Before public release, a privacy-only history rewrite replaced the local OS username in historical absolute paths; the corresponding public scientific commit is `f9c86c904d67ff974f3c43d37aab3619bc93fc1b`. No executable source, canonical condition, or scientific artifact-generation logic changed, so full training was not rerun solely for the privacy rewrite. The short canonical run did not show categorical behavioral improvement.

Do not shorten this to "the fly learned Flyppy" or similar.

## 7. Optional future Zenodo archival

Zenodo is **not used for the initial v0.1.0 publication**. The repository keeps `.zenodo.json` only as prepared metadata for a possible future fixed scholarly archive; its presence does not mean that a Zenodo record or DOI is planned for this release.

If a future project decision requires a fixed archival record, re-check the then-current Zenodo workflow before acting. At that point, keep `CITATION.cff` and `.zenodo.json` semantically synchronized and add a DOI to public metadata only after a real DOI exists.

Current documentation retained for that future decision:

- https://help.zenodo.org/docs/github/
- https://help.zenodo.org/docs/github/describe-software/
- https://help.zenodo.org/docs/github/describe-software/zenodo-json/

## 8. DOI handling

There is no DOI for v0.1.0 and none should be implied in README, `CITATION.cff`, release notes, or badges. DOI work is outside the initial publication procedure. If a fixed scholarly archive is created later, record the real DOI only after the archival service has actually assigned it.

## 9. Creator metadata

For `v0.1.0`, the creator identity is intentionally the established Git/GitHub project identity `shute2004`. No legal name, ORCID, or affiliation is inferred from local machine/account information. Those optional fields may be added before any future DOI-bearing archive only if the project owner explicitly wants them in the permanent citation record.

`CITATION.cff` and `.zenodo.json` are kept semantically synchronized for title, version, license, creator identity, repository identity, and project description. `CITATION.cff` is the active citation metadata for the initial release; `.zenodo.json` is retained only as optional future archival metadata and its handling must be revalidated against Zenodo's current behavior if archival is later enabled.

## 10. Pre-publication privacy/provenance audit

The 2026-09-19 tracked-file scan found no high-confidence credential-shaped strings and no personal email addresses in tracked files. It found the local OS username in absolute filesystem paths in exactly 14 files at the canonical experiment commit: one archived development note and 13 tracked reports/logs/JSON records. Across the later pre-publication history, the same string appeared only in those historical paths (including the pre-archive location of the note) plus this internal audit document.

Before `v0.1.0`, the Git history was rewritten so the username component is represented as `<local-user>`. At the canonical commit, an exact tree comparison between original `7fa464aad7269d34f46f1171080b51e095d1d811` and public equivalent `f9c86c904d67ff974f3c43d37aab3619bc93fc1b` found exactly 14 changed paths, and every changed blob was byte-for-byte equal to the original after only the username substitution. No source-code, canonical configuration, checkpoint, scientific artifact, result value, commit message, author identity, or experiment condition changed. The original→public mapping is recorded in [`../../../canonical/canonical-v1/privacy-redaction-provenance.json`](../../../canonical/canonical-v1/privacy-redaction-provenance.json).

## 11. Third-party fresh-clone preflight

A fresh shallow clone of `main` at pre-rewrite commit `b1245a7022d6460a9046ec6cf7af18bb09818868` was tested on 2026-09-19 without using the development checkout's virtual environment, Cargo build directory, MaleCNS snapshot, or generated artifacts. Its privacy-redacted public equivalent is `54a414da3e50e7c325747c9d580dc4a086a4a7ab`; the test itself was performed before the rewrite.

Verified from README-level instructions:

- `uv sync --frozen` completed successfully;
- Python standard-library discovery ran 103 tests: 103 passed;
- `cargo test --workspace` ran 25 Rust tests: 25 passed;
- `bash -n canonical/canonical-v1/reproduce.sh` passed;
- the canonical README/config/reference manifest/reference report entry files were present.

The full canonical experiment was deliberately not rerun as part of this publication preflight.

## 12. Local release rehearsal

A release candidate bundle was assembled locally without creating a Git tag or GitHub Release. The rehearsal generated a source archive from the clean publication candidate, copied the four planned historical attachments under their release filenames, and verified every attachment size and SHA-256 against `release/release-assets-v0.1.0.json`.

The source archive contained the README, license, citation and optional archival metadata, canonical reproducer package, multilingual documentation, and prepared release notes. It contained no `.git`, `.venv`, Cargo `target`, or local `artifacts/` directory. The rehearsal passed.

A read-only GitHub check on 2026-09-19 confirmed that the repository remained **private**, `main` was the default branch, and no tags or GitHub Releases existed. No publication action was performed by this preflight.

### Hugging Face checkpoint rehearsal

The four prepared Hugging Face repository templates were also assembled locally with the exact source checkpoint directories, without creating or uploading any Hugging Face repository. Every file listed in each prepared `SHA256SUMS` was rehashed from the assembled repository and matched the recorded value. The resulting checkpoint payload sizes were:

- canonical initial: 207,998,044 bytes;
- canonical trained: 207,998,046 bytes;
- historical video Before: 207,997,939 bytes;
- historical video After: 207,997,940 bytes.

The four checkpoint repositories were subsequently created privately, uploaded, and remotely verified against the recorded file sizes and SHA-256 values before public release.

## 13. Release checklist

- [x] publication changes are on default `main`
- [ ] repository visibility is intentionally set for public release
- [x] historical absolute local paths deliberately accepted as preserved provenance, with public explanation
- [x] third-party fresh-clone install/basic validation passed
- [x] clean release checkout validated for the publication candidate
- [x] documentation links / JSON / CFF / package metadata validated
- [x] canonical scientific reference files unchanged
- [x] canonical vs. historical distinction visible in README and release notes
- [x] release assets recorded with SHA-256 hashes and reverified locally
- [x] local release-bundle rehearsal passed without creating a tag or Release
- [x] third-party attribution included
- [x] Hugging Face four-repository publication layout, model-card templates, exact checkpoint hashes, and required file lists prepared
- [x] local four-repository Hugging Face bundle/hash rehearsal passed without upload
- [x] canonical/Historical-video checkpoint lineages explicitly separated in cards and GitHub docs
- [x] raw MaleCNS and FlyBody/FlyGym/MuJoCo assets excluded from checkpoint upload plan; official acquisition paths retained
- [x] creator identity for `v0.1.0` intentionally set to `shute2004`; ORCID/affiliation omitted unless explicitly supplied
- [x] pre-publication history privacy rewrite completed; local OS username removed from all reachable public history and canonical old→public SHA mapping recorded
- [x] choose actual publication date and finalize the two canonical Hugging Face repository names
- [x] create/upload the four Hugging Face checkpoint repositories
- [x] publish/update `shute2004/virtual-fly` overview card with final four links
- [x] replace placeholder canonical Hugging Face names/links in GitHub docs after the repositories exist
- [ ] tag `v0.1.0`
- [ ] GitHub Release created
- [x] Zenodo/DOI explicitly excluded from the initial v0.1.0 publication procedure
- [ ] optional future Zenodo archival only if a later project decision requires a fixed scholarly record
