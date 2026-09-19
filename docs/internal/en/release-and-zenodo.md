# GitHub Release and Zenodo publication plan

This document describes the publication path for the first research-OSS release. It does not create a DOI by itself.

## 1. Publication branch and tag

Before the first public archival release:

1. ensure the publication commit is integrated into the repository's default `main` branch;
2. verify the working tree used to create the release is clean;
3. verify canonical v1 files are present and unchanged;
4. run lightweight documentation/metadata validation;
5. create an annotated release tag, planned as `v0.1.0`;
6. create the GitHub Release from exactly that tag.

Do not tag an older `main` commit while the canonical/publication work exists only on a development branch.

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

The GitHub source archive already contains the canonical package, so the canonical raw source/snapshot/checkpoints do not need to be duplicated as release attachments.

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

If a future archival requirement needs a large immutable checkpoint/data bundle, publish it as a separately identified research-data record rather than silently embedding it in the software release.

## 5. Canonical reproduction statement for release notes

Recommended factual summary:

> Canonical experiment v1 is pinned to clean scientific commit `7fa464aad7269d34f46f1171080b51e095d1d811`. A full end-to-end rerun on 2026-09-19 reproduced all recorded static artifact hashes, completed six canonical training episodes from global v0 to v6, changed exactly 2,163,179 stored edges, reloaded both checkpoints successfully, and reproduced the recorded frozen initial/final outcomes. The short canonical run did not show categorical behavioral improvement.

Do not shorten this to "the fly learned Flyppy" or similar.

## 6. Zenodo GitHub integration

Current Zenodo documentation describes the GitHub integration as follows:

1. connect the GitHub account to Zenodo;
2. enable the repository in Zenodo's GitHub integration;
3. create a GitHub Release;
4. let Zenodo ingest/archive that release and mint the record DOI.

Official documentation:

- https://help.zenodo.org/docs/github/
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/

Zenodo supports both `CITATION.cff` and `.zenodo.json`. When both files are present, Zenodo uses `.zenodo.json` for GitHub-release archival metadata; `CITATION.cff` remains useful for GitHub's **Cite this repository** UI. Therefore the two files in this repository should remain semantically synchronized.

Official metadata documentation:

- https://help.zenodo.org/docs/github/describe-software/
- https://help.zenodo.org/docs/github/describe-software/zenodo-json/

## 7. DOI handling

Do not invent a DOI in README or `CITATION.cff` before one exists.

For the normal GitHub integration path, create the GitHub Release and allow Zenodo to mint the DOI. After the DOI exists:

1. verify the Zenodo metadata and archived files;
2. update `CITATION.cff` with the release version, date, and DOI if desired;
3. add a DOI badge/link to README;
4. make any metadata-only follow-up release/version decision explicitly rather than rewriting the already archived release.

If a DOI must be known before publication, use a manual Zenodo deposition and reserve a DOI there instead of pretending the GitHub integration has already assigned one.

## 8. Creator metadata

For `v0.1.0`, the creator identity is intentionally the established Git/GitHub project identity `shute2004`. No legal name, ORCID, or affiliation is inferred from local machine/account information. Those optional fields may be added before the DOI-bearing release only if the project owner explicitly wants them in the permanent citation record.

`CITATION.cff` and `.zenodo.json` are kept semantically synchronized for title, version, license, creator identity, repository identity, and project description. Because both files are present, current Zenodo GitHub integration uses `.zenodo.json` for archival metadata; `CITATION.cff` remains for GitHub and other citation tooling.

## 9. Pre-publication privacy/provenance audit

The 2026-09-19 tracked-file scan found no high-confidence credential-shaped strings and no personal email addresses in tracked files. It found absolute local filesystem paths in exactly 14 historical provenance files: one archived development note and 13 tracked reports/logs/JSON records.

The decision for `v0.1.0` is to preserve those 14 originals unchanged. Their paths record the machine context of historical runs, tracebacks, or generated reports; rewriting them would alter the original provenance record. Current public READMEs and current multilingual documentation contain no `/Users/<local-user>/...` path. [`../../../reports/README.md`](../../../reports/README.md) and [`../../archive/README.md`](../../archive/README.md) explicitly explain that such historical paths are provenance rather than installation instructions.

## 10. Third-party fresh-clone preflight

A fresh shallow clone of `main` at `b1245a7022d6460a9046ec6cf7af18bb09818868` was tested on 2026-09-19 without using the development checkout's virtual environment, Cargo build directory, MaleCNS snapshot, or generated artifacts.

Verified from README-level instructions:

- `uv sync --frozen` completed successfully;
- Python standard-library discovery ran 103 tests: 103 passed;
- `cargo test --workspace` ran 25 Rust tests: 25 passed;
- `bash -n canonical/canonical-v1/reproduce.sh` passed;
- the canonical README/config/reference manifest/reference report entry files were present.

The full canonical experiment was deliberately not rerun as part of this publication preflight.

## 11. Local release rehearsal

A release candidate bundle was assembled locally without creating a Git tag or GitHub Release. The rehearsal generated a source archive from the clean publication candidate, copied the four planned historical attachments under their release filenames, and verified every attachment size and SHA-256 against `release/release-assets-v0.1.0.json`.

The source archive contained the README, license, citation/Zenodo metadata, canonical reproducer package, multilingual documentation, and prepared release notes. It contained no `.git`, `.venv`, Cargo `target`, or local `artifacts/` directory. The rehearsal passed.

A read-only GitHub check on 2026-09-19 confirmed that the repository remained **private**, `main` was the default branch, and no tags or GitHub Releases existed. No publication action was performed by this preflight.

## 12. Release checklist

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
- [x] creator identity for `v0.1.0` intentionally set to `shute2004`; ORCID/affiliation omitted unless explicitly supplied
- [ ] tag `v0.1.0`
- [ ] GitHub Release created
- [ ] Zenodo repository integration enabled
- [ ] Zenodo record/DOI verified
- [ ] citation metadata updated with real DOI only after minting
