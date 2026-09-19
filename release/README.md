# Release metadata

This directory tracks **small release manifests only**.

Large binaries and raw scientific artifacts must not be committed here. Assemble them outside the Git repository and verify them against the recorded SHA-256 values before attaching them to a GitHub Release.

Current plan:

- [`release-assets-v0.1.0.json`](release-assets-v0.1.0.json) — exact historical media/playback assets planned for the first public release
- [`../docs/internal/en/release-and-zenodo.md`](../docs/internal/en/release-and-zenodo.md) — publication and DOI workflow

Canonical v1 itself is already represented by tracked config/provenance/reproduction files under [`../canonical/canonical-v1/`](../canonical/canonical-v1/); do not duplicate its raw source, snapshots, or checkpoints into Git release metadata.
