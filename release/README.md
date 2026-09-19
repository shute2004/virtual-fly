# Release metadata

This directory tracks **small release manifests only**.

Large binaries and raw scientific artifacts must not be committed here. Historical presentation media may be assembled outside Git and verified before a GitHub Release; large checkpoint artifacts are prepared for separate Hugging Face repositories instead of GitHub Release attachment.

Current plan:

- [`v0.1.0-notes.md`](v0.1.0-notes.md) — prepared GitHub Release notes
- [`release-assets-v0.1.0.json`](release-assets-v0.1.0.json) — exact historical media/playback assets planned for the first public GitHub Release
- [`huggingface/`](huggingface/README.md) — Hugging Face overview/model-card templates, checkpoint hashes, provenance, and upload layout
- [`../docs/internal/en/release-and-zenodo.md`](../docs/internal/en/release-and-zenodo.md) — publication, preflight, Hugging Face, and DOI workflow

Canonical v1 itself is already represented by tracked config/provenance/reproduction files under [`../canonical/canonical-v1/`](../canonical/canonical-v1/). Do not duplicate raw source/snapshots/checkpoints into Git; the canonical initial/trained checkpoints are prepared as separate dated Hugging Face repositories.
