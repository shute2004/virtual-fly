# Documentation

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

The publication-facing documentation is English-first. Japanese and Simplified Chinese entry documents are provided for the main public surfaces. Some deeper developer/scientific notes remain in Japanese because they originated as internal research records; they are labeled by role below rather than silently treated as publication claims.

## Start here

- [`../README.md`](../README.md) — project overview, canonical result, installation and reproduction
- [`results.md`](results.md) — canonical vs. historical result boundary and claim rules
- [`architecture.md`](architecture.md) — current implemented architecture
- [`code-structure.md`](code-structure.md) — current developer-facing module ownership
- [`../canonical/canonical-v1/README.md`](../canonical/canonical-v1/README.md) — canonical v1 package
- [`release-and-zenodo.md`](release-and-zenodo.md) — GitHub Release and Zenodo publication plan

## Current scientific/developer contracts

These documents define or explain active project contracts. Some are currently Japanese-only.

- `scientific-model.md` — neural/plasticity modeling principles and claim boundaries
- `requirements.md` — project requirements and prohibited shortcuts
- `embodiment-v0.md` — embodiment and closed-loop integration notes
- `wing-motor-boundary.md` — motor-neuron-to-body boundary
- `live-observers.md` — observer/viewer non-interference contract
- `data-and-reproducibility.md` — data provenance and large-artifact policy
- `experiments.md` — experimental design and learning criteria
- `references.md` — primary references and upstream assets
- `flyppy/README.md` — current Flyppy development/runtime notes

For current production code ownership, prefer `architecture.md` + `code-structure.md` over older design prose when they differ.

## Reproducibility and provenance

- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — audit of historical provenance/semantics and the fix set
- [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json) — machine-readable canonical provenance
- [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md) — compact canonical result
- [`data-and-reproducibility.md`](data-and-reproducibility.md) — general data policy

## Historical material

Dated reviews, handoff notes, superseded designs, and one-off investigations live under `archive/YYYY-MM-DD/` or `reports/`.

They are intentionally retained for provenance. They must not be interpreted as if they were generated under the current canonical semantics unless a current document explicitly says so.

## Roadmap

`roadmap.md` is a development-planning document, not a statement that every listed biological mechanism is already implemented. For publication claims, use `README.md`, `results.md`, the canonical package, and the current code.
