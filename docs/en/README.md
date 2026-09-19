# Documentation — English

[English](README.md) · [日本語](../ja/README.md) · [简体中文](../zh-CN/README.md)

This directory contains the public research documentation in English. The Japanese and Simplified Chinese directories mirror the same document set and filenames.

## Start here

- [`architecture.md`](architecture.md) — current implemented architecture and scientific boundaries
- [`results.md`](results.md) — canonical v1 results and the boundary to historical results
- [`scientific-model.md`](scientific-model.md) — neural-system, plasticity, sensory, and motor modeling principles
- [`requirements.md`](requirements.md) — project requirements, non-goals, and scientific invariants

## Reproducibility and provenance

- [`data-and-reproducibility.md`](data-and-reproducibility.md) — data layers, provenance categories, large-artifact policy, and reproducibility levels
- [`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) — pre-OSS reproducibility/semantic-contract audit and fixes
- [`code-structure.md`](code-structure.md) — current module ownership and Python/Rust responsibility boundaries

## Experimental and biological boundaries

- [`experiments.md`](experiments.md) — staged experiments, controls, behavioral/neural metrics, and learning criteria
- [`wing-motor-boundary.md`](wing-motor-boundary.md) — individual wing motor-neuron to peripheral-muscle boundary
- [`references.md`](references.md) — scientific references and an evidence map from primary literature to implementation decisions

## Canonical experiment

The publication-facing reference experiment is outside `docs/` so it can remain a self-contained reproducibility package:

- [`../../canonical/canonical-v1/README.md`](../../canonical/canonical-v1/README.md)
- [`../../canonical/canonical-v1/reference-report.md`](../../canonical/canonical-v1/reference-report.md)
- [`../../canonical/canonical-v1/reference-manifest.json`](../../canonical/canonical-v1/reference-manifest.json)

## Non-public-documentation categories

Development-only notes are under [`../internal/`](../internal/README.md). Historical records are under [`../archive/`](../archive/README.md). They are intentionally not mirrored into every language.
