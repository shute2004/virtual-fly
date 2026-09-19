# Reports directory

`reports/` contains small tracked development reports, diagnostics, benchmarks, and historical experiment summaries.

These files are retained for engineering/scientific provenance. They are **not automatically publication claims** and should not be treated as equivalent to the canonical experiment simply because they are tracked in Git.

For publication-facing results, start with:

- [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md)
- [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json)
- [`../docs/en/results.md`](../docs/en/results.md)

Files named `latest.*`, continuously updated development summaries, one-off diagnostics, and historical fixed evaluations may describe older or mixed lineages. Their original context should be preserved rather than rewritten to match current semantics.

Some tracked historical reports intentionally retain absolute local filesystem paths such as the machine path from which a run or traceback was recorded. These paths are provenance from the original execution, not installation instructions and not required by the current code. They are preserved instead of being silently rewritten; current public documentation and commands use repository-relative or portable paths.

Large checkpoints, trajectories, videos, raw source data, and temporary reproduction bundles belong under gitignored local `artifacts/` or external archival storage, not under `reports/`.
