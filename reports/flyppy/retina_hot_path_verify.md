# MaleCNS retinal hot-path verification

- generated_at_utc: 2026-09-15T05:44:34+00:00
- overall: PASS
- synthetic compound-eye steps: 64
- ommatidia per eye: 721
- emitted body currents: bitwise equal on every step
- diagnostics: bitwise equal on every step
- final adaptation state: bitwise equal
- scalar reference transduction wall time: 0.369638 s
- optimized transduction wall time: 0.168319 s
- transduction-only speedup: 2.196x

## Mapping multiplicity

| side | MaleCNS columns | unique FlyBody ommatidia | duplicate assignments |
|---|---:|---:|---:|
| L | 300 | 174 | 126 |
| R | 524 | 232 | 292 |

Duplicate assignments are intentionally not collapsed. They share the same per-ommatidium adaptation state and are processed in the original MaleCNS column order.
