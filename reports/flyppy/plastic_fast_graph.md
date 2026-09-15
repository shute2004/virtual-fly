# MaleCNS PlasticFastGraph structural analysis

- generated_at_utc: 2026-09-15T03:52:47+00:00
- dataset: `male-cns:v1.0`
- mutates training/checkpoint state: false
- contract: current bootstrap dopamine/modulation + fast-transmitter plasticity rule

## Structural result

| metric | count | fraction |
|---|---:|---:|
| neurons N | 166,700 | 100% |
| all edges E | 25,582,938 | 100% |
| fast-transmitter edges | 24,559,135 | 95.9981% of E |
| dopamine incoming edges | 241,694 | 0.9447% of E |
| dopamine-capable posts S_D | 39,845 | 23.9022% of N |
| PlasticFastGraph edges P | **10,871,322** | **42.4944% of E** |
| non-plastic edges E-P | 14,711,616 | 57.5056% of E |

Under the current rule, only P edges can ever acquire a non-zero weight delta. Eligibility on E-P cannot affect any future weight because their postsynaptic modulation is structurally zero.

## Plastic incoming degree on S_D

- mean: 272.840
- p50: 206.0
- p90: 495.0
- p99: 1306.6
- max: 11475

## Per-slot GPU-state implications

These are state-size projections only; they do not claim the future runtime will use every listed intermediate representation.

| design | bytes / slot | MiB / slot | reduction vs current |
|---|---:|---:|---:|
| current full-E state | 516,326,360 | 492.41 | 1.00x |
| compact P, but still dense transaction on P | 222,094,040 | 211.81 | 2.32x |
| compact P weight+eligibility + P-bit dirty bitmap, before dirty txn payload | 92,997,092 | 88.69 | 5.55x |

A full-E dirty bitmap would be only 3,197,868 bytes (3.05 MiB); a P-only bitmap is 1,358,916 bytes (1.30 MiB).

If each dirty transaction record uses 16 bytes (u32 edge id + shift/lo/hi):

| dirty fraction of P | projected MiB / slot |
|---:|---:|
| 0.1% | 88.85 |
| 1.0% | 90.35 |
| 10.0% | 105.28 |
| 100.0% | 254.57 |

## Generated compact graph

- metadata: `artifacts/malecns-v1.0/plastic-fast-graph-v1.json`
- row offsets: `plastic-fast-graph-v1.row_offsets.u32le` (166,701 u32)
- source edge indices: `plastic-fast-graph-v1.edge_indices.u32le` (10,871,322 u32)
- post indices: `plastic-fast-graph-v1.post_indices.u32le` (10,871,322 u32)
- ordering: original incoming-CSR edge order is preserved within every post

## Interpretation boundary

This graph is only Phase B. It reduces the *possible mutable set* from E to P. The target runtime still needs dirty/sparse transaction state, outgoing adjacency, signal propagation frontiers, and eventually a live plasticity frontier so per-step work scales with current circuit activity rather than with P.
