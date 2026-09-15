# Flyppy transaction dirty profile

- generated_at_utc: 2026-09-15T04:15:15+00:00
- training target: temporary copy of production state
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- episodes: 4
- population: 1
- max control steps / episode: 96
- PlasticFastGraph P: 10,871,322

## Per-episode dirty transaction set

| sample | episode | control steps | dirty edges D | D/P | nonzero shift | lower bound changed | upper bound changed | gates | collision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 96 | 4,810,086 | 44.245640% | 4,810,086 | 2,464,394 | 4,813 | 1 | False |
| 1 | 1 | 96 | 4,684,329 | 43.088863% | 4,684,329 | 2,380,871 | 1,521 | 1 | False |
| 2 | 2 | 96 | 4,538,857 | 41.750736% | 4,538,857 | 2,341,576 | 493 | 1 | False |
| 3 | 3 | 96 | 4,529,625 | 41.665816% | 4,529,625 | 2,257,712 | 417 | 1 | False |

## Aggregate

- min D/P: 41.665816%
- median D/P: 42.419800%
- mean D/P: 42.687764%
- max D/P: 44.245640%
- max dirty edges: 4,810,086

## Memory implication

- P weight+eligibility + neuron state + P-bit dirty bitmap: 88.69 MiB / slot
- 16-byte compact records for the observed maximum D: 73.40 MiB / slot
- lower-bound combined size at observed maximum D: 162.09 MiB / slot

The compact-record number is a lower bound, not the final implementation size: an online sparse transaction table also needs a lookup/indexing strategy so repeated deltas for the same edge update the same shift/lo/hi transform. This profile is used to size and choose that structure rather than guessing D/P.
