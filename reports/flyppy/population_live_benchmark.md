# Flyppy eager-live population runtime benchmark

- generated_at_utc: 2026-09-15T05:25:45+00:00
- overall: PASS
- episodes per case: 8
- max control steps per episode: 32
- curriculum drift: effectively frozen (1e-9 reset-step changes)
- production checkpoint unchanged: yes
- production checkpoint digest before: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- production checkpoint digest after: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Throughput

| runtime | population | aggregate control steps/s | trainer elapsed s | outer wall s | mean staleness | max staleness |
|---|---:|---:|---:|---:|---:|---:|
| dense | 1 | 2.557114 | 100.113 | 104.596 | 0.000 | 0 |
| dense | 2 | 2.538828 | 100.834 | 106.556 | 0.875 | 1 |
| live | 1 | 2.525217 | 101.377 | 105.131 | 0.000 | 0 |
| live | 2 | 2.574808 | 99.425 | 105.712 | 0.875 | 1 |
| live | 4 | 2.435807 | 105.099 | 115.021 | 2.250 | 3 |
| live | 8 | 2.577160 | 99.334 | 117.666 | 3.500 | 7 |

## Dense-P → eager-live speedup

- N=1: 0.988x (2.557114 → 2.525217 aggregate control steps/s)
- N=2: 1.014x (2.538828 → 2.574808 aggregate control steps/s)

## Live-runtime population scaling

- N=1: 2.525217 aggregate control steps/s; vs N=1 = 1.000x
- N=2: 2.574808 aggregate control steps/s; vs N=1 = 1.020x
- N=4: 2.435807 aggregate control steps/s; vs N=1 = 0.965x
- N=8: 2.577160 aggregate control steps/s; vs N=1 = 1.021x

## Numerical / trajectory parity

Dense-P and eager-live cases use the same production checkpoint, frozen curriculum, seeds, body physics, and trainer. The following compares episode-result signatures and final checkpoint tree digests.

| population | episode results equal | final checkpoint digest equal |
|---:|---|---|
| 1 | yes | yes |
| 2 | yes | yes |
