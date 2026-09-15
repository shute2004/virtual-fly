# Sparse population runtime verification

- generated_at_utc: 2026-09-15T04:04:07+00:00
- overall: PASS
- training performed: false
- production checkpoint unchanged: yes
- checkpoint digest before: `0891c7f3b1aa9e80ce1a4daadddcbf12b4e22d6e40d53f1bbdfcaf73c37df651`
- checkpoint digest after: `0891c7f3b1aa9e80ce1a4daadddcbf12b4e22d6e40d53f1bbdfcaf73c37df651`

## Structural state reduction

- N: 166,700
- E: 25,582,938
- P: 10,871,322
- P/E: 42.49442343%
- previous slot-local GPU state model: 516,326,360 bytes (492.41 MiB)
- current P-backed slot-local GPU state model: 222,094,040 bytes (211.81 MiB)
- structural reduction: 2.325x

The current sparse runtime still keeps dense transaction state over P. Dirty/sparse transaction storage is the next phase.

## Verification stages

| stage | result | seconds | returncode |
|---|---|---:|---:|
| vf-neural tests | PASS | 1.778 | 0 |
| population bridge check | PASS | 0.906 | 0 |
| real MaleCNS population smoke | PASS | 4.826 | 0 |

### vf-neural tests

```text

running 3 tests
...
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 13 filtered out; finished in 0.00s
```

### real MaleCNS population smoke

```text
population_shared_weight_smoke=PASS
checkpoint_modified=false
commit_order=slot0:v0->v1,slot1:stale-v0-rebased-on-v1->v2
```
