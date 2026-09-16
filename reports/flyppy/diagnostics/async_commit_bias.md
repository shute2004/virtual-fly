# Flyppy async shared-weight completion-order bias

- commit log: `artifacts/experiments/flyppy-v3/commit-log.jsonl`
- records: 184
- episode coverage: 72..255
- persisted max commit version: 184
- unpersisted commit rows ignored: 18

## Reward episode と no-reward episode

| window | class | episodes | mean steps | mean staleness | max staleness | mean commit rank | reward events | aversive events |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | PAMあり | 55 | 91.15 | 4.909 | 9 | 0.546 | 62 | 55 |
| all | PAMなし | 129 | 31.22 | 1.302 | 4 | 0.480 | 0 | 129 |
| last48 | PAMあり | 18 | 86.00 | 4.222 | 9 | 0.681 | 23 | 18 |
| last48 | PAMなし | 30 | 24.53 | 1.300 | 3 | 0.391 | 0 | 30 |
| last24 | PAMあり | 12 | 85.83 | 2.917 | 7 | 0.717 | 16 | 12 |
| last24 | PAMなし | 12 | 24.67 | 1.250 | 3 | 0.283 | 0 | 12 |

## Episode長とstaleness

| window | Pearson r(control steps, staleness) |
|---|---:|
| all | 0.6759 |
| last48 | 0.6004 |
| last24 | 0.3903 |

## 同一attempt roundのsource weight version幅

| window | complete rounds | mean spread | max spread | zero-spread rounds |
|---|---:|---:|---:|---:|
| all | 8 | 7.250 | 13 | 2 |
| last48 | 8 | 7.250 | 13 | 2 |
| last24 | 6 | 9.167 | 13 | 1 |

## 最新window (last24) のslot別

| slot | episodes | reward episodes | reward rate | mean steps | mean staleness | max staleness |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 100.0% | 88.67 | 3.000 | 7 |
| 1 | 6 | 0 | 0.0% | 23.00 | 1.167 | 3 |
| 2 | 6 | 6 | 100.0% | 83.00 | 2.833 | 7 |
| 3 | 6 | 0 | 0.0% | 26.33 | 1.333 | 3 |

## Interpretation

shared-weight transactionのrebaseはstale episodeの局所更新を捨てないが、そのepisode中のsensory/neural/body trajectoryはsource weight version上で生成される。
したがって長いepisodeほどstalenessが大きい場合、成功経験と失敗経験が異なるweight世代を経験するschedule biasが残る。
boundary-bandの固定slot割当はcourse seedごとのepisode数を揃えるが、async modeでは同一attempt roundのslotが異なるsource weight versionから出発し得る。
wave modeの直接の狙いはcommit時stalenessをゼロにすることではなく、同一attempt roundのsource weight version幅を0にしてtrajectory生成世代を揃えることである。
