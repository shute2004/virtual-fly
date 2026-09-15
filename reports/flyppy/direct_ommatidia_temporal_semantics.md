# Flyppy direct ommatidia temporal semantics

- generated_at_utc: 2026-09-15T09:28:15+00:00
- overall: PASS
- samples: 64
- direct path: body-excluded mj_multiRay; no RGB framebuffer
- production checkpoint modified: no

## Temporal comparison

| rays/ommatidium | sensor ms | current Pearson r | current MAE | light-delta Pearson r | light-delta MAE |
|---:|---:|---:|---:|---:|---:|
| 3 | 3.099 | 0.897551 | 0.214666 | 0.754122 | 0.023395 |
| 7 | 6.853 | 0.911810 | 0.203167 | 0.834855 | 0.016696 |
| 13 | 12.594 | 0.915181 | 0.198368 | 0.853268 | 0.014596 |

## Current sign agreement by magnitude

| rays/ommatidium | threshold | compared | sign agreement |
|---:|---:|---:|---:|
| 3 | 0.001 | 177638 | 55.105% |
| 3 | 0.01 | 174809 | 55.509% |
| 3 | 0.05 | 158168 | 60.524% |
| 3 | 0.1 | 146596 | 64.448% |
| 7 | 0.001 | 177599 | 57.268% |
| 7 | 0.01 | 174709 | 57.713% |
| 7 | 0.05 | 158350 | 62.813% |
| 7 | 0.1 | 146388 | 66.949% |
| 13 | 0.001 | 177538 | 59.484% |
| 13 | 0.01 | 174659 | 59.948% |
| 13 | 0.05 | 157927 | 65.264% |
| 13 | 0.1 | 145804 | 69.146% |

Near-zero sign flips are excluded by the thresholded table. Raster is a temporal compatibility reference, not biological ground truth.
