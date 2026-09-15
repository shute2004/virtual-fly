# Flyppy propagation frontier activity profile

- generated_at_utc: 2026-09-15T04:37:47+00:00
- overall: PASS
- training target: temporary copy of production state
- production checkpoint modified: no
- production checkpoint digest: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- N: 166,700
- E: 25,582,938
- P: 10,871,322
- episodes: 4
- max control steps / episode: 96
- sample stride: 12
- samples: 32

## Aggregate frontier density

- active pre mean: 2,966.8 (1.7797% of N)
- active pre median: 4,202.5
- active pre max: 5,585 (3.3503% of N)
- active outgoing F mean: 858,141.4 (3.3544% of E)
- candidate posts C mean: 80,751.0 (48.4409% of N)
- candidate incoming I(C) mean: 14,638,539.1 (57.2199% of E)
- propagation edge-touch F+I(C) mean: 15,496,680.5 (60.5743% of E)
- candidate-post plastic edges mean: 6,606,456.0 (60.7696% of P)

## Current total edge-work proxy

- previous dense propagation + dense-P plasticity: E+P = 36,454,260
- current frontier propagation + dense-P plasticity mean: 26,368,002.5
- current/previous mean edge-work ratio: 0.723317
- implied reduction: 1.383x

The candidate-post plastic-edge count is not itself a valid plasticity live set: eligibility and dopamine modulation can remain causally live after the immediate propagation frontier moves on. It is reported only as a structural bound/signal for the next live-plasticity analysis.

## Samples

| sample | neural step | normal step | active pre | F outgoing | C posts | I(C) incoming | plastic under C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7389 | 0 | 387 | 1,670 | 650 | 17,724 | 0 |
| 1 | 7401 | 12 | 291 | 4,301 | 2,562 | 377,555 | 72,618 |
| 2 | 7413 | 24 | 574 | 10,708 | 6,377 | 962,188 | 249,218 |
| 3 | 7425 | 36 | 2,296 | 726,074 | 114,912 | 22,016,206 | 10,102,135 |
| 4 | 7441 | 48 | 5,015 | 1,564,505 | 134,951 | 23,971,980 | 10,751,437 |
| 5 | 7453 | 60 | 4,308 | 1,401,419 | 125,777 | 23,537,961 | 10,755,211 |
| 6 | 7465 | 72 | 4,735 | 1,480,409 | 132,978 | 23,905,625 | 10,829,427 |
| 7 | 7477 | 84 | 4,722 | 1,439,281 | 129,854 | 23,625,243 | 10,794,229 |
| 8 | 7489 | 96 | 402 | 1,749 | 664 | 18,139 | 0 |
| 9 | 7501 | 108 | 299 | 3,041 | 1,744 | 244,475 | 42,576 |
| 10 | 7513 | 120 | 492 | 19,628 | 9,770 | 1,361,451 | 302,225 |
| 11 | 7525 | 132 | 1,340 | 217,918 | 53,759 | 12,455,538 | 6,341,180 |
| 12 | 7541 | 144 | 4,951 | 1,511,448 | 126,551 | 23,361,779 | 10,763,496 |
| 13 | 7553 | 156 | 5,479 | 1,626,304 | 133,016 | 23,864,852 | 10,792,875 |
| 14 | 7565 | 168 | 5,075 | 1,537,427 | 128,434 | 23,546,246 | 10,767,546 |
| 15 | 7577 | 180 | 5,068 | 1,473,015 | 129,969 | 23,656,629 | 10,775,825 |
| 16 | 7589 | 192 | 397 | 1,724 | 655 | 17,929 | 0 |
| 17 | 7601 | 204 | 268 | 3,733 | 2,562 | 309,514 | 42,340 |
| 18 | 7613 | 216 | 492 | 16,475 | 8,972 | 1,519,590 | 424,665 |
| 19 | 7625 | 228 | 3,826 | 1,261,048 | 126,143 | 23,180,409 | 10,463,071 |
| 20 | 7637 | 240 | 4,653 | 1,468,774 | 132,036 | 24,027,105 | 10,787,560 |
| 21 | 7653 | 252 | 4,930 | 1,536,159 | 130,181 | 23,542,590 | 10,702,260 |
| 22 | 7665 | 264 | 5,585 | 1,697,990 | 132,019 | 23,714,261 | 10,716,619 |
| 23 | 7677 | 276 | 5,249 | 1,556,729 | 133,678 | 23,940,845 | 10,809,080 |
| 24 | 7689 | 288 | 401 | 1,738 | 665 | 18,101 | 0 |
| 25 | 7701 | 300 | 293 | 4,517 | 2,927 | 387,773 | 70,096 |
| 26 | 7713 | 312 | 570 | 30,819 | 16,779 | 2,027,357 | 284,788 |
| 27 | 7725 | 324 | 4,097 | 1,296,350 | 129,056 | 23,489,668 | 10,743,930 |
| 28 | 7737 | 336 | 4,731 | 1,488,489 | 127,940 | 23,406,209 | 10,735,437 |
| 29 | 7753 | 348 | 4,935 | 1,441,138 | 137,564 | 24,209,265 | 10,810,568 |
| 30 | 7765 | 360 | 4,594 | 1,308,281 | 129,747 | 23,413,345 | 10,663,000 |
| 31 | 7777 | 372 | 4,483 | 1,327,664 | 141,139 | 24,305,698 | 10,813,181 |
