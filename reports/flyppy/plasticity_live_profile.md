# Flyppy plasticity live-frontier profile

- generated_at_utc: 2026-09-15T04:48:37+00:00
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
- sampled normal steps: 32
- replayed neural steps: 400

## Aggregate live-set density

- eligibility before != 0 mean: 1,795,231.0 (16.5135% of P); median: 1,326,262.0; max: 4,611,218 (42.4164%)
- new local contribution != 0 mean: 625,420.0 (5.7529% of P); median: 701,132.5; max: 1,365,581 (12.5613%)
- eager live union mean: 1,845,349.5 (16.9745% of P); median: 1,534,657.5; max: 4,633,509 (42.6214%)
- eligibility after != 0 mean: 1,845,349.5 (16.9745% of P); median: 1,534,657.5; max: 4,633,509 (42.6214%)
- plastic edges under nonzero post modulation mean: 5,499,678.8 (50.5889% of P); median: 5,868,163.5; max: 10,843,648 (99.7454%)
- actual weight delta != 0 mean: 1,815,317.6 (16.6982% of P); median: 1,401,557.5; max: 4,620,024 (42.4974%)
- lazy immediate union (local OR delta) mean: 1,827,899.3 (16.8140% of P); median: 1,486,337.5; max: 4,622,623 (42.5213%)

## Dense-P replacement signal

- current plasticity dispatch proxy: 10,871,322 edges / neural step
- eager active-set proxy mean: 1,845,349.5 (0.169745 of dense P; 5.891x reduction)
- lazy immediate-work proxy mean: 1,827,899.3 (0.168140 of dense P; 5.947x reduction)

`eager live union` is the set an implementation must touch if eligibility decay remains explicit every step. `lazy immediate union` is the stronger optimization target if the geometric decay of untouched eligibility is deferred and reconstructed exactly when a new local contribution or nonzero modulation makes that edge causally relevant again.

## Samples

| sample | normal step | eligibility before | local | eager union | eligibility after | edges under modulation | delta | lazy union |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 24 | 7 | 0 | 7 | 7 | 0 | 0 | 0 |
| 3 | 36 | 50,921 | 134,147 | 176,127 | 176,127 | 4,180,464 | 154,216 | 171,772 |
| 4 | 48 | 2,637,967 | 1,082,908 | 2,804,584 | 2,804,584 | 10,376,914 | 2,769,360 | 2,779,655 |
| 5 | 60 | 3,853,744 | 1,214,075 | 3,891,660 | 3,891,660 | 10,814,858 | 3,884,137 | 3,886,133 |
| 6 | 72 | 4,276,601 | 1,365,581 | 4,311,304 | 4,311,304 | 10,816,153 | 4,300,831 | 4,304,062 |
| 7 | 84 | 4,611,218 | 1,337,250 | 4,633,509 | 4,633,509 | 10,816,281 | 4,620,024 | 4,622,623 |
| 8 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | 108 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | 120 | 7 | 0 | 7 | 7 | 0 | 0 | 0 |
| 11 | 132 | 2,050 | 8,706 | 10,573 | 10,573 | 0 | 0 | 8,706 |
| 12 | 144 | 2,331,224 | 1,017,035 | 2,410,457 | 2,410,457 | 7,555,863 | 2,243,359 | 2,292,557 |
| 13 | 156 | 3,306,608 | 1,304,431 | 3,406,637 | 3,406,637 | 9,729,592 | 3,334,504 | 3,352,624 |
| 14 | 168 | 4,099,679 | 1,335,164 | 4,145,924 | 4,145,924 | 10,027,678 | 4,073,551 | 4,088,585 |
| 15 | 180 | 4,478,461 | 1,332,538 | 4,497,096 | 4,497,096 | 10,783,109 | 4,477,887 | 4,481,267 |
| 16 | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 17 | 204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | 216 | 8 | 12 | 20 | 20 | 0 | 0 | 12 |
| 19 | 228 | 196,353 | 303,819 | 427,830 | 427,830 | 3,573,252 | 319,187 | 393,324 |
| 20 | 240 | 2,229,158 | 894,300 | 2,285,212 | 2,285,212 | 10,008,241 | 2,251,452 | 2,261,588 |
| 21 | 252 | 3,288,964 | 1,220,608 | 3,377,255 | 3,377,255 | 10,755,272 | 3,364,548 | 3,368,389 |
| 22 | 264 | 3,739,522 | 1,353,614 | 3,779,214 | 3,779,214 | 10,768,029 | 3,763,647 | 3,766,974 |
| 23 | 276 | 4,207,874 | 1,346,707 | 4,241,925 | 4,241,925 | 10,843,648 | 4,237,182 | 4,237,770 |
| 24 | 288 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | 300 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | 312 | 7 | 1 | 8 | 8 | 0 | 0 | 1 |
| 27 | 324 | 423,366 | 507,965 | 784,103 | 784,103 | 3,573,252 | 559,756 | 711,087 |
| 28 | 336 | 2,258,429 | 987,994 | 2,310,509 | 2,310,509 | 10,147,366 | 2,281,406 | 2,290,588 |
| 29 | 348 | 3,162,875 | 1,012,176 | 3,213,035 | 3,213,035 | 10,237,383 | 3,178,938 | 3,186,226 |
| 30 | 360 | 3,975,150 | 1,081,481 | 4,005,282 | 4,005,282 | 10,478,651 | 3,972,714 | 3,978,981 |
| 31 | 372 | 4,317,198 | 1,172,929 | 4,338,906 | 4,338,906 | 10,503,714 | 4,303,463 | 4,309,853 |
