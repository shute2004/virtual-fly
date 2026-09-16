# Flyppy reinforcement DAN reach

- checkpoint: `artifacts/experiments/flyppy-v3/checkpoint`
- checkpoint neural step: 15781
- reward DAN: PAM01 / 44 neurons
- aversive DAN: PPL101 / 2 neurons
- all released dopamine neurons: 392
- weight epsilon: 1.0e-07

## DAN source reach

| source | neurons | released outgoing edges | released contacts | directly reached posts |
|---|---:|---:|---:|---:|
| reward_dan | 44 | 18790 | 28019 | 2061 |
| aversive_dan | 2 | 5743 | 17287 | 3782 |
| all_dopamine | 392 | 241694 | 583117 | 39845 |

## Reward / aversive direct-post overlap

- shared posts: 1646
- reward-only posts: 415
- aversive-only posts: 2136
- reward側 overlap: 79.864%
- aversive側 overlap: 43.522%
- Jaccard: 39.218%

| motor distance | reward posts | aversive posts | shared | reward overlap | aversive overlap | Jaccard |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| 1 | 3 | 5 | 1 | 33.3% | 20.0% | 14.3% |
| 2 | 246 | 486 | 101 | 41.1% | 20.8% | 16.0% |
| 3 | 1812 | 3291 | 1544 | 85.2% | 46.9% | 43.4% |
| 4 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% |

## Async commit log exposure proxy

- coverage: episode 72..255 / 184 commits
- reward events: 62
- aversive events: 184

| source | events | event × neurons | event × outgoing edges | event × released contacts |
|---|---:|---:|---:|---:|
| reward_dan | 62 | 2728 | 1164980 | 1737178 |
| aversive_dan | 184 | 368 | 1056712 | 3180808 |

event × released contacts の reward / aversive 比: **0.546x**

この値は実DAN spike数そのものではなく、観測event数とreleased connectivityを掛けた構造的exposure proxyである。
commit logはasync shared-weight化後のみを含み、それ以前のv3 serial episodeは含まない。

## Reached-post set と学習weight

| post set | posts | plastic posts | plastic incoming edges | changed incoming | changed % | positive | negative | motor d0 | motor d1 | motor d2 | motor d3 | motor d4 | outside d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_reached | 2061 | 2061 | 588232 | 559556/676938 | 82.660% | 219047 | 340509 | 0 | 3 | 246 | 1812 | 0 | 0 |
| aversive_reached | 3782 | 3782 | 929098 | 869725/1057546 | 82.240% | 355319 | 514406 | 0 | 5 | 486 | 3291 | 0 | 0 |
| reward_only | 415 | 415 | 162087 | 140808/177714 | 79.233% | 68660 | 72148 | 0 | 2 | 145 | 268 | 0 | 0 |
| aversive_only | 2136 | 2136 | 502953 | 450977/558322 | 80.774% | 204932 | 246045 | 0 | 4 | 385 | 1747 | 0 | 0 |
| reward_and_aversive | 1646 | 1646 | 426145 | 418748/499224 | 83.880% | 150387 | 268361 | 0 | 1 | 101 | 1544 | 0 | 0 |
| task_reinforcement_union | 4197 | 4197 | 1091185 | 1010533/1235260 | 81.807% | 423979 | 586554 | 0 | 7 | 631 | 3559 | 0 | 0 |
| other_dopamine_only | 35648 | 35648 | 9780137 | 5068230/10249833 | 49.447% | 2117475 | 2950755 | 0 | 453 | 15830 | 19352 | 11 | 2 |
| all_dopamine_reached | 39845 | 39845 | 10871322 | 6078763/11485093 | 52.927% | 2541454 | 3537309 | 0 | 460 | 16461 | 22911 | 11 | 2 |

## Motor近傍のreinforcement到達post

| post set | motor distance | posts | changed incoming | changed % | positive | negative | mean signed delta | mean abs delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reward_reached | 1 | 3 | 2819/3402 | 82.863% | 971 | 1848 | -0.000113695 | 0.000863049 |
| reward_reached | 2 | 246 | 135058/171622 | 78.695% | 55237 | 79821 | -0.000130693 | 0.00212428 |
| reward_reached | 3 | 1812 | 421679/501914 | 84.014% | 162839 | 258840 | -9.46015e-06 | 0.000320623 |
| aversive_reached | 1 | 5 | 5710/6802 | 83.946% | 2438 | 3272 | 0.000112819 | 0.00085841 |
| aversive_reached | 2 | 486 | 216296/272121 | 79.485% | 96081 | 120215 | -4.27781e-05 | 0.00140902 |
| aversive_reached | 3 | 3291 | 647719/778623 | 83.188% | 256800 | 390919 | 6.68651e-06 | 0.00025588 |
| task_reinforcement_union | 1 | 7 | 6681/7960 | 83.932% | 2748 | 3933 | 8.31466e-05 | 0.000753963 |
| task_reinforcement_union | 2 | 631 | 284001/364137 | 77.993% | 123189 | 160812 | -6.34412e-05 | 0.0012754 |
| task_reinforcement_union | 3 | 3559 | 719851/863163 | 83.397% | 298042 | 421809 | 5.22946e-06 | 0.000241705 |

## Interpretation boundary

`reward_reached` はPAM01から、`aversive_reached` はPPL101からreleased edgeを直接受けるpost集合である。
direct-post overlapは構造的な共通到達先を示すだけで、同じepisode履歴でPAM/PPLが同じweight deltaを作ることまでは意味しない。
`other_dopamine_only` はtaskで明示刺激する2群からは直接入力を受けないが、他のreleased dopamine neuronから入力を受けるpost集合である。
weight変化はcheckpointの最終差分であり、その変化をPAM/PPLイベント単独へ因果帰属するものではない。
motor distanceはreleased connectivity上でactuated motor neuronを0としてincoming方向へ遡った最短距離である。
