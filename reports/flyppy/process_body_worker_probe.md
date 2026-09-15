# Flyppy process-isolated body-worker probe

- generated_at_utc: 2026-09-15T06:48:56+00:00
- overall: PASS
- training performed: false
- production checkpoint touched: false
- requested slots: 8
- control steps per slot and scheduling mode: 24
- physics steps per control step: 10
- multiprocessing start method: spawn
- child ownership: FlyBody + MuJoCo + eye renderer + MaleCNS retina seam + periphery + Flyppy course
- parent/child boundary: retinal body currents out; individual motor-neuron spikes in
- numerical contract: complete observe+act transcript digest must match between sequential and parallel scheduling

| concurrency | aggregate control steps | sequential s | parallel s | sequential steps/s | parallel steps/s | body-side speedup | trajectory bitwise equal |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 24 | 0.7865 | 0.7925 | 30.52 | 30.28 | 0.992x | True |
| 2 | 48 | 1.5570 | 1.0174 | 30.83 | 47.18 | 1.530x | True |
| 4 | 96 | 3.0651 | 1.6264 | 31.32 | 59.02 | 1.885x | True |
| 8 | 192 | 6.1475 | 2.9303 | 31.23 | 65.52 | 2.098x | True |

Interpretation: this probe measures only the body-side process boundary. It intentionally excludes CNS compute; production can place the shared GPU CNS batch between the observe and act phases.
