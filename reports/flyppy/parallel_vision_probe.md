# FlyBody process-parallel compound-eye rendering probe

- generated_at_utc: 2026-09-15T06:41:29+00:00
- overall: PASS
- training performed: false
- production checkpoint touched: false
- requested slots: 8
- measured iterations per scheduling mode: 12
- multiprocessing start method: spawn
- ownership: one complete FlyBody/MuJoCo/renderer per child process
- renderer thread: child-process main thread only
- numerical contract: static-scene L/R ommatidia arrays must be bitwise identical

| concurrency | aggregate eye-readout calls | sequential s | parallel s | sequential readouts/s | parallel readouts/s | speedup | bitwise equal |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 0.3116 | 0.2615 | 38.51 | 45.89 | 1.192x | True |
| 2 | 24 | 0.5821 | 0.3506 | 41.23 | 68.46 | 1.660x | True |
| 4 | 48 | 1.1232 | 0.7235 | 42.74 | 66.34 | 1.552x | True |
| 8 | 96 | 2.1721 | 1.2016 | 44.20 | 79.89 | 1.808x | True |

Interpretation: process isolation is deliberate on macOS. A child-native rendering failure is reported here rather than moving an OpenGL context across Python threads or terminating the parent probe.
