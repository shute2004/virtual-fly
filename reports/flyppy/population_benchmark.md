# Flyppy shared-weight population benchmark

- episodes per run: 8
- max control steps per episode: 40
- current serial trainer aggregate control steps/s: 2.785
- population=1 aggregate control steps/s: 3.008
- population=2 aggregate control steps/s: 3.080
- population=1 / serial throughput ratio: 1.080x
- population=2 / population=1 throughput ratio: 1.024x
- population=2 / serial throughput ratio: 1.106x
- serial elapsed: 114.881 s
- population=1 elapsed: 106.369 s
- population=2 elapsed: 103.886 s
- population=2 mean staleness: 0.875
- population=2 max staleness: 1
- benchmark curriculum drift: effectively frozen (1e-9 steps)
- serial viewer telemetry cadence: effectively disabled (one initial sample per episode)
- production checkpoint modified: false
