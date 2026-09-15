# Flyppy shared-weight population benchmark

- episodes per run: 8
- max control steps per episode: 40
- population=1 aggregate control steps/s: 2.793
- population=2 aggregate control steps/s: 2.825
- population=2 / population=1 throughput ratio: 1.011x
- population=1 elapsed: 114.573 s
- population=2 elapsed: 113.283 s
- population=2 mean staleness: 0.875
- population=2 max staleness: 1
- benchmark curriculum drift: effectively frozen (1e-9 steps)
- production checkpoint modified: false
