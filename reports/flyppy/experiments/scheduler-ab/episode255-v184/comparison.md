# Flyppy scheduler A/B comparison

- left: `async` / episode 256..279
- right: `wave` / episode 256..279

| metric | left | right |
|---|---:|---:|
| first gate | 12/24 (50.0%) | 12/24 (50.0%) |
| second gate | 5/24 (20.8%) | 5/24 (20.8%) |
| total passed gates | 17 | 17 |
| mean passed gates | 0.708 | 0.708 |
| collision rate | 100.0% | 100.0% |
| mean altitude gain mm | -0.0055 | -0.0055 |
| control steps/s | 8.988 | 8.968 |
| realtime factor | 0.00449 | 0.00448 |
| mean commit staleness | 2.042 | 1.500 |
| reward mean staleness | 2.833 | 2.500 |
| no-reward mean staleness | 1.250 | 0.500 |

## left: async

- complete attempt rounds: 6
- mean source-version spread: 9.5
- max source-version spread: 13
- zero-spread rounds: 1

## right: wave

- complete attempt rounds: 6
- mean source-version spread: 0
- max source-version spread: 0
- zero-spread rounds: 6

## Frozen fixed evaluation

| condition | left first | right first | left second | right second | left mean gates | right mean gates |
|---|---:|---:|---:|---:|---:|---:|
| part3_frontier | 2 | 2 | 0 | 0 | 0.500 | 0.500 |
| midpoint | 2 | 2 | 0 | 0 | 0.500 | 0.500 |
| target | 1 | 1 | 0 | 0 | 0.250 | 0.250 |
