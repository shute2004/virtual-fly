# Canonical experiment v1 result

- Git: `7fa464aad7269d34f46f1171080b51e095d1d811`; dirty: `false`
- MaleCNS: 166,700 neurons / 25,582,938 edges / class-DAN 338
- Body/environment: v7 / v7
- Vision: direct-ray K13
- Haltere: 97-neuron timing subset, gain 0.05, interaction-load-v2
- Training: 6 episodes, population 2, async, boundary-band
- Global weight version: 0 -> 6
- Aggregate neural step: 484
- Gate passes: 6 total; collisions: 4

## Weight change

- changed exactly: 2,163,179 / 25,582,938 edges
- |Δw| > 1e-7: 1,461,896
- |Δw| > 1e-6: 536,060
- strengthened: 961,611
- weakened: 1,201,568
- max |Δw|: 0.0018288875
- mean |Δw|: 1.82520009e-07

## Frozen evaluation

Evaluation plasticity: OFF. Neuromodulatory DAN stimulation: OFF. Same body/environment/sensory/start condition for both.

- initial/global v0: passed 1 gate(s), terminal `gate`, step 120
- final/global v6: passed 1 gate(s), terminal `gate`, step 120

No categorical behavioral improvement was observed in this short canonical run. That is retained as the result rather than extending or tuning the experiment.

## Interpretation

This establishes a fully traceable current-code execution chain and demonstrates that local plasticity changed stored CNS weights. It does **not** establish generalization, long-term learning stability, or improved behavior.
