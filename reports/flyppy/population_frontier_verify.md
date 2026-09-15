# Outgoing propagation frontier runtime verification

- generated_at_utc: 2026-09-15T04:26:55+00:00
- overall: PASS
- training performed: false
- production checkpoint unchanged: yes
- checkpoint digest before: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- checkpoint digest after: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Contract

The frontier may use outgoing adjacency only to decide which posts can have received a previous-step synaptic event. Actual synaptic current accumulation remains in the original incoming-CSR order. The required golden contract compares causally relevant state, not dead scratch state.

## Verification stages

| stage | result | seconds | returncode |
|---|---|---:|---:|
| synthetic full-state golden | PASS | 3.511 | 0 |
| real MaleCNS parity | PASS | 5.112 | 0 |
| shared-weight population smoke | PASS | 3.038 | 0 |

### synthetic full-state golden

Command:

```text
cargo test -p vf-neural outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference -- --nocapture
```

stdout:

```text

running 1 test
test frontier_parity::outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 19 filtered out; finished in 0.43s
```

stderr:

```text
   Compiling vf-neural v0.1.0 (/Users/<local-user>/Desktop/virtual-fly/crates/vf-neural)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 2.38s
     Running unittests src/lib.rs (target/debug/deps/vf_neural-f05174437beaf85e)
```

### real MaleCNS parity

Command:

```text
cargo run -q -p vf-runner --bin population_frontier_malecns_parity --release -- --snapshot artifacts/malecns-v1.0
```

stdout:

```text
{
  "neuron_count": 166700,
  "edge_count": 25582938,
  "plastic_edge_count": 10871322,
  "chosen_pre": 6,
  "chosen_post": 0,
  "chosen_dopamine_pre": 831,
  "steps_compared": 10,
  "full_plastic_state_compared": true,
  "committed_global_weights_compared": true,
  "bitwise_equal": true
}
```

### shared-weight population smoke

Command:

```text
uv run python scripts/analysis/smoke_population_neural_bridge.py
```

stdout:

```text
population_shared_weight_smoke=PASS
checkpoint_modified=false
commit_order=slot0:v0->v1,slot1:stale-v0-rebased-on-v1->v2
```
