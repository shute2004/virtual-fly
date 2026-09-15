# Eager live-plasticity population runtime verification

- generated_at_utc: 2026-09-15T05:10:47+00:00
- overall: PASS
- training performed: false
- production checkpoint unchanged: yes
- checkpoint digest before: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`
- checkpoint digest after: `bc3185a6cbff63b23e35faaec8b01f4a26777bd1db85a53d904f7ad137ec854b`

## Contract

Propagation uses outgoing adjacency only to discover candidate posts; actual current accumulation preserves original incoming-CSR order. Plasticity uses an eager live bitmap: every edge with non-zero incoming eligibility remains live, and edges whose current local term can become non-zero are added before applying the unchanged eligibility/weight/transaction equations. The two physical bitmap buffers are mirrored after every active plasticity step so slots that are idle during another slot's extra teaching steps preserve their frontier exactly. No lazy decay approximation is used.

## Expected structural cost

- MaleCNS PlasticFastGraph P: 10,871,322 edges
- dense plasticity dispatch: P edge invocations / active slot / neural step
- live bitmap dispatch: ceil(P/32) = 339,729 word invocations / active slot / neural step, plus equation work only for live bits
- two mirrored live bitmaps: about 2.59 MiB / slot

## Verification stages

| stage | result | seconds | returncode |
|---|---|---:|---:|
| synthetic per-step full-state golden | PASS | 3.860 | 0 |
| two-slot async teaching parity | PASS | 0.220 | 0 |
| real MaleCNS live-runtime parity | PASS | 6.167 | 0 |
| shared-weight population smoke | PASS | 3.473 | 0 |

### synthetic per-step full-state golden

Command:

```text
cargo test -p vf-neural outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference -- --nocapture
```

stdout:

```text

running 1 test
test frontier_parity::outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 21 filtered out; finished in 0.28s
```

stderr:

```text
   Compiling vf-neural v0.1.0 (/Users/<local-user>/Desktop/virtual-fly/crates/vf-neural)
warning: field `plasticity_pipeline` is never read
   --> crates/vf-neural/src/gpu_population_frontier.rs:112:5
    |
 90 | pub struct GpuPopulationRuntime {
    |            -------------------- field in this struct
...
112 |     plasticity_pipeline: wgpu::ComputePipeline,
    |     ^^^^^^^^^^^^^^^^^^^
    |
    = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `step_batch_with_read`, `step_batch_no_read`, and `encode_neural_step` are never used
   --> crates/vf-neural/src/gpu_population_frontier.rs:516:12
    |
123 | impl GpuPopulationRuntime {
    | ------------------------- methods in this implementation
...
516 |     pub fn step_batch_with_read(
    |            ^^^^^^^^^^^^^^^^^^^^
...
596 |     pub fn step_batch_no_read(
    |            ^^^^^^^^^^^^^^^^^^
...
692 |     fn encode_neural_step(
    |        ^^^^^^^^^^^^^^^^^^

warning: `vf-neural` (lib test) generated 2 warnings
    Finished `test` profile [unoptimized + debuginfo] target(s) in 2.85s
     Running unittests src/lib.rs (target/debug/deps/vf_neural-f05174437beaf85e)
```

### two-slot async teaching parity

Command:

```text
cargo test -p vf-neural live_plasticity_bitmap_preserves_idle_slot_across_async_teaching_steps -- --nocapture
```

stdout:

```text

running 1 test
test frontier_parity::live_plasticity_bitmap_preserves_idle_slot_across_async_teaching_steps ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 21 filtered out; finished in 0.11s
```

stderr:

```text
warning: field `plasticity_pipeline` is never read
   --> crates/vf-neural/src/gpu_population_frontier.rs:112:5
    |
 90 | pub struct GpuPopulationRuntime {
    |            -------------------- field in this struct
...
112 |     plasticity_pipeline: wgpu::ComputePipeline,
    |     ^^^^^^^^^^^^^^^^^^^
    |
    = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `step_batch_with_read`, `step_batch_no_read`, and `encode_neural_step` are never used
   --> crates/vf-neural/src/gpu_population_frontier.rs:516:12
    |
123 | impl GpuPopulationRuntime {
    | ------------------------- methods in this implementation
...
516 |     pub fn step_batch_with_read(
    |            ^^^^^^^^^^^^^^^^^^^^
...
596 |     pub fn step_batch_no_read(
    |            ^^^^^^^^^^^^^^^^^^
...
692 |     fn encode_neural_step(
    |        ^^^^^^^^^^^^^^^^^^

warning: `vf-neural` (lib test) generated 2 warnings
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.06s
     Running unittests src/lib.rs (target/debug/deps/vf_neural-f05174437beaf85e)
```

### real MaleCNS live-runtime parity

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
  "all_neuron_events_compared_each_step": true,
  "final_full_neuron_state_compared": true,
  "full_plastic_state_compared": true,
  "committed_global_weights_compared": true,
  "bitwise_equal": true
}
```

stderr:

```text
warning: field `plasticity_pipeline` is never read
   --> crates/vf-neural/src/gpu_population_frontier.rs:112:5
    |
 90 | pub struct GpuPopulationRuntime {
    |            -------------------- field in this struct
...
112 |     plasticity_pipeline: wgpu::ComputePipeline,
    |     ^^^^^^^^^^^^^^^^^^^
    |
    = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `step_batch_with_read`, `step_batch_no_read`, and `encode_neural_step` are never used
   --> crates/vf-neural/src/gpu_population_frontier.rs:516:12
    |
123 | impl GpuPopulationRuntime {
    | ------------------------- methods in this implementation
...
516 |     pub fn step_batch_with_read(
    |            ^^^^^^^^^^^^^^^^^^^^
...
596 |     pub fn step_batch_no_read(
    |            ^^^^^^^^^^^^^^^^^^
...
692 |     fn encode_neural_step(
    |        ^^^^^^^^^^^^^^^^^^
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

stderr:

```text
warning: field `plasticity_pipeline` is never read
   --> crates/vf-neural/src/gpu_population_frontier.rs:112:5
    |
 90 | pub struct GpuPopulationRuntime {
    |            -------------------- field in this struct
...
112 |     plasticity_pipeline: wgpu::ComputePipeline,
    |     ^^^^^^^^^^^^^^^^^^^
    |
    = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `step_batch_with_read`, `step_batch_no_read`, and `encode_neural_step` are never used
   --> crates/vf-neural/src/gpu_population_frontier.rs:516:12
    |
123 | impl GpuPopulationRuntime {
    | ------------------------- methods in this implementation
...
516 |     pub fn step_batch_with_read(
    |            ^^^^^^^^^^^^^^^^^^^^
...
596 |     pub fn step_batch_no_read(
    |            ^^^^^^^^^^^^^^^^^^
...
692 |     fn encode_neural_step(
    |        ^^^^^^^^^^^^^^^^^^
```
