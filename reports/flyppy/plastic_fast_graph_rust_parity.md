# PlasticFastGraph Rust/Python parity

- generated_at_utc: 2026-09-15T03:58:47+00:00
- mutates training/checkpoint state: false
- overall: PASS
- rust_returncode: 0

## Structural count parity

| metric | Python expected | Rust actual | match |
|---|---:|---:|---|
| neuron_count | 166700 | 166700 | yes |
| edge_count | 25582938 | 25582938 | yes |
| dopamine_capable_post_count | 39845 | 39845 | yes |
| plastic_edge_count | 10871322 | 10871322 | yes |

## Rust output

```text
{
  "neuron_count": 166700,
  "edge_count": 25582938,
  "dopamine_capable_post_count": 39845,
  "plastic_edge_count": 10871322,
  "plastic_edge_fraction": 0.4249442343174189
}
```
