# Embodiment v0

## Scope

This slice connects the current MaleCNS neural runtime to a FlyBody body without introducing an external policy.

Implemented:

- FlyGym 2.1.0 / FlyBody dependency.
- FlyBody wing DoF actuation smoke.
- Persistent Rust CNS bridge over JSONL stdin/stdout.
- MaleCNS annotation-driven discovery of bilateral DNg02, PAM08, and PPL1 groups.
- Bilateral DNg02 population activity mapped to modulation of left/right wing-beat amplitude.
- Physical Flyppy gate world with explicit MuJoCo contact pairs.
- FlyBody compound-eye ommatidia rendering inside the Flyppy world.
- One-command embodiment validation in `scripts/dev/embodiment.sh`.

Not implemented in this slice:

- A fabricated retinotopic mapping from FlyGym ommatidia to MaleCNS visual neurons.
- A claim that the analytic prototype wing beat is measured Drosophila kinematics.
- Flyppy task learning end-to-end.
- Direct motor-neuron-to-flight-muscle modeling.

## Boundary

The intended runtime path is:

```text
Flyppy geometry
  -> FlyGym compound-eye ommatidia
  -> [retinotopic MaleCNS sensory mapping: next slice]
  -> persistent MaleCNS neural runtime
  -> bilateral DNg02 population activity
  -> wing-beat amplitude adapter
  -> FlyBody / MuJoCo physics
  -> position + collision / gate outcome
  -> PAM08 or PPL1 stimulation
  -> local neural plasticity
```

The motor adapter is intentionally not allowed to inspect obstacle geometry or game state. It receives only CNS activity. Likewise, the environment is not allowed to change synaptic weights; outcome events may only stimulate configured neuromodulatory neurons.

## Why DNg02

DNg02 is used only as the first coarse flight-amplitude descending readout. Published experiments associate population activity of DNg02 descending neurons with wing stroke amplitude / thrust regulation. This does not imply that DNg02 alone constitutes the full flight motor pathway.

## Vision status

FlyGym 2.1 exposes FlyBody compound-eye cameras and per-ommatidium readouts. The Flyppy world is already rendered into those ommatidia. The remaining sensory problem is biological correspondence: which MaleCNS optic-lobe neurons should receive each ommatidial channel.

That mapping must be based on MaleCNS optic-lobe spatial/eyemap information or another documented anatomical correspondence. It must not be generated from arbitrary neuron ordering merely to make the game learnable.
