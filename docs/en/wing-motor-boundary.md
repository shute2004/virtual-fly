# Wing motor-neuron and peripheral-muscle boundary

[English](wing-motor-boundary.md) · [日本語](../ja/wing-motor-boundary.md) · [简体中文](../zh-CN/wing-motor-boundary.md)

## 1. Purpose

To avoid placing a behavior selector outside the CNS, the MaleCNS-to-body boundary is pushed down to individual released wing motor neurons.

The following path is not used:

```text
DNg02 population activity
  -> mean firing rate
  -> left/right difference and global average
  -> wing amplitude
```

The current path is:

```text
FlyBody eye
  -> local current into R1-R6 body IDs
  -> whole MaleCNS
  -> VNC premotor circuitry
  -> spikes of individual wing motor neurons
  -> individual motor-unit / neuromuscular state
  -> identified wing muscles
  -> virtual-muscle torque
  -> FlyBody / MuJoCo
```

This is not an action decoder that selects behavior from a motor-neuron activity vector. It is a peripheral chain from neurons to muscles and from muscles to joint mechanics.

## 2. MaleCNS wing-motor inventory and included neurons

`scripts/data/prepare_wing_motor_map.py` extracts released MaleCNS v1.0 neurons matching:

```text
superclass = vnc_motor
subclass   = wm
```

In the dataset inspected on 2026-09-14:

- 67 neurons;
- left 33 / right 34;
- 26 annotated types;
- `identified` 38;
- `identified_group` 22;
- `identified_variable` 2;
- `putative` 2;
- `unknown` 3.

The initial learning boundary uses only the 60 neurons labeled `identified` or `identified_group`. The remaining seven are not connected by guessing their targets.

Representative exclusions include:

- `MNwm35 -> iii4`: `putative`;
- `MNwm36`: muscle target unresolved;
- `tpn`: variable innervation of tp1 / tp2;
- released wing motor neurons without an identified type.

Grouped types such as DLM / DVM are not assigned to individual muscle fibers merely from body-ID ordering without additional evidence.

## 3. Implemented peripheral layer

### 3.1 Individual motor-neuron spikes to motor-unit state

`scripts/embodiment/wing_muscle_periphery.py`:

- requests spikes for all 60 body IDs individually from the neural bridge;
- maintains independent activation state per body ID;
- uses different time constants for power, direct-steering, and indirect-control muscles;
- fails closed if any requested body ID is missing;
- does not construct population firing rates or left/right averages.

When multiple motor units converge on the same muscle group, each unit is updated independently and then combined through a saturating recruitment function at the muscle. This represents peripheral force convergence rather than action decoding.

Current time constants and spike-recruitment increments are `calibrated` bootstrap parameters, not measured MaleCNS constants.

### 3.2 Asynchronous power muscles

DLM / DVM are asynchronous indirect flight muscles. The model therefore does not assume that motor neurons explicitly command each ~200 Hz wingbeat.

`scripts/embodiment/flybody_muscle_adapter.py` converts low-frequency DLM / DVM activation into available power for a thoracic flight oscillator. Wingbeat phase is a peripheral mechanical state approximating stretch activation and thoracic resonance, not an action selected by the CNS.

Only the power muscles receive initial activation to represent that an episode begins in flight. This is an initial condition, like initial forward velocity or wing phase, not a motor command injected every step.

### 3.3 Steering muscles to torque

FlyBody does not contain anatomical wing muscles as direct MJCF muscle elements, so the current first-stage implementation uses a virtual-muscle layer through `qfrc_applied`.

Only direct-steering muscles whose qualitative effect is comparatively well constrained currently contribute torque:

- b1: increases stroke-amplitude-side drive;
- b2: increases stroke-amplitude-side drive;
- b3: antagonist to the basalar pair;
- i1: decreases stroke amplitude.

Numeric gains are `calibrated`; they are not measured moment arms.

Other identified steering and indirect-control muscles retain activation state, but the implementation does not invent an action direction and torque for them.

### 3.4 Connection to FlyBody

The old DNg02 adapter remains only for historical physics diagnostics and is not part of the current Flyppy learning path.

`FlyBodyMuscleAdapter` neutralizes the six inherited idealized position actuators by setting them to the current angle every physics step. This prevents the old analytic position commands from generating behavior. Virtual-muscle torque is then applied to the wing degrees of freedom.

## 4. Reward- and aversion-associated stimulation

Neuromodulatory outcome stimulation is separate from the motor boundary.

The only direct outcome-related input external code may inject into the neural runtime is current into real DAN body IDs.

The current Flyppy experiment uses:

```text
gate passage
  -> current into released PAM01 (PAM-gamma5) DANs

collision
  -> current into released PPL1 aversion-associated ensemble
     (PPL101/PPL103/PPL106)
```

There is no `reward=+1`, `punishment=-1`, scalar reward, or direct weight target. Dopamine state and plasticity evolve inside the neural runtime over MaleCNS connectivity after stimulation.

PAM01 may itself be physiologically heterogeneous, so the implementation does not claim that every PAM01 cell represents an identical reward signal in all natural conditions. The same caution applies to the PPL1 ensemble; it is an experimental stimulation condition built from real released DANs.

With PPL101 alone, some observed collision trajectories produced no additional plasticity because pre-event eligibility and its projection targets did not overlap. Expanding to the six-cell PPL101/PPL103/PPL106 ensemble produced event-specific differences on 4,935 plastic edges for the same trajectory. This was adopted as the smallest change that avoided dependence on one compartment without introducing scalar punishment or target weights.

## 5. Preflight checks before learning

`scripts/dev/train_flyppy.sh` regenerates and validates:

1. reward/aversion and diagnostic neural groups;
2. the individual wing-motor-neuron map;
3. the R1-R6 retinotopic map;
4. a smoke test for the wing neuromuscular boundary;
5. Flyppy closed-loop learning inputs.

The old guard based on DNg02 population averages was removed when the learning path moved to the individual motor-neuron boundary.

## 6. Remaining limitations

The current boundary is grounded in data and literature through:

```text
real MaleCNS -> individual motor neurons -> anatomically identified muscles
```

The final step from muscle to FlyBody wing hinge is still a virtual-muscle approximation.

Higher biological fidelity will require muscle-specific attachment points, moment arms, activation-force relationships, and anatomical muscle models inside FlyBody. Unknown values must not be reverse-engineered merely to improve task performance.

## 7. References

- Lesser et al., *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*, eLife, version of record 2026.
  - https://elifesciences.org/articles/96084
- Ehrhardt et al., *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/
- Azevedo et al., *Synaptic architecture of leg and wing premotor control networks in Drosophila*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10312524/
- *How tp1, an indirect wing steering muscle, stabilizes Drosophila's flight*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/
- Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*, Nature 2025.
  - https://www.nature.com/articles/s41586-025-09029-4

## 8. Provenance categories

- `observed`: MaleCNS body IDs, types, sides, and released CNS connectivity.
- `literature`: wing motor neuron-to-muscle targets; classification of power, direct-steering, and indirect-control muscles; asynchronous DLM/DVM flight-muscle properties; qualitative effects of b1/b2/b3/i1.
- `putative`: MNwm35 -> iii4.
- `unknown`: MNwm36, untyped motor neurons, and within-group mapping to individual muscle fibers.
- `calibrated`: peripheral activation time constants, recruitment per spike, thoracic-oscillator coupling, virtual-muscle gains, and torque limits.
