# Requirements

[English](requirements.md) · [日本語](../ja/requirements.md) · [简体中文](../zh-CN/requirements.md)

## 1. Purpose

`virtual-fly` aims to initialize a simulated nervous system from the adult male Drosophila CNS connectome, evolve neural activity and plasticity over time, and connect that nervous system in closed loop with a virtual body, sensory organs, and environment.

The first validation task is a Flappy-Bird-like environment. The experiment asks whether behavior can change through sensory experience and outcome-related neuromodulatory stimulation without an external learning algorithm directly optimizing the nervous system.

## 2. Non-goals

The initial system is not intended to:

- place a conventional artificial neural network or Transformer on top of the connectome;
- optimize the nervous system through backpropagation, gradient descent, Q-learning, policy gradients, or similar methods;
- build a high-scoring game-playing AI for its own sake;
- claim a complete biological reproduction of Drosophila from the start;
- reproduce development from embryo to adult.

## 3. Core concepts

### 3.1 The connectome is an initial condition

Measured connectomes such as MaleCNS `v1.0` are treated as the nervous-system state at `t = 0`.

Connections are not assumed to remain permanently fixed. Synaptic strength, functional effectiveness, and eventually structural state may change through simulated experience.

### 3.2 Learning occurs inside the nervous system

External software does not specify synaptic update values.

External input is limited, in principle, to two categories:

1. stimulation produced by converting the environment and body state into sensory neural activity;
2. neuromodulatory stimulation representing events such as success or failure through biologically relevant circuits.

The nervous system changes through the resulting local neural activity and plasticity.

### 3.3 Behavior originates from the nervous system

The environment must not issue decisions such as `move_up()` or `turn_left()`.

Neural activity propagates into motor pathways, is converted into muscle/joint/wing state, and the resulting body motion changes the environment.

## 4. Functional requirements

### FR-01: MaleCNS ingestion

The system must:

- load a specific release of the adult male whole-CNS connectome;
- support `male-cns:v1.0` as the initial reference dataset;
- preserve neuron IDs, cell types, morphology information where available, synaptic connectivity, connection counts, neurotransmitter information, and annotations;
- record dataset version, acquisition time, and conversion procedure.

### FR-02: time-evolving neural state

- Each neuron has time-dependent internal state.
- External and synaptic input can advance that state.
- Neural computation is treated as a dynamical process rather than a single feed-forward pass over a fixed graph.
- The neural model is replaceable.

### FR-03: dynamic synapses

At minimum, each synapse or aggregated connection can represent mutable:

- effective strength;
- plasticity state;
- active/inactive state;
- state required for future formation/removal.

The first implementation need not enable every mechanism simultaneously, but the architecture must not assume permanently fixed connections.

### FR-04: neuromodulation

The system must support:

- stimulation of reward-associated dopaminergic circuits;
- stimulation of circuits associated with aversive learning;
- no abstract scalar such as `reward` as a direct nervous-system learning input;
- explicit recording of stimulation target, intensity, and temporal pattern as experimental conditions.

### FR-05: sensory transduction

Vision is the first priority.

The system must:

- generate visual input reaching a virtual compound eye from the 3D environment;
- convert that input into neural stimulation at the visual entry boundary of MaleCNS;
- clearly separate supplementary models for peripheral sensory structures absent from MaleCNS.

Future sensory modalities may include olfaction, mechanosensation, and proprioception.

### FR-06: body connection

- FlyBody / FlyGym / MuJoCo are preferred body/physics foundations.
- A layer maps CNS motor output into body actuation.
- That mapping must not contain behavioral decision logic; it should represent biological correspondence.
- The layer should be replaceable by a more detailed muscle model later.

### FR-07: closed-loop environment

The system must continuously cycle through:

1. environment state;
2. sensory transduction;
3. neural activity;
4. motor output;
5. body physics;
6. updated environment state.

### FR-08: Flyppy experiment

The system must be able to:

- generate a flight environment with obstacles;
- update position from physical body motion only;
- convert events such as success and collision into neural stimulation;
- compare behavior before and after experience;
- rerun with fixed environment and stimulation conditions.

### FR-09: state persistence

A virtual fly state should be saveable and resumable with at least:

- dataset and model versions;
- neuron state;
- synaptic state;
- plasticity state;
- neuromodulation state;
- body state;
- experiment time;
- random-number-generator state.

Specific production checkpoint formats may intentionally persist only a documented subset; such semantics must be explicit.

### FR-10: experiment records

Each experiment records:

- initial individual ID;
- derived individual ID;
- code commit SHA;
- input-data version;
- neural-model version;
- plasticity-model version;
- body-model version;
- environment configuration;
- stimulation history;
- key neural statistics;
- behavioral result.

### FR-11: multi-individual experiments

The same initial CNS state can be cloned and branched into multiple individuals with different experience, random seeds, or stimulation conditions.

### FR-12: future browser-distributed execution

The architecture should leave room for:

- WebAssembly / WebGPU execution;
- independent experimental individuals on participant machines;
- result submission to a central service;
- explicit consent for contributed compute;
- view-only mode;
- validation under potentially duplicated, modified, or anomalous client results.

## 5. Non-functional requirements

### NFR-01: scientific provenance

Each value should be classifiable as:

- `observed`;
- `literature`;
- `inferred`;
- `assumed`;
- `calibrated`.

Assumptions must not be presented as measurements.

### NFR-02: reproducibility

Given the same version, initial state, and random sequence, the same experiment should be reproducible within the stated numerical tolerance.

### NFR-03: modularity

The following should be independently replaceable:

- neural dynamics;
- plasticity;
- neuromodulation;
- sensory transduction;
- neural-to-body mapping;
- body model;
- environment;
- persistence format.

### NFR-04: performance

- Correctness should first be testable on small subcircuits.
- The full CNS should use sparse representations rather than large numbers of per-neuron/per-synapse Python objects.
- A CPU reference implementation should remain separable from GPU/WebGPU acceleration.

### NFR-05: observability

During simulation, the state, activity, synaptic change, and neuromodulation of selected neurons or populations should be inspectable.

### NFR-06: separation of external assets

MaleCNS, FlyBody, FlyGym, MuJoCo, and other external assets are not redistributed unconditionally. Their own licenses and terms apply; retrieval scripts, conversion procedures, hashes, and citations are tracked.

## 6. Initial success criteria

The first milestone is not defined by "solving Flappy Bird."

Minimum success means that:

1. some or all of MaleCNS can be converted into a reproducible internal representation;
2. neural activity evolves over time;
3. synaptic state changes through activity and neuromodulation;
4. a bidirectional closed loop with a virtual body is established;
5. behavior with and without plasticity can be compared under the same conditions;
6. if persistent behavioral change occurs, it can be related back to changes in neural and synaptic state.

## 7. Open design questions

Items to be fixed before implementation or per experiment include:

- the neural-dynamics model;
- how initial synaptic strength is derived from connection counts;
- how excitatory/inhibitory and neurotransmitter information are introduced;
- the initial plasticity rule;
- when structural plasticity is introduced;
- how missing peripheral visual structures are modeled;
- the first mapping from CNS motor output to FlyBody;
- the specific cell groups and stimulation patterns used for reward- and aversion-associated stimulation.
