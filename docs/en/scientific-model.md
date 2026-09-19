# Neural-system and plasticity model policy

[English](scientific-model.md) · [日本語](../ja/scientific-model.md) · [简体中文](../zh-CN/scientific-model.md)

## 1. What this project is trying to reconstruct

This project does not treat the connectome as wiring for a conventional artificial neural network.

The goal is to use the adult male Drosophila CNS as the initial structure and continuously evolve neural activity and plasticity over time on a computer.

The connectome is therefore an initial condition, not a finished nervous system state.

```text
C(0) = observed adult male CNS connectome
C(t > 0) = C(0) + functional plastic changes + structural changes
```

## 2. Keep measurements separate from model assumptions

The project must not conflate:

- measured connections with assumed connection strength;
- measured neurotransmitter information with inference for unobserved parts;
- electrophysiological data with parameters introduced for numerical stability;
- real sensory organs with supplementary models needed to drive the CNS.

Where practical, parameters should carry an explicit provenance category.

## 3. Choosing a neural model

We do not assume that the most detailed model is automatically the most correct choice.

Possible models include simple spiking models, conductance-based models, and compartmental models. A candidate model should:

1. be computationally feasible at the intended neuron and synapse count;
2. accept external stimulation as electrical or neural input;
3. support neuromodulation and local plasticity;
4. be replaceable by a more detailed model later;
5. make the information lost through approximation traceable.

The architecture should allow experiments that compare one model across the whole CNS with cell-type-specific models.

## 4. Synapses

Synapses are represented in at least three layers.

### 4.1 Observed structure

- presynaptic and postsynaptic neuron;
- synapse count;
- location information when available;
- connection annotations.

This layer is a snapshot of the upstream data and is not modified.

### 4.2 Functional state

The simulated individual owns mutable state such as:

- effective connection strength;
- transmission state;
- short-term state;
- plasticity eligibility;
- neuromodulator sensitivity.

### 4.3 Structural state

Future versions may model formation and removal of connections.

Changes should be represented as structural deltas owned by the simulated individual rather than by editing the source dataset.

## 5. Plasticity

In this project, "learning" means local activity-dependent change inside the nervous system, not parameter updates produced by an external optimizer.

Conceptually:

```text
presynaptic activity
postsynaptic activity
local synaptic state
neuromodulator state
recent activity history
        ↓
local plasticity dynamics
        ↓
local synaptic change
```

The initial plasticity rule should be replaceable so that literature-based alternatives can be compared.

A central constraint is that Flyppy score or task state must not be read directly and converted into synaptic updates.

## 6. Neuromodulation

Success and failure are not injected into the nervous system as abstract scalar rewards.

The experiment layer converts events into neural stimulation. For example:

```text
successful gate passage
   ↓
stimulation of reward-associated dopaminergic circuitry

collision or failure
   ↓
stimulation of aversion-associated circuitry
```

The stimulated cell groups, amplitude, and duration are explicit experimental conditions chosen from literature and observed data.

PAM, PPL1, and related populations must not be treated as universal fixed "+reward" or "-reward" neurons. Cell type, projection target, and task dependence matter.

## 7. Structural plasticity

The long-term goal includes connection formation and removal as well as changes in synaptic strength.

Because inventing structural rules without biological grounding would undermine the project, structural plasticity is introduced in stages.

### Stage A

Existing connections only; strength and functional state may change.

### Stage B

Existing connections may become functionally inactive and later reactivate.

### Stage C

New connections may form and existing connections may be permanently removed.

Stage C should only be implemented after the biological basis and computational candidate-generation rule are explicit.

## 8. Sensory input

Training data are not injected directly into the CNS as tensors or feature vectors.

For vision, the conceptual path is:

```text
3D physical environment
  ↓
local compound-eye sampling
  ↓
peripheral visual transduction
  ↓
activity in CNS visual-entry neurons
```

Retina and peripheral sensory structures absent from MaleCNS are supplied by separate modules.

## 9. Motor output

The nervous system is not given behavior labels from which it selects an action.

```text
neural activity
  ↓
descending and motor pathways
  ↓
mapping to muscles or physical actuators
  ↓
body motion
```

Even when an early implementation drives body actuators directly, that mapping should represent an anatomical boundary rather than logic that inspects the environment and decides which action is correct.

## 10. Spontaneous activity

The virtual fly should be able to begin moving without an external action command.

The nervous system therefore must not be implemented as a function evaluated only when an input arrives.

Candidate mechanisms include:

- baseline activity;
- spontaneous firing;
- intrinsic neuronal dynamics;
- tonic neuromodulatory drive;
- background sensory input.

Which mechanisms are used should be determined from Drosophila measurements where possible.

The goal is to model a nervous-system state corresponding to spontaneous biological movement, not to hide game-control logic that simply makes the fly move forward.

## 11. Validation

Each layer is evaluated separately rather than by game performance alone.

### Neural level

- activity rate;
- spike/state distributions;
- propagation delays;
- responses of specific circuits;
- changes under neuromodulation.

### Synaptic level

- strength distribution;
- potentiation/depression rates;
- spatial localization of plasticity;
- formation/removal rate once structural plasticity is introduced.

### Behavioral level

- spontaneous movement;
- stimulus response;
- change before and after experience;
- retention;
- extinction;
- generalization.

## 12. Core principle

Do not add hidden AI components merely to obtain behavior that looks more fly-like.

If the result is poor, the project should still report what follows from the implemented neural mechanisms.

The scientific value is not the Flyppy score itself, but what kinds of learning and behavior can arise from an observed neural structure combined with biologically grounded approximations of neural dynamics and plasticity.
