# Experimental design

[English](experiments.md) · [日本語](../ja/experiments.md) · [简体中文](../zh-CN/experiments.md)

## 1. Goal

The first research question is whether a plastic virtual nervous system initialized from the adult male CNS can develop persistent, experience-dependent behavioral changes when it receives only sensory input plus neural stimulation corresponding to reward- and aversion-related events.

The Flyppy environment is the first closed-loop task used for this purpose. The goal is not to compete on game-playing performance.

## 2. Staged experiments

### E0: neural-runtime validation

Before connecting a body, verify on a small neural subgraph that:

- stimulation propagates along connections;
- time evolution is numerically stable;
- identical conditions are reproducible;
- plasticity ON and OFF produce the expected difference in synaptic state;
- checkpoints can be resumed continuously.

### E1: large-CNS static execution

Run the whole MaleCNS, or a large portion of it, with plasticity disabled.

Measure:

- memory use and execution speed;
- runaway or extinguished activity;
- population activity statistics;
- locality of stimulation.

This stage does not require fly-like behavior.

### E2: spontaneous activity

Close the loop between nervous system and body without an explicit game command.

Check whether:

- spontaneous motor output appears;
- activity settles into a steady, periodic, silent, or runaway regime;
- background sensory input and baseline drive alter the outcome.

### E3: sensory response

Apply visual stimuli and examine responses from the visual system through motor pathways.

Candidate stimuli include:

- light/dark changes;
- looming stimuli;
- left/right motion;
- optic flow.

Where possible, compare the result with known Drosophila neural and behavioral responses.

### E4: neuromodulation and local plasticity

Before relying on the full body task, test whether pairing sensory input with reward- or aversion-associated stimulation produces persistent neural-response changes.

Check:

- whether the response to the same sensory input changes before and after pairing;
- whether the change persists after stimulation ends;
- whether it differs from a no-neuromodulation control;
- where the altered synapses and circuits are located.

### E5: Flyppy closed loop

Place obstacles in a 3D space and fly the virtual animal through a physical closed loop.

```text
scene
  ↓
visual input
  ↓
CNS
  ↓
motor output
  ↓
FlyBody
  ↓
changed scene
```

Events include successful gate passage and collision. These events are converted into stimulation of corresponding neuromodulatory circuits.

## 3. Initial Flyppy rules

Keep the first task simple:

- a flight corridor extending forward;
- obstacles with openings;
- body position, pose, and velocity determined by physics;
- body motion generated only from CNS motor output;
- gate passage as a candidate success event;
- collision as a candidate failure event.

The priority is a task whose causal chain is easy to inspect, not additional game mechanics.

## 4. Controls

A higher score alone is not sufficient evidence of learning.

At minimum compare:

### C0: no plasticity

Use the same nervous system, body, and environment while disabling synaptic plasticity.

### C1: no neuromodulation

Provide the same sensory experience but no success/failure-triggered neuromodulatory stimulation.

### C2: noncontingent stimulation

Match the number and total amount of reward/aversion stimulation while delivering it at times unrelated to the individual's own outcomes.

### C3: yoked control

Replay individual A's neuromodulatory stimulation history into individual B without linking it to B's own behavior.

This separates "received stimulation" from "received stimulation contingent on my own behavioral outcome."

## 5. Behavioral metrics

Examples include:

- survival time;
- number of gates passed;
- collision rate;
- trajectory changes near obstacles;
- distributions of altitude, speed, and pose;
- repeatability of behavior in the same situation;
- retention across sessions;
- generalization to unseen obstacle layouts.

Do not rely on a single best score as the primary metric.

## 6. Neural metrics

Examples include:

- activity changes in target circuits;
- population-level neural statistics;
- synaptic-strength distribution;
- localization of strengthened and weakened connections;
- response differences before and after neuromodulatory stimulation;
- number of plasticity events;
- formation/removal counts after structural plasticity is introduced;
- changes shared across independently successful individuals.

## 7. Minimum criteria for calling a result a learning candidate

At the initial stage, a result should satisfy all of the following before it is treated as a learning candidate:

1. repeated experience produces a persistent behavioral change;
2. the change is larger than in no-plasticity and no-neuromodulation controls;
3. disrupting contingency between outcome and neuromodulatory stimulation reduces the effect;
4. a persistent internal neural-state change is observed;
5. the effect is statistically reproducible across multiple individuals under the same condition.

These criteria are a practical safeguard against self-deception, not a final philosophical definition of learning.

## 8. Branching-individual experiments

A checkpoint can be cloned into multiple virtual individuals that begin from an identical neural state but receive different experience.

```text
base fly
 ├─ experience A -> fly A1
 ├─ experience A -> fly A2
 ├─ experience B -> fly B1
 └─ control      -> fly C1
```

This enables comparisons that are difficult in biological experiments: identical starting nervous systems with experience as the manipulated variable.

## 9. Public distributed-experiment stage

A future browser-based system may assign an experiment ID and condition to each client.

Candidate returned data include:

- experiment condition;
- initial checkpoint ID;
- final checkpoint summary/delta;
- neural-activity summary;
- plasticity-event summary;
- behavioral sequence;
- result;
- minimal execution-environment information when needed.

The system should not collect identifying information without a research need. Participation in client-side computation should be explicit, and users should be able to view the demonstration without contributing compute.
