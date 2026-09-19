# Current architecture

[English](architecture.md) · [日本語](architecture.ja.md) · [简体中文](architecture.zh-CN.md)

This document describes the architecture implemented by the current production/canonical path. It is not a roadmap and does not retroactively describe historical artifacts.

## 1. System boundary

`virtual-fly` separates four responsibilities:

1. **nervous system** — MaleCNS dynamics, modulation, eligibility, local plasticity;
2. **sensory/peripheral embodiment** — local physical transduction and motor-neuron-to-muscle state;
3. **body and environment** — FlyBody / MuJoCo and Flyppy geometry;
4. **experiment orchestration** — resets, curriculum conditions, persistence, provenance, reports.

The experiment layer may decide *when* a physical task event occurred and which neural population is stimulated by that event. It does not compute an action target or a desired synaptic weight.

```text
physical scene
    ↓
local sensory transduction
    ↓
MaleCNS neural runtime
    ↓
individual motor-neuron activity
    ↓
peripheral muscle state
    ↓
FlyBody / MuJoCo
    ↓
physical scene
```

## 2. Canonical closed loop

The current canonical Flyppy path is:

```text
Flyppy v7 physical environment
        ↓
FlyBody compound-eye geometry
        ↓
local direct-ray sampling (K=13)
        ↓
released MaleCNS R1-R6 body-ID currents
        ↓
whole-MaleCNS signed neural dynamics
        ↓
local eligibility + dopaminergic modulation
        ↓
individual released motor-neuron spikes
        ↓
WholeBodyPeriphery
        ↓
FlyBody v7 physical actuation
        ↓
MuJoCo state transition
        ↓
Flyppy event detection
        ├─ gate passage → selected reward-associated DAN current
        └─ collision    → selected aversive-associated DAN current
```

There is no external neural network or action policy in this path.

## 3. MaleCNS source and mutable state

The released MaleCNS data is treated as the initial structural snapshot, not as a mutable training file.

```text
immutable source snapshot
        +
mutable functional weight state
        +
short-term neural/plasticity state
        =
current simulated nervous system
```

Canonical v1 uses:

- 166,700 neurons;
- 25,582,938 directed connection edges;
- released neurotransmitter and annotation information;
- class-DAN definition requiring dopamine consensus and released `class=DAN` annotation.

Source files are never modified in place by learning.

## 4. Neural runtime

The large neural runtime is implemented in Rust under `crates/`. Python owns orchestration and body integration; the Rust process owns whole-CNS state evolution.

The current activity semantics distinguish silent, depolarizing activity events, and hyperpolarizing activity deviations so inhibitory/graded pathways are not reduced to a single unsigned spike bit.

Major runtime responsibilities include:

- sparse MaleCNS graph storage;
- state propagation;
- modulation state;
- eligibility dynamics;
- local weight updates;
- shared-weight population transactions;
- checkpoint loading/saving.

GPU execution is a compute backend, not a separate learned controller.

## 5. Local plasticity

The production plasticity path is local. Conceptually, an edge update depends on local eligibility and postsynaptic neuromodulation:

```text
pre/post activity history
        ↓
eligibility
        +
local dopaminergic modulation
        ↓
local bounded synaptic update
```

The environment does not pass `reward = +1`, `reward = -1`, Q-values, target actions, or policy loss into a weight optimizer.

The accelerated PlasticFastGraph implementation reduces which edges must be scanned/updated; it does not substitute a different learning rule.

## 6. Neuromodulatory events

Task events are translated into neural stimulation.

Canonical class-DAN semantics use real released DAN body IDs. The task-specific reward-associated and aversive-associated subsets are experimental stimulation conditions, not a claim that every neuron in those labels has a universal scalar reward meaning.

```text
gate passage
    ↓
current into selected reward-associated DAN subset

collision
    ↓
current into selected aversive-associated DAN subset
```

Canonical frozen evaluation sets both task-triggered DAN currents to zero so stored-weight state can be compared without new task reinforcement.

## 7. Vision boundary

The production vision path does not classify obstacles outside the CNS.

```text
MuJoCo / FlyBody eye geometry
        ↓
per-ommatidium local direct rays
        ↓
local photoreceptor transduction/adaptation
        ↓
corresponding released R1-R6 body IDs
        ↓
MaleCNS visual circuitry
```

Spatial identity is preserved into the CNS. External edge detection, object recognition, motion labels, or global spatial pooling are not used as the production visual answer.

## 8. Haltere boundary

The haltere path derives local mechanical signals from the physical body and maps them to MaleCNS sensory afferents.

Canonical v1 uses the inferred 97-neuron timing subset (46 left / 51 right), current gain 0.05, with `interaction-load-v2` transduction.

This mapping is explicitly an inferred sensory boundary; it must not be described as directly observed complete haltere physiology.

## 9. Motor boundary

Production behavior is not produced from a DNg02 population-average decoder.

The current boundary is:

```text
released motor-neuron body ID activity
        ↓
WholeBodyPeriphery
        ↓
individual motor-unit / muscle activation state
        ↓
versioned FlyBody adapter
        ↓
physical force/torque in MuJoCo
```

The peripheral/body seam contains engineering approximations, but it does not inspect obstacle position, reward state, or desired action.

Historical DNg02 aggregate code remains only as legacy/diagnostic material.

## 10. Body version lineage

Body version numbers are historical labels rather than a simple inheritance sequence:

```text
v3
└─ v4
   ├─ v5
   ├─ v6
   └─ v7

v8 = v6 measured steering + v7 neutral trim
```

Important consequence: **v7 is not "v6 plus improvements"**. Canonical v1 uses v7 because v7 was the pinned canonical condition, not because it subsumes every v5/v6 mechanic.

## 11. Environment v7

Flyppy environment versions are also explicit experimental conditions. Current v7 includes physical side walls so a 3D body cannot count as passing by bypassing the gate laterally.

Gate passage uses full-body geometry around the wall plane rather than treating the thorax center as a zero-size point.

## 12. Population training

The production population trainer uses one shared global weight state.

Each slot:

1. starts an episode from a specific global weight version;
2. accumulates episode-local plasticity operations;
3. finishes independently in `async` mode;
4. rebases its transaction onto the latest global weights;
5. commits to create the next global weight version.

This is **not weight averaging**.

Because completion order can vary, asynchronous shared-weight training is not claimed to be bitwise equivalent to serial single-fly training. Source weight version and staleness are recorded for provenance.

## 13. Curriculum boundary

Curriculum code chooses reset/experience conditions; it does not choose motor actions.

The boundary-band curriculum separates easier/focus experience from full-course evaluation logic. Curriculum success is experiment scheduling state, not a scalar synaptic reward API.

## 14. Checkpoint boundary

Current population persistence is intentionally `global-weights-only-v1`.

Across episodes, learned global weights persist. Episode-local state such as membrane values, spikes, refractory state, activity traces, modulation, and eligibility is reset.

This means a current population checkpoint must **not** be described as a complete persisted biological individual state.

The on-disk checkpoint container still has compatibility arrays for broader neural state; publication claims follow the weights-only semantic contract rather than the raw file layout.

## 15. Frozen evaluation

A publication-quality weight comparison must distinguish:

- plasticity disabled;
- task-triggered DAN stimulation disabled;
- initial vs. final stored weight state;
- identical body/environment/sensory conditions.

Canonical v1 applies these conditions. Older fixed evaluations did not always disable event-triggered DAN currents, which is one reason they remain historical rather than canonical evidence.

## 16. Observer/viewer boundary

The 3D body viewer and neural viewer are observers only.

When no viewer is attached, production training does not need to generate viewer snapshots continuously. Attaching or detaching the viewer must not change neural, physical, plasticity, or curriculum state.

## 17. Current code ownership

```text
src/virtual_fly/embodiment/   body factory, versioned body seams, retina, haltere, periphery
src/virtual_fly/runtime/      neural bridge, body workers, packed processes, telemetry
src/virtual_fly/training/     config, curriculum, population scheduling, checkpoint semantics
src/virtual_fly/playback/     frozen playback/evaluation orchestration
src/virtual_fly/reporting/    stable report/history output
src/virtual_fly/reproducibility.py
                              provenance/hash/semantic validation
src/virtual_fly/semantics.py  compatibility-sensitive semantic identifiers
crates/                       Rust neural runtime and command-line runners
scripts/                      thin launchers, data preparation, analysis, legacy diagnostics
```

See [`code-structure.md`](code-structure.md) for the developer-facing module map.

## 18. Historical code and artifacts

Historical scripts, reports, and artifacts remain useful provenance. Their presence in the repository does not make them part of the current production path.

New production code should depend on `src/virtual_fly/` modules rather than historical script-module compatibility shims.

Historical experiment results must retain their original semantic/provenance labels. See [`results.md`](results.md).

## 19. Not currently implemented as canonical biology

The following should not be inferred from the current architecture:

- complete conductance-level biophysics for every neuron and synapse;
- complete receptor-specific neurotransmitter dynamics;
- validated structural synapse formation/pruning as part of canonical v1;
- complete peripheral sensory physiology;
- complete flight-muscle mechanics;
- complete persistent neural state across training episodes.

These are model boundaries, not hidden components.
