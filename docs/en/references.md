# Primary references and external assets

[English](references.md) · [日本語](../ja/references.md) · [简体中文](../zh-CN/references.md)

This document is the entry point for primary literature, official datasets, and major external software used by the implementation.

When an external source is actually used, record not only the URL but also the publication or dataset version, license, and acquisition date in the experiment provenance.

## 1. MaleCNS

### Male CNS Connectome project

- Project: https://male-cns.janelia.org/
- Download / programmatic access: https://male-cns.janelia.org/download/
- Dataset: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/
- Publication supplemental repository: https://github.com/flyconnectome/2025malecns

The official download page provides both `neuprint-python` API access and bulk data downloads. The dataset is distributed under CC-BY.

### Publication

Berg et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature.

- https://www.nature.com/articles/s41586-026-10735-w

This is the primary reference for treating the brain and ventral nerve cord as a continuous CNS connectome.

### MaleCNS optic-lobe retinotopy

The official `assignedOlHex1` / `assignedOlHex2` annotations are used as retinotopic coordinates for optic-lobe columns. The visual-input implementation must not treat arbitrary body-ID order or array index as spatial position.

MaleCNS also publishes media illustrating example visual-motor pathways from R1-R6 through descending neurons.

- https://male-cns.janelia.org/media/

## 2. Early Drosophila visual system

### Complete visual-system connectome

Nern et al. (2025), *Connectome-driven neural inventory of a complete visual system*, Nature.

- https://www.nature.com/articles/s41586-025-08746-0

This work notes that the lamina is not completely contained within the imaged volume and that reconstructed Lai and R1-R6 counts therefore underestimate the biological total. `virtual-fly` consequently does not fabricate missing R1-R6 cells or force every column to contain six cells. Only R1-R6 cells actually released in MaleCNS are treated as `observed`.

### Neural superposition

Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

R1-R6 cells from different neighboring ommatidia that share the same visual axis converge onto the same lamina cartridge. This is an important wiring principle for retinotopic input and is not equivalent to averaging all six cells of one ommatidium into one point.

Juusola et al., electrophysiology protocol overview:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

This provides supporting context for local retinotopic processing in lamina cartridges and histaminergic output from R1-R6 to L1-L3 and related targets.

### Direction selectivity and columnar organization

Fisher et al. (2015/2016), *Complementary mechanisms create direction selectivity in the fly*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

T4/T5 direction selectivity is formed by local circuitry on the optic-lobe columnar array. `virtual-fly` therefore avoids computing a T4/T5-like motion feature externally and injecting the answer downstream; the preferred path is to let it arise from the released visual circuitry beginning at R1-R6.

## 3. FlyBody

- Repository: https://github.com/TuragaLab/flybody
- MuJoCo Menagerie copy: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

FlyBody is a primary source for the anatomical 3D body and MuJoCo-based physical model.

Important: the reinforcement-learning policies bundled with FlyBody are not used as the learning mechanism for `virtual-fly`. The project uses body geometry, physics, actuators, and related assets while motor commands originate from the virtual CNS.

## 4. FlyGym / NeuroMechFly

- Documentation: https://neuromechfly.org/
- Installation: https://neuromechfly.org/installation/
- Tutorials: https://neuromechfly.org/tutorials/

These projects are reference implementations for sensory integration, body simulation, MuJoCo integration, FlyBody use, and GPU execution.

FlyGym 2.x is not API-compatible with older versions, so dependency versions must be pinned explicitly.

FlyGym's standard compound-eye `Retina` uses a synthetic hexagonal grid. It is therefore not treated as the actual body-ID-to-retinal-location mapping for a MaleCNS individual. The current MaleCNS boundary uses FlyBody's raw eye camera as a local light source while prioritizing the observed MaleCNS column coordinates.

## 5. MuJoCo

- Project: https://mujoco.org/
- Repository: https://github.com/google-deepmind/mujoco
- Menagerie: https://github.com/google-deepmind/mujoco_menagerie

MuJoCo is the primary physics-engine candidate.

## 6. Accessing MaleCNS

A typical official Python access pattern is:

```python
from neuprint import Client

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token="...",
)
```

Authentication tokens must never be committed to source or configuration files.

## 7. Primary literature still to organize

Before adding or refining mechanisms, primary Drosophila literature should be collected for at least:

- membrane potential and firing properties;
- chemical synaptic transmission;
- electrical synapses;
- neurotransmitters and receptors;
- mushroom-body plasticity;
- PAM / PPL1 and related dopaminergic circuits;
- appetitive and aversive conditioning;
- activity-dependent plasticity such as STDP;
- homeostatic plasticity;
- structural plasticity;
- spontaneous locomotion;
- flight central pattern generation and descending control;
- wing motor neurons and flight muscles;
- compound-eye phototransduction;
- proprioception and mechanosensation.

Implementation parameters should prefer Drosophila-specific evidence rather than borrowing generic neuroscience values solely because they are convenient.

## 8. Rules for using references

1. Prefer primary papers and official data over blogs or explanatory articles.
2. Values borrowed from another species must be labeled `assumed` or `inferred` as appropriate.
3. Do not automatically follow dataset updates; pin the version per experiment.
4. Keep the measured connectome separate from mutable simulated-individual state.
5. Do not silently substitute reinforcement-learning or neural-network controllers bundled with external software for the CNS.
6. Do not replace feature extraction performed by real local circuitry with convenient external feature engineering.
