# Scientific references and evidence map

[English](references.md) · [日本語](../ja/references.md) · [简体中文](../zh-CN/references.md)

`virtual-fly` is built from published connectomics, neurobiology, electrophysiology, motor-control, and biomechanics rather than from a single source. This page collects the primary literature and official upstream resources that materially informed the current implementation.

The table below is an **evidence map**: it records not only what was read, but what each source was used to justify. It is a curated implementation-facing bibliography, not an exhaustive list of every source consulted during development. A citation here does not imply that every numerical parameter in `virtual-fly` was directly measured in that paper; inferred, assumed, and calibrated boundaries remain labeled separately in the implementation and experiment provenance.

## Core literature used by the current implementation

| Reference | Status | Area | What it informs in `virtual-fly` |
|---|---|---|---|
| Bates et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature | Peer-reviewed | MaleCNS | Adult male brain-and-cord connectome as the structural starting point; CNS-wide connectivity and annotations |
| Nern et al. (2025), *Connectome-driven neural inventory of a complete visual system*, Nature | Peer-reviewed | Vision | Optic-lobe organization; limitations of reconstructed lamina/R1-R6 coverage; avoiding invented missing photoreceptors |
| Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*, Cell | Peer-reviewed | Vision | Neural superposition and the relationship between neighboring ommatidia, R1-R6, and lamina cartridges |
| Juusola et al. (2016), *Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo*, JoVE | Peer-reviewed | Vision / electrophysiology | R1-R6 and lamina physiology; local photoreceptor/interneuron responses to light |
| Haag et al. (2016), *Complementary mechanisms create direction selectivity in the fly*, eLife | Peer-reviewed | Vision | Direction selectivity as a computation emerging in local optic-lobe circuitry rather than an external motion classifier |
| Lesser et al. (2024), *Synaptic architecture of leg and wing premotor control networks in Drosophila*, Nature | Peer-reviewed | Motor control | Wing premotor modules, motor-neuron recruitment structure, and the distinction between leg and wing premotor organization |
| Cheong et al. (2026), *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*, eLife | Peer-reviewed, version of record | Motor control | Descending-to-premotor-to-motor organization in the male adult nerve cord |
| Ehrhardt et al. (2025), *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster* | Preprint | Motor control | Cell-type-level wing premotor circuitry and motor-neuron/muscle interpretation |
| Teoh et al. (2025), *How tp1, an indirect wing steering muscle, stabilizes Drosophila’s flight* | Preprint | Flight control | Role of the indirect steering muscle tp1 in stabilization and wing-hinge mechanics |
| Vaxenburg et al. (2025), *Whole-body physics simulation of fruit fly locomotion*, Nature | Peer-reviewed | Body / physics | FlyBody whole-body geometry, MuJoCo embodiment, flight and locomotion physics |

## 1. MaleCNS and the adult male CNS connectome

### Bates et al. (2026)

**Bates, A. S., Phelps, J. S., Kim, M. et al.** *Distributed control circuits across a brain-and-cord connectome*. Nature 656, 957–970 (2026).

- DOI: https://doi.org/10.1038/s41586-026-10735-w
- Article: https://www.nature.com/articles/s41586-026-10735-w
- Project: https://male-cns.janelia.org/
- Download / programmatic access: https://male-cns.janelia.org/download/
- Dataset used by this project: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/
- Publication supplemental repository: https://github.com/flyconnectome/2025malecns

This is the principal scientific source for treating the adult male brain and ventral nerve cord as a continuous CNS connectome. `virtual-fly` uses the released MaleCNS data as the `t = 0` structural state, while learned functional changes are stored separately.

The official `assignedOlHex1` / `assignedOlHex2` annotations are also used when resolving optic-lobe retinotopy. MaleCNS media provide additional visualizations of example visual-motor pathways.

- https://male-cns.janelia.org/media/

## 2. Compound eye and optic-lobe circuitry

### Nern et al. (2025)

**Nern, A., Loesche, F., Takemura, S.-y. et al.** *Connectome-driven neural inventory of a complete visual system*. Nature 641, 1225–1237 (2025).

- DOI: https://doi.org/10.1038/s41586-025-08746-0
- Article: https://www.nature.com/articles/s41586-025-08746-0

This work is used to interpret the visual-system connectome and its coverage limits. In particular, incomplete lamina/R1-R6 reconstruction is treated as a data boundary rather than an invitation to fabricate missing photoreceptors.

### Langen et al. (2015)

**Langen, M., Agi, E. et al.** *The Developmental Rules of Neural Superposition in Drosophila*. Cell 162, 120–133 (2015).

- DOI: https://doi.org/10.1016/j.cell.2015.05.055
- Open article: https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

This is the key reference for neural superposition: R1-R6 photoreceptors from neighboring ommatidia that share a visual axis converge onto the same lamina cartridge. This supports preserving local retinotopic identity rather than collapsing one ommatidium to a single averaged signal.

### Juusola et al. (2016)

**Juusola, M., Dau, A., Zheng, L. & Rien, D.** *Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo*. Journal of Visualized Experiments, issue 112, 54142 (2016).

- DOI: https://doi.org/10.3791/54142
- Open article: https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

This provides electrophysiological context for R1-R6 photoreceptors and lamina interneurons responding locally to light stimuli.

### Haag et al. (2016)

**Haag, J., Arenz, A., Serbe, E., Gabbiani, F. & Borst, A.** *Complementary mechanisms create direction selectivity in the fly*. eLife 5, e17421 (2016).

- DOI: https://doi.org/10.7554/eLife.17421
- Open article: https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

T4/T5 direction selectivity emerges from local circuitry on the optic-lobe columnar array. The current visual boundary therefore avoids calculating a T4/T5-like direction label outside the CNS and injecting that answer downstream.

## 3. Wing premotor and motor circuitry

### Lesser et al. (2024)

**Lesser, E., Azevedo, A. W., Phelps, J. S. et al.** *Synaptic architecture of leg and wing premotor control networks in Drosophila*. Nature 631, 369–377 (2024).

- DOI: https://doi.org/10.1038/s41586-024-07600-z
- Article: https://www.nature.com/articles/s41586-024-07600-z

This paper supports the organization of wing premotor networks into motor modules and the treatment of wing motor control as a structured premotor-to-motor system rather than a population-average action decoder.

### Cheong et al. (2026)

**Cheong, H. S. J., Eichler, K., Stürner, T. et al.** *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*. eLife, version of record (2026).

- DOI: https://doi.org/10.7554/eLife.96084.3
- Article: https://elifesciences.org/articles/96084

This is a principal reference for tracing descending input through premotor circuitry to motor output in the male adult nerve cord.

### Ehrhardt et al. (2025 preprint)

**Ehrhardt, E., Whitehead, S. C. et al.** *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster*. bioRxiv preprint, version 3 (2025).

- DOI: https://doi.org/10.1101/2023.05.31.542897
- Open record: https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/

This preprint is used as supporting evidence for cell-type-level wing premotor organization and the interpretation of identified wing motor pathways. It is explicitly treated as a preprint rather than as a peer-reviewed final article.

## 4. Flight muscles, steering, and whole-body mechanics

### Teoh et al. (2025 preprint)

**Teoh, H. K., Biswas, D., Leung, A. et al.** *How tp1, an indirect wing steering muscle, stabilizes Drosophila’s flight*. bioRxiv preprint, version 2 (2025).

- DOI: https://doi.org/10.1101/2025.11.02.686144
- Open record: https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/

This preprint is used when interpreting tp1 as an indirect steering/tension muscle contributing to flight stabilization and wing-hinge mechanics.

### Vaxenburg et al. (2025)

**Vaxenburg, R., Siwanowicz, I., Merel, J. et al.** *Whole-body physics simulation of fruit fly locomotion*. Nature 643, 1312–1320 (2025).

- DOI: https://doi.org/10.1038/s41586-025-09029-4
- Article: https://www.nature.com/articles/s41586-025-09029-4
- FlyBody repository: https://github.com/TuragaLab/flybody
- MuJoCo Menagerie copy: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

This is the primary scientific reference for the anatomically detailed FlyBody model and its MuJoCo-based whole-body physics. `virtual-fly` uses the body and physics infrastructure, not the reinforcement-learning policies used in the paper to demonstrate locomotion.

## 5. Major upstream software and official resources

These are not substitutes for the primary literature above, but they are important implementation dependencies and official data sources.

- **MaleCNS** — https://male-cns.janelia.org/
- **neuPrint** — https://neuprint.janelia.org/
- **FlyBody** — https://github.com/TuragaLab/flybody
- **FlyGym / NeuroMechFly documentation** — https://neuromechfly.org/
- **MuJoCo** — https://mujoco.org/
- **MuJoCo repository** — https://github.com/google-deepmind/mujoco
- **MuJoCo Menagerie** — https://github.com/google-deepmind/mujoco_menagerie

FlyGym's standard compound-eye `Retina` uses a synthetic hexagonal grid, so it is not treated as the observed MaleCNS body-ID-to-retinal-location mapping. The current sensory boundary instead prioritizes MaleCNS retinotopic annotations and released connectivity.

## 6. Citation and provenance rules

1. Prefer primary papers and official datasets over secondary summaries.
2. Record which implementation boundary a paper actually supports; do not cite a paper as blanket validation of unrelated parameters.
3. Distinguish peer-reviewed articles, preprints, official datasets, and software documentation.
4. Values not directly measured in the cited literature remain labeled `inferred`, `assumed`, or `calibrated` as appropriate.
5. Pin dataset and software versions per experiment instead of silently following upstream updates.
6. Keep measured source data separate from mutable simulated-individual state.
7. Do not silently substitute reinforcement-learning or neural-network controllers bundled with external software for the virtual CNS.
