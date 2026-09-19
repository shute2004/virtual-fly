# Third-party data, software, and assets

The MIT License in this repository applies to original `virtual-fly` source code and documentation unless a file states otherwise. It does **not** relicense upstream datasets, software, model assets, publications, or third-party material incorporated into generated outputs.

## MaleCNS v1.0

`virtual-fly` uses the released adult male *Drosophila melanogaster* CNS connectome (`male-cns:v1.0`) as an upstream scientific data source.

- Project: https://male-cns.janelia.org/
- Download: https://male-cns.janelia.org/download/
- Dataset used by canonical v1: `male-cns:v1.0`
- The official download page links the dataset license to Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

The raw MaleCNS dataset is not committed to this Git repository and is not mirrored into the planned Hugging Face checkpoint repositories. The canonical reproducer downloads the official source and records source hashes/provenance.

Published `virtual-fly` checkpoint weights are derived numerical states based on MaleCNS connectivity. The planned Hugging Face checkpoint repositories therefore use `cc-by-4.0` metadata, preserve MaleCNS attribution, link the official source/license, and indicate that the checkpoint is a transformed/derived artifact.

## FlyBody

The physical fly body and related measured flight material are derived from/upstream of the FlyBody project and associated distributed data.

- Repository: https://github.com/TuragaLab/flybody

The FlyBody **source/model repository** is distributed under the Apache License 2.0.

Canonical v1 also acquires a separately distributed FlyBody supporting dataset:

- Janelia Figshare article: https://janelia.figshare.com/articles/dataset/25309105
- DOI: https://doi.org/10.25378/janelia.25309105
- Version used by canonical v1: 4
- Upstream dataset license shown by Janelia Figshare: **GPL 3.0+**
- Canonical use: `scripts/dev/prefetch_flybody_flight_data.py` obtains the flight-imitation material needed to extract `wing_pattern_fmech.npy`, the measured baseline wing-beat pattern.

The Apache-2.0 license of the FlyBody code repository and the GPL-3.0+ license of this supporting dataset are therefore separate license boundaries. `virtual-fly` does not relicense either under the repository MIT License. The raw supporting dataset is not committed to Git and is not copied into the planned Hugging Face checkpoint repositories; the reproducer obtains it from the official Figshare source.

## FlyGym / NeuroMechFly

`virtual-fly` depends on FlyGym for parts of the fly body/sensory simulation stack.

- Documentation: https://neuromechfly.org/
- Repository/source links are available from the upstream project.
- Canonical dependency lock currently uses FlyGym 2.1.0.

FlyGym is distributed under the Apache License 2.0. Its upstream license and notices continue to apply to FlyGym itself. FlyGym source/assets are not duplicated into the planned Hugging Face checkpoint repositories.

## MuJoCo

Physical simulation uses MuJoCo.

- Project: https://mujoco.org/
- Repository: https://github.com/google-deepmind/mujoco

MuJoCo is distributed under the Apache License 2.0 and remains under that upstream license. MuJoCo source/assets are not duplicated into the planned Hugging Face checkpoint repositories.

## Generated videos and images

Some generated `virtual-fly` media visually incorporates or is derived from third-party body assets/data. Such media should be distributed together with this notice and the relevant upstream attribution rather than assumed to be wholly relicensed by the repository MIT License.

The historical Before/After media intended for GitHub Release distribution is documented by hash and provenance in [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json). The exact historical checkpoints used to generate the Before and After sides are instead prepared for separate Hugging Face repositories; their hashes and provenance are tracked in [`release/huggingface/checkpoint-artifacts-v1.json`](release/huggingface/checkpoint-artifacts-v1.json).

## Scientific references

Primary scientific references and upstream source links are maintained in [`docs/en/references.md`](docs/en/references.md). Experiment-specific machine-readable source hashes and provenance are stored in the canonical/reference manifests.
