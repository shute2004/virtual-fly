# Third-party data, software, and assets

The MIT License in this repository applies to original `virtual-fly` source code and documentation unless a file states otherwise. It does **not** relicense upstream datasets, software, model assets, publications, or third-party material incorporated into generated outputs.

## MaleCNS v1.0

`virtual-fly` uses the released adult male *Drosophila melanogaster* CNS connectome (`male-cns:v1.0`) as an upstream scientific data source.

- Project: https://male-cns.janelia.org/
- Download: https://male-cns.janelia.org/download/
- Dataset used by canonical v1: `male-cns:v1.0`
- Project documentation identifies the released dataset as CC-BY.

The raw MaleCNS dataset is not committed to this Git repository. The canonical reproducer downloads the official source and records source hashes/provenance.

## FlyBody

The physical fly body and related measured flight material are derived from/upstream of the FlyBody project and associated distributed data.

- Repository: https://github.com/TuragaLab/flybody

The FlyBody repository is distributed under the Apache License 2.0. FlyBody source, assets, and separately distributed datasets remain governed by their applicable upstream terms. `virtual-fly` does not relicense them under the repository MIT License.

## FlyGym / NeuroMechFly

`virtual-fly` depends on FlyGym for parts of the fly body/sensory simulation stack.

- Documentation: https://neuromechfly.org/
- Repository/source links are available from the upstream project.
- Canonical dependency lock currently uses FlyGym 2.1.0.

FlyGym is distributed under the Apache License 2.0. Its upstream license and notices continue to apply to FlyGym itself.

## MuJoCo

Physical simulation uses MuJoCo.

- Project: https://mujoco.org/
- Repository: https://github.com/google-deepmind/mujoco

MuJoCo is distributed under the Apache License 2.0 and remains under that upstream license.

## Generated videos and images

Some generated `virtual-fly` media visually incorporates or is derived from third-party body assets/data. Such media should be distributed together with this notice and the relevant upstream attribution rather than assumed to be wholly relicensed by the repository MIT License.

The historical Before/After media intended for GitHub Release distribution is documented by hash and provenance in [`release/release-assets-v0.1.0.json`](release/release-assets-v0.1.0.json).

## Scientific references

Primary scientific references and upstream source links are maintained in [`docs/references.md`](docs/references.md). Experiment-specific machine-readable source hashes and provenance are stored in the canonical/reference manifests.
