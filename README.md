# prtm

**Pr**o**t**ein **M**odels (prtm) is a Python package for deep learning protein models.


## Background

This library started out as a learning project to catch up on the deep learning models being
used in protein science. After cloning a few repos it became clear that a nascent ecosystem was
forming and that there was a need for a common interface to accelerate the creation of new workflows. 
The goal of `prtm` is to provide an (hopefully) enjoyable and interactive API for running, comparing, and 
chaining together protein DL models. Currently covered use cases include:

- Folding
- Inverse Folding
- Structure design

With many more to come!

## Motivating Example

A very common workflow is to design a protein structure, apply inverse folding to generate
plausible sequences, and then fold those sequences to see if they match the designed structure.

In `prtm`, we accomplish this with a few lines of code:

```python
from prtm import models
from prtm import visual

# Define models for structure design, inverse folding and folding
designer = models.RFDiffusionForStructureDesign(model_name="auto")
inverse_folder = models.ProteinMPNNForInverseFolding(model_name="ca_only_model-20")
folder = models.OmegaFoldForFolding()

# Tell RFDiffusion to create a structure with exactly 128 residues
designed_structure, _ = designer(
    models.rfdiffusion_config.UnconditionalSamplerConfig(
        contigmap_params=models.rfdiffusion_config.ContigMap(contigs=["128-128"]),
    )
)

# Design a sequence and fold it!
designed_sequence, _ = inverse_folder(designed_structure)
predicted_designed_structure, _ = folder(designed_sequence)

# Visualize the designed structure and the predicted structure overlaid in a notebook
visual.view_superimposed_structures(designed_structure, predicted_designed_structure)

# Convert to PBD
pdb_str = predicted_designed_structure.to_pdb()

```

## Installation

At this early stage, `prtm` has only been tested on a Linux system with a CUDA-enabled GPU.
There are no guarantees that it will work on other systems.

Before getting started it's assumed that you've already installed `conda` or `mamba` (preferred), 
then clone this repo and create a `prtm` environment:

```bash
git clone https://github.com/conradry/prtm.git
cd prtm
mamba env create -f environment.yaml
mamba activate prtm
pip install -e .
```

To make `prtm` more accessible it was decided to remove custom CUDA kernels from all models that
previously used them, so that's it for most cases!

Optionally, `Pyrosetta` is a soft-dependency of `prtm` and is only required for the
`protein_seq_des` model. A license is required to use `Pyrosetta` and can 
be obtained for free for academic use. For installation instructions, see 
[here](https://www.pyrosetta.org/downloads#h.6vttn15ac69d).

## What's implemented

| Model Name | Function | Source Code | License |
|------------|----------|-------------|---------|
| OpenFold | Folding | https://github.com/aqlaboratory/openfold | [Apache 2.0](https://github.com/aqlaboratory/openfold/blob/main/LICENSE) |
| ESMFold | Folding | https://github.com/facebookresearch/esm | [MIT License](https://github.com/facebookresearch/esm/blob/main/LICENSE) |
| RoseTTAFold| Folding | https://github.com/RosettaCommons/RoseTTAFold | [MIT License](https://github.com/RosettaCommons/RoseTTAFold/blob/main/LICENSE) |
| OmegaFold | Folding | https://github.com/HeliXonProtein/OmegaFold | [Apache 2.0](https://github.com/HeliXonProtein/OmegaFold/blob/main/LICENSE) |
| DMPfold2 | Folding | https://github.com/psipred/DMPfold2 | [GPL v3.0](https://github.com/psipred/DMPfold2/blob/master/LICENSE) |
| ESM-IF | Inverse Folding | https://github.com/facebookresearch/esm | [MIT License](https://github.com/facebookresearch/esm/blob/main/LICENSE) |
| ProteinMPNN| Inverse Folding | https://github.com/dauparas/ProteinMPNN | [MIT License](https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE) |
| ProteinSeqDes| Inverse Folding| https://github.com/nanand2/protein_seq_des | [BSD-3](https://github.com/nanand2/protein_seq_des/blob/master/LICENSE) |
| ProteinSolver| Inverse Folding| https://github.com/ostrokach/proteinsolver | [MIT License](https://github.com/ostrokach/proteinsolver/blob/master/LICENSE) |
| RFDiffusion | Design | https://github.com/RosettaCommons/RFdiffusion | [BSD](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE) |
| ProteinGenerator | Design | https://github.com/RosettaCommons/protein_generator | [MIT License](https://github.com/RosettaCommons/protein_generator/blob/main/LICENSE) |
| Genie | Design | https://github.com/aqlaboratory/genie | [Apache 2.0](https://github.com/aqlaboratory/genie/blob/main/LICENSE.md) |
| FoldingDiff | Design | https://github.com/microsoft/foldingdiff | [MIT License](https://github.com/microsoft/foldingdiff/blob/main/LICENSE) |
| SE3-Diffusion | Design | https://github.com/jasonkyuyim/se3_diffusion | [MIT License](https://github.com/jasonkyuyim/se3_diffusion/blob/master/LICENSE) |
| EigenFold | Fold sampling | https://github.com/bjing2016/EigenFold | [MIT License](https://github.com/bjing2016/EigenFold/blob/master/LICENSE) |

Links for papers can be found on the Github repos for each model.

## Documentation

A real docs page is a work in progress, but to get started the provided notebooks should be enough.
In addition to minimal usage notebooks for each implemented model, there are also more general notebooks
that cover common use cases and some features of the `prtm` API. A good order to try is:

- [protein.ipynb](./notebooks/protein.ipynb)
- [folding.ipynb](./notebooks/folding.ipynb)
- [inverse_folding.ipynb](./notebooks/inverse_folding.ipynb)
- [unconditional_design.ipynb](./notebooks/unconditional_design.ipynb)

## Roadmap and Contributing

The currently implemented models only scratch the surface of what's available. There's a sketchy [model tracking Google sheet](https://docs.google.com/spreadsheets/d/1iMhFXJnUU16ycRVcEvXi8jSQsZ0qsfwQerOgaosBl-E/edit#gid=0) for papers and code repos that are being considered for implementation. If you'd like to contribute or suggest priorities, please open an issue or PR and we can discuss!

There's, of course, also a lot of technical debt to payoff that accumulated from duct taping together code from many different sources. Docstrings, API improvements, bug fixes, and better tests are very welcome!


## Acknowledgments

This project is an achievement of copy-paste engineering :wink:. It would not have been possible without the hard work of the authors of the models that are implemented here. Please cite their work if you use their model!