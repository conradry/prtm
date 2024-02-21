# prtm

**Pr**o**t**ein **M**odels (prtm) is an inference-only library for deep learning protein models.


## Background

This library started out as a learning project to catch up on the deep learning models being
used in protein science. After cloning a few repos it became clear that a nascent ecosystem was
forming and that there was a need for a common interface to accelerate the creation of new workflows. 
The goal of `prtm` is to provide an (hopefully) enjoyable and interactive API for running, comparing, and 
chaining together protein DL models. Currently covered use cases include:

- Folding
- Inverse Folding
- Structure design
- Sequence language modeling

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

**Note**: Most, but not all models, allow commerial use. Please check the license of each model.

AlphaFold is written and JAX but all other models are written in PyTorch, therefore we chose not
to directly integrate the AlphaFold inference code into this repo. Both `OpenFold` and `Uni-Fold`
allow for the conversion of the AlphaFold JAX weights into PyTorch. The `Uni-Fold` implementation
is designed to work with `MMSeqs2` and has support for multimers which is why we adopted it. Eventually,
we may decide to subsume the `OpenFold` models under `Uni-Fold`.

| Model Name | Function | Notebook | Source Code | License |
|------------|----------|----------|-------------|---------|
| AlphaFold/Uni-Fold | Folding | [Notebook](./notebooks/model_notebooks/Uni-Fold.ipynb) | https://github.com/dptech-corp/Uni-Fold | [Apache 2.0](https://github.com/dptech-corp/Uni-Fold/blob/main/LICENSE) |
| AlphaFold/UniFold-Multimer | Folding | [Notebook](./notebooks/model_notebooks/Uni-Fold.ipynb) | https://github.com/dptech-corp/Uni-Fold | [Apache 2.0](https://github.com/dptech-corp/Uni-Fold/blob/main/LICENSE) |
| OpenFold | Folding | [Notebook](./notebooks/model_notebooks/OpenFold.ipynb) | https://github.com/aqlaboratory/openfold | [Apache 2.0](https://github.com/aqlaboratory/openfold/blob/main/LICENSE) |
| ESMFold | Folding |[Notebook](./notebooks/model_notebooks/ESMFold.ipynb) | https://github.com/facebookresearch/esm | [MIT License](https://github.com/facebookresearch/esm/blob/main/LICENSE) |
| RoseTTAFold| Folding | [Notebook](./notebooks/model_notebooks/RoseTTAFold.ipynb) | https://github.com/RosettaCommons/RoseTTAFold | [MIT License](https://github.com/RosettaCommons/RoseTTAFold/blob/main/LICENSE) |
| OmegaFold | Folding | [Notebook](./notebooks/model_notebooks/OmegaFold.ipynb) | https://github.com/HeliXonProtein/OmegaFold | [Apache 2.0](https://github.com/HeliXonProtein/OmegaFold/blob/main/LICENSE) |
| DMPfold2 | Folding | [Notebook](./notebooks/model_notebooks/DMPfold.ipynb) | https://github.com/psipred/DMPfold2 | [GPL v3.0](https://github.com/psipred/DMPfold2/blob/master/LICENSE) |
| Uni-Fold Symmetry | Folding | [Notebook](./notebooks/model_notebooks/Uni-Fold.ipynb) | https://github.com/dptech-corp/Uni-Fold | [GPL v3.0](https://github.com/dptech-corp/Uni-Fold/blob/main/LICENSE) |
| IgFold | Antibody Folding | [Notebook](./notebooks/model_notebooks/IgFold.ipynb) | https://github.com/Graylab/IgFold | [JHU License](https://github.com/Graylab/IgFold/blob/main/LICENSE.md) |
| ESM-IF | Inverse Folding | [Notebook](./notebooks/model_notebooks/ESM-IF.ipynb) | https://github.com/facebookresearch/esm | [MIT License](https://github.com/facebookresearch/esm/blob/main/LICENSE) |
| ProteinMPNN| Inverse Folding | [Notebook](./notebooks/model_notebooks/ProteinMPNN.ipynb) | https://github.com/dauparas/ProteinMPNN | [MIT License](https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE) |
| PiFold | Inverse Folding | [Notebook](./notebooks/model_notebooks/PiFold.ipynb) | https://github.com/A4Bio/PiFold | [MIT License](https://github.com/A4Bio/PiFold/blob/main/license) |
| ProteinSeqDes| Inverse Folding| [Notebook](./notebooks/model_notebooks/ProteinSeqDes.ipynb) | https://github.com/nanand2/protein_seq_des | [BSD-3](https://github.com/nanand2/protein_seq_des/blob/master/LICENSE) |
| ProteinSolver| Inverse Folding| [Notebook](./notebooks/model_notebooks/ProteinSolver.ipynb) | https://github.com/ostrokach/proteinsolver | [MIT License](https://github.com/ostrokach/proteinsolver/blob/master/LICENSE) |
| RFDiffusion | Design | [Notebook](./notebooks/model_notebooks/RFDiffusion.ipynb) | https://github.com/RosettaCommons/RFdiffusion | [BSD](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE) |
| ProteinGenerator | Design | [Notebook](./notebooks/model_notebooks/ProteinGenerator.ipynb) | https://github.com/RosettaCommons/protein_generator | [MIT License](https://github.com/RosettaCommons/protein_generator/blob/main/LICENSE) |
| Genie | Design | [Notebook](./notebooks/model_notebooks/Genie.ipynb) | https://github.com/aqlaboratory/genie | [Apache 2.0](https://github.com/aqlaboratory/genie/blob/main/LICENSE.md) |
| FoldingDiff | Design | [Notebook](./notebooks/model_notebooks/FoldingDiff.ipynb) | https://github.com/microsoft/foldingdiff | [MIT License](https://github.com/microsoft/foldingdiff/blob/main/LICENSE) |
| SE3-Diffusion | Design | [Notebook](./notebooks/model_notebooks/SE3Diffusion.ipynb) | https://github.com/jasonkyuyim/se3_diffusion | [MIT License](https://github.com/jasonkyuyim/se3_diffusion/blob/master/LICENSE) |
| EigenFold | Fold sampling | [Notebook](./notebooks/model_notebooks/EigenFold.ipynb) | https://github.com/bjing2016/EigenFold | [MIT License](https://github.com/bjing2016/EigenFold/blob/master/LICENSE) |
| AntiBERTy | Antibody language modeling | [Notebook](./notebooks/model_notebooks/AntiBERTy.ipynb) | https://github.com/jeffreyruffolo/AntiBERTy | [MIT License](https://github.com/jeffreyruffolo/AntiBERTy/blob/main/LICENSE.MD) |

Links for papers can be found on the Github repos for each model.

## Documentation

A real docs page is a work in progress, but to get started the provided notebooks should be enough.
In addition to minimal usage notebooks for each implemented model, there are also more general notebooks
that cover common use cases and some features of the `prtm` API. A good order to try is:

- [protein.ipynb](./notebooks/protein.ipynb)
- [folding.ipynb](./notebooks/folding.ipynb)
- [inverse_folding.ipynb](./notebooks/inverse_folding.ipynb)
- [unconditional_design.ipynb](./notebooks/unconditional_design.ipynb)

For more complex design algorithms like `RFDiffusion` and `ProteinGenerator`, there are detailed
example notebooks to look at:

- [RFDiffusion.ipynb](./notebooks/model_notebooks/RFDiffusion.ipynb)
- [ProteinGenerator.ipynb](./notebooks/model_notebooks/ProteinGenerator.ipynb)

## Roadmap and Contributing

The currently implemented models only scratch the surface of what's available. There's a sketchy [model tracking Google sheet](https://docs.google.com/spreadsheets/d/1iMhFXJnUU16ycRVcEvXi8jSQsZ0qsfwQerOgaosBl-E/edit#gid=0) for papers and code repos that are being considered for implementation. If you'd like to contribute or suggest priorities, please open an issue or PR and we can discuss!

There's, of course, also a lot of technical debt to payoff that accumulated from duct taping together code from many different sources. Docstrings, API improvements, bug fixes, and better tests are very welcome!


## Acknowledgments

This project is an achievement of copy-paste engineering :wink:. It would not have been possible without the hard work of the authors of the models that are implemented here. Please cite their work if you use their model!
