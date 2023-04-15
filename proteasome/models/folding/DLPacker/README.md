<br/>
<h1 align="center">DLPacker</h1>
<br/>

<br/>

This repo contains the code from DLPacker paper [DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins](https://doi.org/10.1101/2021.05.23.445347).
<br/>
<p align="center">
    <img width="70%" src="https://github.com/nekitmm/DLPacker/blob/main/img5v3.png" alt="Side chain restroration example">
</p>
<br/>

<h2 align="center">What can this code do?</h2>

* Restore full-atom protein structure from backbone (packing)
* Generate structures of point mutants (assumes the backbone has not changed)
* Pack or refine parts of protein structure (e.g. after you modelled backbone of a missing loop)
* Restore partially of fully missing side chains (to be implemented)
* *probably more*


Input may contain any protein/protein complex/RNA/DNA/small molecules etc. Only water molecules are removed by default and MSE residues are renamed into MET, the rest will stay the same (except side chains of course).


<h2 align="center">Usage</h2>

As easy as three lines of code:

```python
from dlpacker import DLPacker
dlp = DLPacker('my_structure.pdb')
dlp.reconstruct_protein(order = 'sequence', output_filename = 'my_structure_repacked.pdb')
```

Input stricture might or might not contain side chains, existing side chains, if present, will be ignored.

You can find more examples with explanations in the jupyter notebook **[DLPacker.ipynb](DLPacker.ipynb)**.

**Key dependencies:**
* tensorflow 2.x
* biopython

<h2 align="center" style="color:blue">Additional files</h2>

This repo does not contain neural network's weights due to their large size. You will need to download them from [here](https://drive.google.com/file/d/1J4fV9aAr2nssrWN8mQ7Ui-9PVQseE0LQ/view?usp=sharing).

<h2 align="center">Performance</h2>

The table below shows validation RMSD (Å) for DLPacker as well as two other state of the art algorithms, SCWRL4 and Rosetta Packer (*fixbb*):

|AA Name | SCWRL4 | Rosetta Packer | DLPacker|
|:------:|:------:|:--------------:|:-------:|
|  Arg   |  2.07  |      1.84      |   1.44  |
|  Lys   |  1.54  |      1.40      |   1.21  |
|  Phe   |  0.67  |      0.53      |   0.32  |
|  Tyr   |  0.83  |      0.68      |   0.38  |
|  Trp   |  1.27  |      0.96      |   0.46  |
|  His   |  1.18  |      1.05      |   0.81  |
|  Glu   |  1.34  |      1.26      |   1.02  |
|  Gln   |  1.43  |      1.24      |   1.09  |
|  Met   |  1.08  |      0.91      |   0.76  |
|  Asn   |  0.88  |      0.80      |   0.65  |
|  Asp   |  0.68  |      0.65      |   0.47  |
|  Ser   |  0.59  |      0.52      |   0.36  |
|  Leu   |  0.49  |      0.45      |   0.36  |
|  Thr   |  0.36  |      0.33      |   0.27  |
|  Ile   |  0.40  |      0.36      |   0.31  |
|  Cys   |  0.40  |      0.30      |   0.24  |
|  Val   |  0.24  |      0.23      |   0.19  |
|  Pro   |  0.21  |      0.19      |   0.14  |

<h2 align="center">Citing our work</h2>

If you use our code in your work, please cite the DLPacker paper:

	@article {Misiura2021.05.23.445347,
		author = {Misiura, Mikita and Shroff, Raghav and Thyer, Ross and Kolomeisky, Anatoly},
		title = {DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins},
		elocation-id = {2021.05.23.445347},
		year = {2021},
		doi = {10.1101/2021.05.23.445347},
		publisher = {Cold Spring Harbor Laboratory},
		URL = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347},
		eprint = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347.full.pdf},
		journal = {bioRxiv}
	}
