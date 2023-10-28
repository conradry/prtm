import os
from glob import glob
from time import time

import prtm.models.igfold as igfold
import torch
from prtm.models.igfold.antiberty.runner import AntiBERTyRunner
from prtm.models.igfold.model import IgFold
from prtm.models.igfold.utils.embed import embed
from prtm.models.igfold.utils.folding import fold
from prtm.models.igfold.utils.general import exists


def display_license():
    license_url = "https://github.com/Graylab/IgFold/blob/main/LICENSE.md"
    license_message = f"""
    The code, data, and weights for this work are made available for non-commercial use 
    (including at commercial entities) under the terms of the JHU Academic Software License 
    Agreement. For commercial inquiries, please contact awichma2[at]jhu.edu.
    License: {license_url}
    """
    print(license_message)


class IgFoldRunner:
    """
    Wrapper for IgFold model predictions.
    """

    def __init__(self, num_models=4, model_ckpts=None, try_gpu=True):
        """
        Initialize IgFoldRunner.

        :param num_models: Number of pre-trained IgFold models to use for prediction.
        :param model_ckpts: List of model checkpoints to use (instead of pre-trained).
        """

        display_license()

        if exists(model_ckpts):
            num_models = len(model_ckpts)
        else:
            if num_models < 1 or num_models > 4:
                raise ValueError("num_models must be between 1 and 4.")

            if not exists(model_ckpts):
                project_path = os.path.dirname(os.path.realpath(igfold.__file__))

                ckpt_path = os.path.join(
                    project_path,
                    "trained_models/IgFold/*.ckpt",
                )
                model_ckpts = list(glob(ckpt_path))

            model_ckpts = list(sorted(model_ckpts))[:num_models]

        print(f"Loading {num_models} IgFold models...")

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and try_gpu else "cpu"
        )
        print(f"Using device: {device}")

        self.models = []
        for ckpt_file in model_ckpts:
            print(f"Loading {ckpt_file}...")
            self.models.append(IgFold.load_from_checkpoint(ckpt_file).eval().to(device))

        print(f"Successfully loaded {num_models} IgFold models.")

        self.antiberty = AntiBERTyRunner()
        self.antiberty.model.eval()
        self.antiberty.model.to(device)
        print("Loaded AntiBERTy model.")

    def fold(
        self,
        pdb_file,
        fasta_file=None,
        sequences=None,
        template_pdb=None,
        ignore_cdrs=None,
        ignore_chain=None,
        skip_pdb=False,
        do_refine=True,
        use_openmm=False,
        do_renum=True,
        truncate_sequences=False,
    ):
        """
        Predict antibody structure with IgFold.

        :param pdb_file: PDB file to predict.
        :param fasta_file: FASTA file containing sequences.
        :param sequences: Dictionary of sequences.
        :param template_pdb: PDB file containing template structure.
        :param ignore_cdrs: List of CDRs to ignore.
        :param ignore_chain: Chain to ignore.
        :param skip_pdb: Skip PDB processing.
        :param do_refine: Perform PyRosetta refinement.
        :param do_renum: Renumber PDB to Chothia with AbNum.
        :param truncate_sequences: Truncate sequences with AbNumber.
        """
        start_time = time()
        model_out = fold(
            self.antiberty,
            self.models,
            pdb_file=pdb_file,
            fasta_file=fasta_file,
            sequences=sequences,
            template_pdb=template_pdb,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
            skip_pdb=skip_pdb,
            do_refine=do_refine,
            use_openmm=use_openmm,
            do_renum=do_renum,
            truncate_sequences=truncate_sequences,
        )

        print(f"Completed folding in {time() - start_time:.2f} seconds.")

        return model_out

    def embed(
        self,
        model_idx=0,
        fasta_file=None,
        sequences=None,
        template_pdb=None,
        ignore_cdrs=None,
        ignore_chain=None,
    ):
        """
        Embed antibody sequences with IgFold.

        :param fasta_file: FASTA file containing sequences.
        :param sequences: Dictionary of sequences.
        :param template_pdb: PDB file containing template structure.
        :param ignore_cdrs: List of CDRs to ignore.
        :param ignore_chain: Chain to ignore.
        """

        start_time = time()
        model_out = embed(
            self.antiberty,
            self.models[model_idx],
            fasta_file=fasta_file,
            sequences=sequences,
            template_pdb=template_pdb,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
        )

        print(f"Completed embedding in {time() - start_time:.2f} seconds.")

        return model_out
