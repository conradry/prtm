import random
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from proteome import protein
from proteome.constants.residue_constants import restypes
from proteome.models.protein_generator import config
from proteome.models.protein_generator.rosettafold_model import RoseTTAFoldModule
from proteome.models.protein_generator.sampler import SeqDiffSampler

PROTGEN_MODEL_URLS = {
    "default": "http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt",
    "t1d_29": "http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_230205_dssp_hotspots_25mask_EQtasks_mod30.pt",
}
PROTGEN_MODEL_CONFIGS = {
    "default": config.BaseConfig,
    "t1d_29": config.ComplexConfig,
}


def _get_model_config(model_name: str) -> config.RoseTTAFoldModuleConfig:
    """Get the model config for a given model name."""
    return PROTGEN_MODEL_CONFIGS[model_name]


def _select_model_from_config(sampler_config: config.InferenceConfig) -> str:
    if (
        sampler_config.hotspot_params.hotspot_res != None
        or sampler_config.secondary_structure_params.secondary_structure != None
        or sum(asdict(sampler_config.structure_bias_params).values()) > 0
        or sampler_config.secondary_structure_params.dssp_structure != None
    ):
        return "t1d_29"
    else:
        return "default"


def _validate_model_for_config(sampler_config: config.InferenceConfig, model_name: str):
    if (
        sampler_config.hotspot_params.hotspot_res != None
        or sampler_config.secondary_structure_params.secondary_structure != None
        or sum(asdict(sampler_config.structure_bias_params).values()) > 0
        or sampler_config.secondary_structure_params.dssp_structure != None
    ):
        assert model_name == "t1d_29"


class ProteinGeneratorForJointDesign:
    def __init__(
        self,
        model_name: str = "auto",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.loaded_model_name = None
        if model_name != "auto":
            self.set_model(model_name)

    def load_weights(self, model_name: str, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"{model_name}.pt",
            progress=True,
            map_location="cpu",
        )["model_state_dict"]
        self.model.load_state_dict(state_dict)

    def set_model(self, model_name: str):
        if self.loaded_model_name == model_name:
            # Use the model that's already loaded
            return

        self.cfg = _get_model_config(model_name)
        self.model = RoseTTAFoldModule(**asdict(self.cfg))

        self.load_weights(model_name, PROTGEN_MODEL_URLS[model_name])
        self.model.eval()
        self.model = self.model.to(self.device)
        self.loaded_model_name = model_name

    def design_structure_and_sequence(
        self,
        inference_config: config.InferenceConfig,
    ) -> Tuple[protein.Protein, str]:
        """Design a protein structure."""
        if self.model_name == "auto":
            self.set_model(_select_model_from_config(inference_config))
        else:
            _validate_model_for_config(inference_config, self.loaded_model_name)

        # Setup the sampler class
        sampler = SeqDiffSampler(
            self.model,
            inference_config,
            pad_t1d_to_29=(self.loaded_model_name in ["t1d_29"]),
        )
        sampler.diffuser_init()
        features = sampler.generate_sample()

        # Decode the sequence and make the structure
        L = len(features["best_seq"])
        num_atoms = features["best_xyz"].shape[2]
        sequence = "".join([restypes[aa] for aa in features["best_seq"].cpu().numpy()])
        structure = protein.Protein(
            atom_positions=features["best_xyz"].squeeze().cpu().numpy(),
            aatype=features["best_seq"].cpu().numpy(),
            atom_mask=np.ones((L, num_atoms)),
            residue_index=np.array([t[1] for t in features["pdb_idx"]]),
            b_factors=features["best_pred_lddt"][0]
            .cpu()
            .numpy()[..., None]
            .repeat(num_atoms, axis=1),
            chain_index=np.array(
                [protein.PDB_CHAIN_IDS.index(t[0]) for t in features["pdb_idx"]]
            ),
        )

        return structure, sequence
