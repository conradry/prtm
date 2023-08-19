import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
from proteome import protein
from proteome.models.design.protein_generator import config
from proteome.models.design.protein_generator.rosettafold_model import \
    RoseTTAFoldModule
from proteome.models.design.protein_generator.sampler import SeqDiffSampler
from tqdm import tqdm

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
        sampler_config.hotspots != None
        or sampler_config.secondary_structure != None
        or (
            sampler_config.helix_bias
            + sampler_config.strand_bias
            + sampler_config.loop_bias
        )
        > 0
        or sampler_config.dssp_pdb != None
    ):
        return "t1d_29"
    else:
        return "default"


def _validate_model_for_config(sampler_config: config.InferenceConfig, model_name: str):
    if (
        sampler_config.hotspots != None
        or sampler_config.secondary_structure != None
        or (
            sampler_config.helix_bias
            + sampler_config.strand_bias
            + sampler_config.loop_bias
        )
        > 0
        or sampler_config.dssp_pdb != None
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

    def load_weights(self, weights_url: str, model_name):
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

        self.load_weights(PROTGEN_MODEL_URLS[model_name], model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.loaded_model_name = model_name

    def design_structure_and_sequence(
        self,
        inference_config: config.InferenceConfig,
    ) -> protein.Protein:
        """Design a protein structure."""
        if self.model_name == "auto":
            self.set_model(_select_model_from_config(inference_config))
        else:
            _validate_model_for_config(inference_config, self.loaded_model_name)

        # Setup the sampler class
        sampler = SeqDiffSampler(self.model, inference_config)
        sampler.diffuser_init()

        return sampler.generate_sample()
