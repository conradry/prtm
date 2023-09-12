import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from proteome import protein
from proteome.models.protein_seq_des import config, sampler
from proteome.models.protein_seq_des.models import SeqPred
from proteome.utils.hub_utils import load_state_dict_from_gdrive_zip
from tqdm import tqdm

PSD_MODEL_URLS = {
    "conditional_model_0": "https://drive.google.com/u/0/uc?id=1X66RLbaA2-qTlJLlG9TI53cao8gaKnEt",
    "conditional_model_1": "https://drive.google.com/u/0/uc?id=1X66RLbaA2-qTlJLlG9TI53cao8gaKnEt",
    "conditional_model_2": "https://drive.google.com/u/0/uc?id=1X66RLbaA2-qTlJLlG9TI53cao8gaKnEt",
    "conditional_model_3": "https://drive.google.com/u/0/uc?id=1X66RLbaA2-qTlJLlG9TI53cao8gaKnEt",
}
PSD_MODEL_CONFIGS = {
    "conditional_model_0": config.ConditionalModelConfig(),
    "conditional_model_1": config.ConditionalModelConfig(),
    "conditional_model_2": config.ConditionalModelConfig(),
    "conditional_model_3": config.ConditionalModelConfig(),
}

BASELINE_MODEL_URL = "https://drive.google.com/u/0/uc?id=1X66RLbaA2-qTlJLlG9TI53cao8gaKnEt"
BASELINE_MODEL_CONFIG = config.BaselineModelConfig()


def _get_model_config(model_name: str) -> config.BaselineModelConfig:
    """Get the model config for a given model name."""
    return PSD_MODEL_CONFIGS[model_name]


class ProteinSeqDesForInverseFolding:
    def __init__(
        self,
        model_name: str = "conditional_model_0",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        cond_cfg = _get_model_config(model_name)
        self.model = SeqPred(cfg=cond_cfg)
        self.load_weights(self.model, self.model_name, PSD_MODEL_URLS[model_name])
        self.model.eval()

        init_cfg = BASELINE_MODEL_CONFIG
        self.init_model = SeqPred(cfg=init_cfg)
        self.load_weights(
            self.init_model, "baseline_model", BASELINE_MODEL_URL
        )
        self.init_model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.init_model = self.init_model.to(self.device)
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    @classmethod
    @property
    def available_models(cls):
        return list(PSD_MODEL_CONFIGS.keys())

    def load_weights(self, model, model_name, weights_url: str):
        """Load weights from a weights url."""
        state_dict = load_state_dict_from_gdrive_zip(
            weights_url,
            extract_member=f"models/{model_name}.pt",
            name_prefix="protein_seq_des",
            map_location="cpu",
        )
        msg = model.load_state_dict(state_dict)

    @torch.no_grad()
    def __call__(
        self,
        structure: protein.ProteinBase,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[str, Dict[str, Any]]:
        """Design a protein sequence for a given structure."""
        # Can accept a list of models for ensembling, we only use 1 for now
        # Use atom 14 for compactness, 37 is fine too
        structure = structure.to_protein14()  
        design_sampler = sampler.Sampler(
            inference_config.sampler_config,
            structure,
            [self.model],
            init_model=self.init_model,
        )
        design_sampler.init()
        design_sampler.init_seq()

        # Run the design steps
        best_pose = design_sampler.pose
        best_energy = float("inf")
        energy_calc = inference_config.energy_calculation.value
        for i in tqdm(range(1, inference_config.n_design_iters), desc="Running design"):
            design_sampler.step()

            if getattr(design_sampler, energy_calc) < best_energy:
                best_energy = getattr(design_sampler, energy_calc)
                best_pose = design_sampler.pose

        # Get the best sequence
        sequence = best_pose.sequence()

        return sequence, {"best_energy": best_energy}
