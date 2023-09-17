import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from proteome import protein
from proteome.constants import residue_constants
from proteome.models.se3_diffusion import config
from proteome.models.se3_diffusion.sampler import Sampler
from proteome.models.se3_diffusion.score_network import ScoreNetwork
from proteome.models.se3_diffusion.se3_diffuser import SE3Diffuser

__all__ = ["SE3DiffusionForStructureDesign"]

SE3_MODEL_URLS = {
    "best": "https://github.com/jasonkyuyim/se3_diffusion/raw/master/weights/best_weights.pth",
    "paper": "https://github.com/jasonkyuyim/se3_diffusion/raw/master/weights/paper_weights.pth",
}
SE3_MODEL_CONFIGS = {
    "best": config.ScoreNetworkConfig(),
    "paper": config.ScoreNetworkConfig(),
}


def _get_model_config(model_name: str) -> config.ScoreNetworkConfig:
    """Get the model config for a given model name."""
    return SE3_MODEL_CONFIGS[model_name]


class SE3DiffusionForStructureDesign:
    def __init__(
        self,
        model_name: str = "best",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)

        self.diffuser = SE3Diffuser(config.DiffuserConfig())
        self.model = ScoreNetwork(self.cfg, self.diffuser)

        self.load_weights(SE3_MODEL_URLS[model_name])
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    @classmethod
    @property
    def available_models(cls):
        return list(SE3_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"se3_diffusion_{self.model_name}.pt",
            progress=True,
            map_location="cpu",
        )["model"]
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def __call__(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[protein.ProteinCATrace, Dict[str, Any]]:
        """Design a random protein structure."""
        sampler = Sampler(self.model, self.diffuser, inference_config.diffusion_params)
        trajectory = sampler.sample(inference_config.length)

        # Trajectory is in reverse order, so take the first frame.
        # Only keep the CA atom
        atom_positions = trajectory["prot_traj"][0, :, :4][:, [1]]
        length, num_atoms, _ = atom_positions.shape

        structure = protein.ProteinCATrace(
            atom_positions=atom_positions,
            aatype=np.array(length * [residue_constants.restype_order_with_x["G"]]),
            atom_mask=np.ones((length, num_atoms)),
            residue_index=np.arange(0, length),
            b_factors=np.zeros((length, num_atoms)),
            chain_index=np.zeros((length,), dtype=np.int32),
        )

        return structure, {}
