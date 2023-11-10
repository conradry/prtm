import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from prtm import protein
from prtm.constants.residue_constants import proteinmppn_restypes
from prtm.models.pifold import config
from prtm.models.pifold.featurizer import featurize_structure
from prtm.models.pifold.model import PiFoldModel

__all__ = ["PiFoldForInverseFolding"]

PIFOLD_MODEL_URLS = {
    "base": "https://github.com/A4Bio/PiFold/releases/download/Training%26Data/checkpoint.pth",
}
PIFOLD_MODEL_CONFIGS = {
    "base": config.PiFoldConfig(),
}


def _get_pifold_model_config(model_name: str) -> config.PiFoldConfig:
    """Get the model config for a given model name."""
    return PIFOLD_MODEL_CONFIGS[model_name]


class PiFoldForInverseFolding:
    def __init__(
        self,
        model_name: str = "base",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_pifold_model_config(model_name)
        self.model = PiFoldModel(cfg=self.cfg)
        self.load_weights(PIFOLD_MODEL_URLS[model_name])
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
        return list(PIFOLD_MODEL_CONFIGS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
        msg = self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def __call__(
        self,
        structure: protein.ProteinBase,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        """Design a protein sequence for a given structure."""
        # Expects 3 atom protein structure
        structure = structure.to_protein4()
        alphabet = "".join(proteinmppn_restypes)
        X, S, score, mask = featurize_structure(structure, device=self.device)

        (
            X,
            S,
            score,
            h_V,
            h_E,
            E_idx,
            batch_id,
            mask_bw,
            mask_fw,
            decoding_order,
        ) = self.model._get_features(S, score, X=X, mask=mask)
        log_probs, logits = self.model(h_V, h_E, E_idx, batch_id, return_logit=True)

        probs = F.softmax(logits / temperature, dim=-1)
        sequence_indices = torch.multinomial(probs, 1).view(-1)

        sequence = "".join([alphabet[i] for i in sequence_indices])
        return sequence, {"avg_residue_confidence": probs.max(dim=1)[0].mean().item()}
