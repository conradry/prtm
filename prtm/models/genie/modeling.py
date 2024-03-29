import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from prtm import protein
from prtm.constants import residue_constants
from prtm.models.genie import config
from prtm.models.genie.diffusion import Genie

__all__ = ["GenieForStructureDesign"]

GENIE_MODEL_URLS = {
    "genie_l_128": "https://github.com/aqlaboratory/genie/raw/main/weights/scope_l_128/epoch%3D49999.ckpt",
    "genie_l_256": "https://github.com/aqlaboratory/genie/raw/main/weights/scope_l_256/epoch%3D29999.ckpt",
    "genie_l_256_swissprot": "https://github.com/aqlaboratory/genie/raw/main/weights/swissprot_l_256/epoch%3D99.ckpt",
}
GENIE_MODEL_CONFIGS = {
    "genie_l_128": config.Genie128Config(),
    "genie_l_256": config.Genie256Config(),
    "genie_l_256_swissprot": config.Genie256Config(),
}


def _get_model_config(model_name: str) -> config.GenieConfig:
    """Get the model config for a given model name."""
    return GENIE_MODEL_CONFIGS[model_name]


class GenieForStructureDesign:
    def __init__(
        self,
        model_name: str = "genie_l_128",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = Genie(self.cfg)

        self.load_weights(GENIE_MODEL_URLS[model_name])
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
        return list(GENIE_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"{self.model_name}.pt",
            progress=True,
            map_location="cpu",
        )["state_dict"]
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def __call__(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[protein.ProteinCATrace, Dict[str, Any]]:
        """Design a random protein structure."""
        seq_len = inference_config.seq_len
        batch_size = inference_config.batch_size
        max_len = self.cfg.max_seq_len

        assert seq_len <= max_len, f"seq_len must be <= {max_len}"

        mask = torch.cat(
            [
                torch.ones((batch_size, seq_len)),
                torch.zeros((batch_size, max_len - seq_len)),
            ],
            dim=1,
        ).to(self.device)

        ts = self.model.p_sample_loop(mask, verbose=inference_config.verbose)[-1]
        for sample_idx in range(ts.shape[0]):
            coords = ts[sample_idx].trans.detach().cpu().numpy()
            coords = coords[:seq_len]

        n = len(coords)
        structure = protein.ProteinCATrace(
            atom_positions=coords.reshape((n, 1, 3)),
            aatype=np.array(n * [residue_constants.restype_order_with_x["G"]]),
            atom_mask=np.ones((n, 1)),
            residue_index=np.arange(0, n) + 1,
            b_factors=np.zeros((n, 1)),
            chain_index=np.zeros((n,), dtype=np.int32),
        )

        return structure, {}
