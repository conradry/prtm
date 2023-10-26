import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from prtm import protein
from prtm.constants import residue_constants
from prtm.models.foldingdiff import config, sampling
from prtm.models.foldingdiff.angles_and_coords import create_new_chain_nerf
from prtm.models.foldingdiff.datasets import AnglesEmptyDataset, NoisedAnglesDataset
from prtm.models.foldingdiff.model import BertForDiffusionBase

__all__ = ["FoldingDiffForStructureDesign"]

FOLDINGDIFF_MODEL_URLS = {
    "foldingdiff_cath": "https://huggingface.co/wukevin/foldingdiff_cath/resolve/main/models/best_by_valid/epoch%3D1488-step%3D565820.ckpt",  # noqa: E501
}
FOLDINGDIFF_MODEL_CONFIGS = {
    "foldingdiff_cath": config.FoldingDiffCathConfig,
}


def _get_model_config(model_name: str) -> config.FoldingDiffConfig:
    """Get the model config for a given model name."""
    return FOLDINGDIFF_MODEL_CONFIGS[model_name]


class FoldingDiffForStructureDesign:
    def __init__(
        self,
        model_name: str = "foldingdiff_cath",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = BertForDiffusionBase(self.cfg)

        self.load_weights(FOLDINGDIFF_MODEL_URLS[model_name])
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
        return list(FOLDINGDIFF_MODEL_URLS.keys())

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
        placeholder_dset = AnglesEmptyDataset(
            feature_set_key=inference_config.dataset_config.angles_definitions,
            pad=inference_config.dataset_config.max_seq_len,
            mean_offset=inference_config.dataset_config.mean_offset,
        )
        noised_dset = NoisedAnglesDataset(
            dset=placeholder_dset,
            dset_key="coords"
            if inference_config.dataset_config.angles_definitions == "cart-coords"
            else "angles",
            timesteps=inference_config.dataset_config.timesteps,
            exhaustive_t=False,
            beta_schedule=inference_config.dataset_config.variance_schedule,
            nonangular_variance=1.0,
            angular_variance=inference_config.dataset_config.variance_scale,
        )
        seq_len = inference_config.seq_len
        sampled = sampling.sample(
            self.model,
            noised_dset,
            n=1,
            sweep_lengths=(seq_len, seq_len + 1),
            disable_pbar=False,
        )

        final_sampled = [s[-1] for s in sampled]
        sampled_dfs = [
            pd.DataFrame(s, columns=noised_dset.feature_names["angles"])
            for s in final_sampled
        ]

        coords = create_new_chain_nerf(sampled_dfs[-1])

        # All residues are glycine
        n = len(coords)
        structure = protein.ProteinCATrace(
            atom_positions=coords[:, [1]],
            aatype=np.array(n * [residue_constants.restype_order_with_x["G"]]),
            atom_mask=np.ones((n, 1)),
            residue_index=np.arange(0, n) + 1,
            b_factors=np.zeros((n, 1)),
            chain_index=np.zeros((n,), dtype=np.int32),
        )

        return structure, {}
