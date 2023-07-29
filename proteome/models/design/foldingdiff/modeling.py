import random
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from proteome import protein
from proteome.models.design.foldingdiff import config
from proteome.models.design.foldingdiff.model import BertForDiffusionBase
from proteome.models.design.foldingdiff import sampling
from proteome.models.design.foldingdiff.datasets import AnglesEmptyDataset, NoisedAnglesDataset
from proteome.models.design.foldingdiff.angles_and_coords import create_new_chain_nerf
from proteome.models.design.foldingdiff import utils

FOLDINGDIFF_MODEL_URLS = {
    "foldingdiff_cath": "https://huggingface.co/wukevin/foldingdiff_cath/resolve/main/models/best_by_valid/epoch%3D1488-step%3D565820.ckpt",  # noqa: E501
}
FOLDINGDIFF_MODEL_CONFIGS = {
    "foldingdiff_cath": config.BertForDiffusionConfig(),
}


def _get_model_config(model_name: str) -> config.BertForDiffusionConfig:
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

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, 
            file_name=f"{self.model_name}.pt",
            progress=True, 
            map_location="cpu",
        )["state_dict"]
        msg = self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def design_structure(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[protein.Protein, float]:
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
            self.model, noised_dset, n=1, sweep_lengths=(seq_len, seq_len + 1), disable_pbar=False
        )

        final_sampled = [s[-1] for s in sampled]
        sampled_dfs = [
            pd.DataFrame(s, columns=noised_dset.feature_names["angles"])
            for s in final_sampled
        ]

        out = create_new_chain_nerf("./generated.pdb", sampled_dfs[-1])


        # Create a random noise for generator
        noise_vec = torch.randn(tied_featurize_output.chain_M.shape, device=self.device)  # type: ignore
        pssm_log_odds_mask = (
            tied_featurize_output.pssm_log_odds_all > inference_config.pssm_threshold
        ).float()

        sample_features = asdict(tied_featurize_output)
        sample_features["noise"] = noise_vec
        sample_features["pssm_log_odds_mask"] = pssm_log_odds_mask
        sample_features |= asdict(inference_config)

        # Sample a sequence
        sample_dict = self.model.sample(sample_features)

        # Swap out the placeholder sequence with the sampled sequence
        sampled_sequence = sample_dict["S"]
        sample_features["sequence"] = sampled_sequence

        log_probs = self.model(
            sample_features,
            use_input_decoding_order=True,
            decoding_order=sample_dict["decoding_order"],
        )
        global_scores = get_sequence_scores(sampled_sequence, log_probs, tied_featurize_output.mask)
        global_score = global_scores.cpu().data.numpy()[0]

        seq = decode_sequence(sampled_sequence[0], tied_featurize_output.chain_M[0])

        return seq, global_score 
