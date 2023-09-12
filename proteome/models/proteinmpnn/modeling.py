import random
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from proteome import protein
from proteome.constants.residue_constants import proteinmppn_restypes
from proteome.models.proteinmpnn import config
from proteome.models.proteinmpnn.featurizer import tied_featurize, get_sequence_scores, decode_sequence
from proteome.models.proteinmpnn.model import ProteinMPNN

PROTEINMPNN_MODEL_URLS = {
    "vanilla_model-2": "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_002.pt",
    "vanilla_model-10": "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_010.pt",
    "vanilla_model-20": "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt",
    "vanilla_model-30": "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_030.pt",
    "ca_only_model-2": "https://github.com/dauparas/ProteinMPNN/raw/main/ca_model_weights/v_48_002.pt",
    "ca_only_model-10": "https://github.com/dauparas/ProteinMPNN/raw/main/ca_model_weights/v_48_010.pt",
    "ca_only_model-20": "https://github.com/dauparas/ProteinMPNN/raw/main/ca_model_weights/v_48_020.pt",
}
PROTEINMPNN_MODEL_CONFIGS = {
    "vanilla_model": config.ProteinMPNNConfig(),
    "ca_only_model": config.ProteinMPNNCAOnlyConfig(),
}


def _get_model_config(model_name: str) -> config.ProteinMPNNConfig:
    """Get the model config for a given model name."""
    return PROTEINMPNN_MODEL_CONFIGS[model_name.split("-")[0]]


def _get_default_design_params(sequence_length: int) -> config.DesignParams:
    """Make default design params for a given sequence length."""
    num_aa = len(proteinmppn_restypes)
    design_params = config.DesignParams(
        design_mask=np.ones(sequence_length),
        design_aatype_mask=np.zeros([sequence_length, num_aa], np.int32),
        pssm_coef=np.zeros(sequence_length),
        pssm_bias=np.zeros([sequence_length, num_aa]),
        pssm_log_odds=10000.0 * np.ones([sequence_length, num_aa]),
        bias_per_residue=np.zeros([sequence_length, num_aa]),
    )
    return design_params


class ProteinMPNNForSequenceDesign:
    def __init__(
        self,
        model_name: str = "vanilla_model-30",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = ProteinMPNN(cfg=self.cfg)

        self.load_weights(PROTEINMPNN_MODEL_URLS[model_name])
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.ca_only = isinstance(self.cfg, config.ProteinMPNNCAOnlyConfig)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    @classmethod
    @property
    def available_models(cls):
        return list(PROTEINMPNN_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, 
            file_name=f"proteinmpnn_{self.model_name}.pt",
            progress=True, 
            map_location="cpu",
        )["model_state_dict"]
        msg = self.model.load_state_dict(state_dict)

    def _featurize_input(self, structure: config.DesignableProtein) -> config.TiedFeaturizeOutput:
        return tied_featurize(
            [structure],
            self.device,
            chain_dict=None,  # currently unsupported keep None
            ca_only=self.ca_only,
        )

    @torch.no_grad()
    def __call__(
        self,
        structure: Union[protein.ProteinCATrace, protein.Protein4],
        design_params: Optional[config.DesignParams] = None,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[str, Dict[str, Any]]:
        """Design a protein sequence for a given structure."""
        # Quick check to make sure the structure has the right number
        # of atoms for the chosen model
        if self.ca_only:
            structure = structure.to_ca_trace()
        else:
            structure = structure.to_protein4()

        # Add design params to structure if provided
        # otherwise use the default design params
        if design_params is None:
            design_params = _get_default_design_params(len(structure.aatype))

        designable_structure = config.DesignableProtein(structure, design_params)
        tied_featurize_output = self._featurize_input(designable_structure)

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

        return seq, {"global_score": global_score}
