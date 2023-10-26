from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from prtm import protein
from prtm.constants import residue_constants
from prtm.models.omegafold import config, model, utils

__all__ = ["OmegaFoldForFolding"]

OMEGAFOLD_MODEL_URLS = {
    "model-1": "https://helixon.s3.amazonaws.com/release1.pt",
    "model-2": "https://helixon.s3.amazonaws.com/release2.pt",
}
OMEGAFOLD_MODEL_CONFIGS = {
    "model-1": config.OmegaFoldModel1Config(),
    "model-2": config.OmegaFoldModel2Config(),
}


def _generate_pseudo_msas(
    sequence: str,
    num_cycles: int,
    num_pseudo_msa: int = 15,
    pseudo_msa_mask_rate: float = 0.12,
) -> List[Dict[str, torch.Tensor]]:
    """Generate pseudo-MSAs for a given sequence.

    Args:
        sequence (str): Amino acid sequence string.
        num_cycles (int): Number of pseudo msas to generate.
        num_pseudo_msa (int): Number of sequences in each pseudo msa.
        pseudo_msa_mask_rate (float): Rate of masking in each pseudo msa.
    Returns:
        output (List[List[Dict[str, torch.Tensor]]) : A list of num_cycles
        dictionaries containing the pseudo msas and their masks.

    """
    mapping = residue_constants.restype_order_with_x_dash
    L = len(sequence)
    aatype = torch.tensor([mapping.get(aa, mapping["X"]) for aa in sequence]).long()
    mask = torch.ones_like(aatype).float()

    # Use a fixed seed for reproducibility.
    g = torch.Generator()
    g.manual_seed(L)
    data = []
    for _ in range(num_cycles):
        p_msa = aatype[None, :].repeat(num_pseudo_msa, 1)
        p_msa_mask = torch.rand([num_pseudo_msa, L], generator=g) > pseudo_msa_mask_rate
        p_msa_mask = torch.cat((mask[None, :], p_msa_mask), dim=0)

        p_msa = torch.cat((aatype[None, :], p_msa), dim=0)
        p_msa[~p_msa_mask.bool()] = 21
        data.append({"p_msa": p_msa, "p_msa_mask": p_msa_mask})

    return data


def _get_model_config(model_name: str) -> config.OmegaFoldModelConfig:
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return OMEGAFOLD_MODEL_CONFIGS[model_name]


class OmegaFoldForFolding:
    def __init__(
        self,
        model_name: str = "model-1",
        num_msa_cycles: int = 1,
        num_pseudo_msa: int = 15,
        pseudo_msa_mask_rate: float = 0.12,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = model.OmegaFold(self.cfg)

        self.load_weights(OMEGAFOLD_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.featurizer = partial(
            _generate_pseudo_msas,
            num_cycles=num_msa_cycles,
            num_pseudo_msa=num_pseudo_msa,
            pseudo_msa_mask_rate=pseudo_msa_mask_rate,
        )

    @classmethod
    @property
    def available_models(cls):
        return list(OMEGAFOLD_MODEL_URLS.keys())

    def load_weights(self, weights_url):
        """Load weights from a weights url."""
        # Need to explicitly map to cpu for these weights.
        weights = torch.hub.load_state_dict_from_url(
            weights_url, map_location="cpu", file_name=f"omegafold_{self.model_name}.pt"
        )
        msg = self.model.load_state_dict(weights, strict=True)

    def _validate_input_sequence(self, sequence: str) -> None:
        """Validate that the input sequence is a valid amino acid sequence."""
        sequence = sequence.translate(str.maketrans("", "", " \n\t")).upper()
        aatypes = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(sequence).issubset(aatypes):
            raise ValueError(
                f"Input sequence contains non-amino acid letters: {set(sequence) - aatypes}."
            )

    def _featurize_input(self, sequence: str) -> List[Dict[str, torch.Tensor]]:
        """Prepare the input features for a given sequence and its MSAs and deletion matrices."""
        list_of_feature_dict = self.featurizer(sequence=sequence)
        list_of_feature_dict = utils.recursive_to(
            list_of_feature_dict, device=self.device
        )
        return list_of_feature_dict

    @torch.no_grad()
    def __call__(
        self,
        sequence: str,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[protein.Protein14, Dict[str, Any]]:
        """Fold a protein sequence."""
        self._validate_input_sequence(sequence)

        # Prepare the input features for a given sequence and its MSAs and deletion matrices.
        list_of_feature_dict = self._featurize_input(sequence)

        res = self.model(
            list_of_feature_dict,
            predict_with_confidence=True,
            return_embeddings=False,
            fwd_cfg=inference_config,
        )

        predicted_protein = protein.Protein14(
            atom_positions=res["final_atom_positions"].cpu().numpy(),
            aatype=list_of_feature_dict[0]["p_msa"][0]
            .cpu()
            .numpy(),  # first msa is sequence
            atom_mask=res["final_atom_mask"].cpu().numpy(),
            residue_index=np.arange(len(sequence)) + 1,
            b_factors=100 * res["confidence"].cpu().numpy()[:, None].repeat(14, axis=1),
            chain_index=np.zeros(len(sequence), dtype=np.int32),
        )

        return predicted_protein, {"confidence": 100 * res["confidence_overall"]}
