from typing import Optional, Tuple

import torch
from proteome import protein
from proteome.models.folding.esm import config
from proteome.models.folding.esm.esmfold import ESMFold

ESMFOLD_MODEL_URLS = {
    "esmfold_3B_v0": "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v0.pt",
    "esmfold_3B_v1": "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt",
}
ESMFOLD_MODEL_CONFIGS = {
    "esmfold_3B_v0": config.ESMFoldV0(),
    "esmfold_3B_v1": config.ESMFoldV1(),
}


def _get_model_config(model_name: str) -> config.ESMFoldConfig:
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return ESMFOLD_MODEL_CONFIGS[model_name]


class ESMForFolding:
    def __init__(
        self,
        model_name: str = "esmfold_3B_v1",
        chunk_size: Optional[int] = None,
        half_precision: bool = True,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = ESMFold(cfg=self.cfg)
        self.model.set_chunk_size(512)

        self.load_weights(ESMFOLD_MODEL_URLS[model_name])

        self.model.set_chunk_size(chunk_size)
        if half_precision:
            self.model.half()

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        model_state = torch.hub.load_state_dict_from_url(
            weights_url, progress=False, map_location="cpu"
        )["model"]

        expected_keys = set(self.model.state_dict().keys())
        found_keys = set(model_state.keys())

        missing_essential_keys = []
        for missing_key in expected_keys - found_keys:
            if not missing_key.startswith("esm."):
                missing_essential_keys.append(missing_key)

        if missing_essential_keys:
            raise RuntimeError(
                f"Keys '{', '.join(missing_essential_keys)}' are missing."
            )

        self.model.load_state_dict(model_state, strict=False)

    def _validate_input_sequence(self, sequence: str) -> None:
        """Validate that the input sequence is a valid amino acid sequence."""
        sequence = sequence.translate(str.maketrans("", "", " \n\t")).upper()
        aatypes = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(sequence).issubset(aatypes):
            raise ValueError(
                f"Input sequence contains non-amino acid letters: {set(sequence) - aatypes}."
            )

    @torch.no_grad()
    def fold(self, sequence: str) -> Tuple[protein.Protein, float]:
        """Fold a protein sequence."""
        self._validate_input_sequence(sequence)

        res = self.model.infer(
            sequences=[sequence],
            # predict_with_confidence=True,
            # fwd_cfg=self.forward_config,
        )
        return res

        predicted_protein = protein.Protein(
            atom_positions=res["final_atom_positions"].cpu().numpy(),
            aatype=list_of_feature_dict[0]["p_msa"][0]
            .cpu()
            .numpy(),  # first msa is sequence
            atom_mask=res["final_atom_mask"].cpu().numpy(),
            residue_index=np.arange(len(sequence)),
            b_factors=np.zeros(
                tuple(res["final_atom_positions"].shape[:2])
            ),  # no lddt predicted
        )

        return predicted_protein, res["confidence_overall"]
