import re
from typing import Optional, Tuple

import torch

from proteome import protein
from proteome.models.folding.esm import config
from proteome.models.folding.esm.esmfold import ESMFold
from proteome.models.folding.openfold.utils.feats import atom14_to_atom37

ESM_MODEL_URLS = {
    "esm2_3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
}
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
        num_recycles: int = 1,
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
        self.num_recycles = num_recycles

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        fold_model_state = torch.hub.load_state_dict_from_url(
            weights_url, progress=False, map_location="cpu"
        )["model"]
        language_model_state = torch.hub.load_state_dict_from_url(
            ESM_MODEL_URLS[self.cfg.esm_type], progress=False, map_location="cpu"
        )["model"]

        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {
                "esm." + pattern.sub("", name): param
                for name, param in state_dict.items()
            }
            return state_dict

        model_state = fold_model_state | upgrade_state_dict(language_model_state)

        expected_missing = {
            "esm.contact_head.regression.weight",
            "esm.contact_head.regression.bias",
        }

        expected_keys = set(self.model.state_dict().keys()).difference(expected_missing)
        found_keys = set(model_state.keys())

        missing_essential_keys = []
        for missing_key in expected_keys - found_keys:
            if not missing_key.startswith("esm."):
                missing_essential_keys.append(missing_key)

        if missing_essential_keys:
            raise RuntimeError(
                f"Keys '{', '.join(missing_essential_keys)}' are missing."
            )

        msg = self.model.load_state_dict(model_state, strict=False)

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

        output = self.model.infer(
            sequences=[sequence],
            num_recycles=self.num_recycles,
        )

        # Prepare the output for Protein
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = output["atom37_atom_exists"]

        for k, v in output.items():
            output[k] = v.squeeze()

        predicted_protein = protein.Protein(
            aatype=output["aatype"],
            atom_positions=final_atom_positions.squeeze(),
            atom_mask=final_atom_mask.squeeze(),
            residue_index=output["residue_index"] + 1,
            b_factors=output["plddt"],
            chain_index=output["chain_index"] if "chain_index" in output else None,
        )

        return predicted_protein, float(output["mean_plddt"])
