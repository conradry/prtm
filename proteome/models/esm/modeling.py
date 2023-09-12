import random
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from proteome import protein
from proteome.models.esm import config
from proteome.models.esm.esmfold import ESMFold
from proteome.models.esm.inverse_folding.gvp_transformer import GVPTransformerModel
from proteome.models.openfold.utils.feats import atom14_to_atom37

ESM_MODEL_URLS = {
    "esm2_3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
}
ESMFOLD_MODEL_URLS = {
    "esmfold_3B_v0": "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v0.pt",
    "esmfold_3B_v1": "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt",
}
ESMIF_MODEL_URLS = {
    "esm_if1_gvp4_t16_142M_UR50": "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt",
}
ESMFOLD_MODEL_CONFIGS = {
    "esmfold_3B_v0": config.ESMFoldV0(),
    "esmfold_3B_v1": config.ESMFoldV1(),
}
ESMIF_MODEL_CONFIGS = {
    "esm_if1_gvp4_t16_142M_UR50": config.ESMIFConfig(),
}


def _get_esmfold_model_config(model_name: str) -> config.ESMFoldConfig:
    """Get the model config for a given model name."""
    return ESMFOLD_MODEL_CONFIGS[model_name]


def _get_esmif_model_config(model_name: str) -> config.ESMIFConfig:
    """Get the model config for a given model name."""
    return ESMIF_MODEL_CONFIGS[model_name]


class ESMForFolding:
    def __init__(
        self,
        model_name: str = "esmfold_3B_v1",
        num_recycles: int = 1,
        chunk_size: Optional[int] = None,
        half_precision: bool = True,
    ):
        self.model_name = model_name
        self.cfg = _get_esmfold_model_config(model_name)
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

    @classmethod
    @property
    def available_models(cls):
        return list(ESMFOLD_MODEL_URLS.keys())

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
    def __call__(self, sequence: str) -> Tuple[protein.Protein37, Dict[str, Any]]:
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

        predicted_protein = protein.Protein37(
            aatype=output["aatype"],
            atom_positions=final_atom_positions.squeeze(),
            atom_mask=final_atom_mask.squeeze(),
            residue_index=output["residue_index"] + 1,
            b_factors=output["plddt"],
            chain_index=output["chain_index"] if "chain_index" in output else None,
        )

        return predicted_protein, {"mean_plddt": float(output["mean_plddt"])}


class ESMForInverseFolding:
    def __init__(
        self, 
        model_name: str = "esm_if1_gvp4_t16_142M_UR50",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_esmif_model_config(model_name)
        self.model = GVPTransformerModel(cfg=self.cfg)
        self.load_weights(ESMIF_MODEL_URLS[model_name])
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
        return list(ESMIF_MODEL_CONFIGS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        model_state = torch.hub.load_state_dict_from_url(
            weights_url, progress=False, map_location="cpu"
        )["model"]

        def update_name(s: str):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_state.items()
            if "version" not in sname
        }
        msg = self.model.load_state_dict(model_state, strict=False)

    @torch.no_grad()
    def __call__(
        self, 
        structure: protein.ProteinBase, 
        design_params: config.DesignParams = config.DesignParams(),
        temperature: float = 1.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Design a protein sequence for a given structure."""
        # Expects 3 atom protein structure
        structure = structure.to_protein3()
        coords = torch.tensor(structure.atom_positions).float().to(self.device)
        if design_params.confidence is not None:
            confidence = torch.tensor(design_params.confidence).float().to(self.device)
        else:
            confidence = None

        sequence, avg_prob = self.model.sample(
            coords,
            temperature=temperature,
            confidence=confidence,
            partial_seq=design_params.partial_seq_list,
        )
        return sequence, {"avg_prob": avg_prob}