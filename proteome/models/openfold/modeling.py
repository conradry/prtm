import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from proteome import protein
from proteome.models.openfold import config
from proteome.models.openfold.data import data_pipeline, feature_pipeline
from proteome.models.openfold.model import model
from proteome.models.openfold.utils.tensor_utils import tensor_tree_map
from proteome.query.pipeline import QueryPipelines
from proteome.utils import hub_utils

OPENFOLD_MODEL_URLS = {
    "finetuning-3": "s3://openfold/openfold_params/finetuning_3.pt",
    "finetuning-4": "s3://openfold/openfold_params/finetuning_4.pt",
    "finetuning-5": "s3://openfold/openfold_params/finetuning_5.pt",
    "finetuning_ptm-2": "s3://openfold/openfold_params/finetuning_ptm_2.pt",
    "finetuning_no_templ_ptm-1": "s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt",
}

OPENFOLD_MODEL_CONFIGS = {
    "finetuning-3": config.FinetuningConfig(),
    "finetuning-4": config.FinetuningConfig(),
    "finetuning-5": config.FinetuningConfig(),
    "finetuning_ptm-2": config.FinetuningPTMConfig(),
    "finetuning_no_templ_ptm-1": config.FinetuningNoTemplatePTMConfig(),
}


def _placeholder_template_feats(
    num_templates: int, num_res: int
) -> Dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((num_templates, num_res, 22), dtype=np.int64),
        "template_all_atom_positions": np.zeros(
            (num_templates, num_res, 37, 3), dtype=np.float32
        ),
        "template_all_atom_mask": np.zeros(
            (num_templates, num_res, 37), dtype=np.float32
        ),
        "template_domain_names": np.zeros((num_templates,), dtype=np.float32),
        "template_sum_probs": np.zeros((num_templates, 1), dtype=np.float32),
    }


def _get_model_config(model_name: str) -> config.OpenFoldConfig:
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return OPENFOLD_MODEL_CONFIGS[model_name]


class OpenFoldForFolding:
    def __init__(
        self,
        model_name: str = "finetuning-3",
        msa_pipeline: str = "alphafold_jackhmmer",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = model.AlphaFold(self.cfg)

        self.load_weights(OPENFOLD_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.msa_pipeline = getattr(QueryPipelines, msa_pipeline)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    def load_weights(self, weights_url):
        """Load weights from a weights url."""
        weights = hub_utils.load_state_dict_from_s3_url(weights_url)
        msg = self.model.load_state_dict(weights, strict=True)

    def _validate_input_sequence(self, sequence: str) -> None:
        """Validate that the input sequence is a valid amino acid sequence."""
        sequence = sequence.translate(str.maketrans("", "", " \n\t")).upper()
        aatypes = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(sequence).issubset(aatypes):
            raise ValueError(
                f"Input sequence contains non-amino acid letters: {set(sequence) - aatypes}."
            )

    def _featurize_input(
        self, sequence: str, msas: List[Tuple], deletion_matrices: List[Tuple]
    ) -> Dict[str, torch.Tensor]:
        """Prepare the input features for a given sequence and its MSAs and deletion matrices."""
        num_res = len(sequence)
        num_templates = 1  # dummy number --- is ignored

        feature_dict = {}
        feature_dict.update(data_pipeline.make_sequence_features(sequence, num_res))
        feature_dict.update(
            data_pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices)
        )
        feature_dict.update(_placeholder_template_feats(num_templates, num_res))
        features = config.Features(**feature_dict)

        pipeline = feature_pipeline.FeaturePipeline(self.cfg.data)
        processed_feature_dict = pipeline.process_features(features)
        processed_feature_dict = tensor_tree_map(
            lambda t: t.to(self.device), processed_feature_dict
        )

        assert isinstance(processed_feature_dict, dict)
        return processed_feature_dict

    @torch.no_grad()
    def fold(self, sequence: str) -> Tuple[protein.Protein, float]:
        """Fold a protein sequence."""
        self._validate_input_sequence(sequence)

        # Get MSAs for the input sequence using Jackhmmer.
        msas, deletion_matrices = self.msa_pipeline(sequence)

        # Prepare the input features for a given sequence and its MSAs and deletion matrices.
        feature_dict = self._featurize_input(sequence, msas, deletion_matrices)

        res = self.model(feature_dict)

        # Unpack to a protein and a plddt confidence metric.
        mean_plddt = float(res["plddt"].mean())
        b_factors = res["plddt"][:, None] * res["final_atom_mask"]
        chain_index = np.zeros((len(sequence),), dtype=np.int32)

        structure = protein.Protein(
            atom_positions=res["final_atom_positions"].cpu().numpy(),
            aatype=feature_dict["aatype"].cpu().numpy()[:, 0],
            atom_mask=res["final_atom_mask"].cpu().numpy(),
            residue_index=feature_dict["residue_index"].cpu().numpy()[:, 0] + 1,
            b_factors=b_factors.cpu().numpy(),
            chain_index=chain_index,
        )

        return structure, mean_plddt
