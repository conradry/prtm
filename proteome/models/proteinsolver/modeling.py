import io
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from proteome import protein
from proteome.models.proteinsolver import config
from proteome.models.proteinsolver.proteinnet import ProteinNet
from proteome.models.proteinsolver.protein_structure import (
    AMINO_ACIDS, extract_seq_and_adj, row_to_data, transform_edge_attr
)
from proteome.models.proteinsolver import protein_design
from proteome.models.openfold.np.relax import cleanup
from proteome.models.proteinsolver.parser import Parser

PS_MODEL_URLS = {
    "model_0": "https://models.proteinsolver.org/v0.1/notebooks/protein_4xEdgeConv_bs4/e12-s1652709-d6610836.state",  # noqa: E501
}
PS_MODEL_CONFIGS = {
    "model_0": config.ProteinNetConfig(),
}


def _get_model_config(model_name: str) -> config.ProteinNetConfig:
    """Get the model config for a given model name."""
    return PS_MODEL_CONFIGS[model_name]


class ProteinSolverForInverseFolding:
    def __init__(
        self,
        model_name: str = "model_0",
        random_seed: int = 0,
    ):
        self.model_name = model_name
        cfg = _get_model_config(model_name)
        self.model = ProteinNet(cfg)
        self.load_weights(PS_MODEL_URLS[model_name])
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
        return list(PS_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, 
            progress=True, 
            file_name=f"proteinsolver_{self.model_name}.pt",
            map_location="cpu",
        )
        msg = self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def __call__(
        self,
        structure: protein.Protein27,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[str, Dict[str, Any]]:
        """Design a protein sequence for a given structure."""
        structure = structure.to_protein27()
        # The model needs a very particular structure to run correctly
        # first we need to convert our protein into a pdb file, then
        # cleanup the file by adding missing atoms and hydrogens
        # before finally converting it to the structure the model expects.
        pdb_str = structure.to_pdb()
        pdb_file = io.StringIO(pdb_str)
        fixed_pdb = cleanup.fix_pdb(pdb_file, {})

        ps_structure = Parser().get_structure(fixed_pdb.split("\n"))
        adj_data = extract_seq_and_adj(ps_structure, 'A')
        data = row_to_data(adj_data)
        data = transform_edge_attr(data)
        data.y = data.x
        x_placeholder = torch.ones_like(data.x) * 20

        sequence_ids, mean_proba = protein_design.design_protein(
            self.model, 
            x_placeholder.to(self.device), 
            data.edge_index.to(self.device), 
            data.edge_attr.to(self.device), 
            cutoff=inference_config.log_prob_cutoff,
            max_sequences=inference_config.max_sequences,
        )
        sequence = "".join([AMINO_ACIDS[i] for i in sequence_ids])

        return sequence, {"avg_prob": mean_proba}
