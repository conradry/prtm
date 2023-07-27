import random
import io
from typing import Optional, Tuple

import numpy as np
import torch
from proteome import protein
from proteome.models.design.proteinsolver import config
from proteome.models.design.proteinsolver.proteinnet import ProteinNet
from proteome.models.design.proteinsolver.protein_structure import extract_seq_and_adj
from proteome.models.design.proteinsolver import protein_design
from proteome.models.folding.openfold.np.relax import cleanup
from proteome.models.design.proteinsolver.parser import Parser
from tqdm import tqdm

PS_MODEL_URLS = {
    "model_0": "https://models.proteinsolver.org/v0.1/notebooks/protein_4xEdgeConv_bs4/e12-s1652709-d6610836.state",  # noqa: E501
}
PS_MODEL_CONFIGS = {
    "model_0": config.ProteinNetConfig(),
}


def _get_model_config(model_name: str) -> config.ProteinNetConfig:
    """Get the model config for a given model name."""
    return PS_MODEL_CONFIGS[model_name]


class ProteinSeqDesForSequenceDesign:
    def __init__(
        self,
        model_name: str = "model_0",
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

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, 
            progress=True, 
            map_location="cpu",
        )
        msg = self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def design_sequence(
        self,
        structure: protein.Protein,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[str, float]:
        """Design a protein sequence for a given structure."""
        # The model needs a very particular structure to run correctly
        # first we need to convert our protein into a pdb file, then
        # cleanup the file by adding missing atoms and hydrogens
        # before finally converting it to the structure the model expects.
        pdb_str = protein.to_pdb(structure)
        pdb_file = io.StringIO(pdb_str)
        fixed_pdb = cleanup.fix_pdb(pdb_file, {})

        ps_structure = Parser().get_structure(fixed_pdb.split("\n"))
        adj_data = extract_seq_and_adj(ps_structure, 'A')
        data = dataset.row_to_data(pdata)
data = dataset.transform_edge_attr(data)
data.y = data.x
x_in = torch.ones_like(data.x) * 20

        return sequence, total_prob
