import random
from typing import Optional

import numpy as np
import torch
from proteome import protein
from proteome.constants import residue_constants
from proteome.models.design.rfdiffusion import config

RFD_MODEL_URLS = {
    "base": "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "complex_base": "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
    "complex_fold_base": "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
    "inpaint_seq": "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
    "inpaint_seq_fold": "http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt",
    "active_site": "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
    "base_epoch8": "http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt",
    "complex_beta": "http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt",
    "rf_structure_prediction_weights": "http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt",  # noqa: E501
}
GENIE_MODEL_CONFIGS = {
    "genie_l_128": config.GenieConfig(),
}


def _get_model_config(model_name: str) -> config.GenieConfig:
    """Get the model config for a given model name."""
    return GENIE_MODEL_CONFIGS[model_name]


class GenieForStructureDesign:
    def __init__(
        self,
        model_name: str = "genie_l_128",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = Genie(self.cfg)

        self.load_weights(GENIE_MODEL_URLS[model_name])
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
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def design_structure(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> protein.Protein:
        """Design a random protein structure."""
        seq_len = inference_config.seq_len
        batch_size = inference_config.batch_size
        max_len = self.cfg.max_seq_len

        assert seq_len <= max_len, f"seq_len must be <= {max_len}"

        mask = torch.cat(
            [
                torch.ones((batch_size, seq_len)),
                torch.zeros((batch_size, max_len - seq_len)),
            ],
            dim=1,
        ).to(self.device)

        ts = self.model.p_sample_loop(mask, verbose=inference_config.verbose)[-1]
        for sample_idx in range(ts.shape[0]):
            coords = ts[sample_idx].trans.detach().cpu().numpy()
            coords = coords[:seq_len]

        coords = coords.reshape(-1, 1, 3)
        n = len(coords)
        structure = protein.Protein(
            atom_positions=coords,
            aatype=np.array(n * [residue_constants.restype_order_with_x["G"]]),
            atom_mask=np.ones((n, 1)),
            residue_index=np.arange(0, n),
            b_factors=np.zeros((n, 1)),
            chain_index=np.zeros((n,), dtype=np.int32),
        )

        return structure
