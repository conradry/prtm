from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from proteome import protein
from proteome.models.folding.dmpfold2 import config, model, utils
from proteome.query.pipeline import QueryPipelines

DMPFOLD_MODEL_URLS = {
    "base": (
        "https://github.com/psipred/DMPfold2/raw/master/dmpfold/trained_model/FINAL_fullmap_e2e_model_part1.pt",
        "https://github.com/psipred/DMPfold2/raw/master/dmpfold/trained_model/FINAL_fullmap_e2e_model_part2.pt",
    ),
}
DMPFOLD_MODEL_CONFIGS = {
    "base": config.DMPFold2Config(),
}


def _get_model_config(model_name: str) -> config.DMPFold2Config:
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return DMPFOLD_MODEL_CONFIGS[model_name]


class DMPFoldForFolding:
    def __init__(
        self,
        model_name: str = "base",
        msa_pipeline: str = "alphafold_jackhmmer",
        recycling_iters: int = 5,
        refinement_steps: int = 100,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = model.GRUResNet(self.cfg)

        self.load_weights(DMPFOLD_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.msa_pipeline = getattr(QueryPipelines, msa_pipeline)
        self.recycling_iters = recycling_iters
        self.refinement_steps = refinement_steps

    def load_weights(self, weights_urls: Tuple[str, str]):
        """Load weights from shards defined at weights urls."""
        state_dict = {}
        for url in weights_urls:
            state_dict.update(torch.hub.load_state_dict_from_url(url))
        msg = self.model.load_state_dict(state_dict, strict=True)

    def _validate_input_sequence(self, sequence: str) -> None:
        """Validate that the input sequence is a valid amino acid sequence."""
        sequence = sequence.translate(str.maketrans("", "", " \n\t")).upper()
        aatypes = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(sequence).issubset(aatypes):
            raise ValueError(
                f"Input sequence contains non-amino acid letters: {set(sequence) - aatypes}."
            )

    def _encode_msas(self, sequence: str, msas: List[Tuple]) -> np.ndarray:
        """Prepare the input features for a given sequence and its MSAs and deletion matrices."""
        msa_list = [sequence]
        for db_hits in msas:
            # Skip first which is the sequence itself
            msa_list.extend(list(db_hits)[1:])

        aa_trans = str.maketrans(
            "ARNDCQEGHILKMFPSTWYVBJOUXZ-.", "ABCDEFGHIJKLMNOPQRSTUUUUUUVV"
        )
        nseqs = len(msa_list)
        length = len(sequence)
        msa = (
            np.frombuffer(
                "".join(msa_list).translate(aa_trans).encode("latin-1"), dtype=np.uint8
            )
            - ord("A")
        ).reshape(nseqs, length)

        return msa

    def _featurize_input(
        self, msas: np.ndarray, max_msas: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Featurize the input msas for the model."""
        msas = msas[:, :max_msas]
        nseqs, length = msas.shape
        msa_tensor = torch.from_numpy(msas).long().to(self.device)

        msa1hot = F.one_hot(torch.clamp(msa_tensor, max=20), 21).float()
        w = utils.reweight(msa1hot, cutoff=0.8)

        if nseqs > 1:
            f2d_dca = utils.fast_dca(msa1hot, w).float()
        else:
            f2d_dca = torch.zeros((length, length, 442), device=self.device)

        f2d_dca = f2d_dca.permute(2, 0, 1).unsqueeze(0)
        # dmap is null because we aren't using templates
        dmap = -torch.ones((1, 1, length, length), device=self.device)
        cov_tensor = torch.cat(
            (f2d_dca, dmap), dim=1
        )  # template and coevolution features

        return {"msa": msa_tensor, "cov": cov_tensor}

    def fold(
        self, sequence: str, max_msas: int = 1000
    ) -> Tuple[protein.Protein, float]:
        """Fold a protein sequence."""
        self._validate_input_sequence(sequence)

        # Get MSAs for the input sequence
        L = len(sequence)
        msas, _ = self.msa_pipeline(sequence)
        msa_tensor = self._encode_msas(sequence, msas)
        feature_dict = self._featurize_input(msa_tensor, max_msas=max_msas)
        with torch.no_grad():
            xyz, confs = self.model(
                feature_dict,
                nloops=self.recycling_iters,
                refine_steps=self.refinement_steps,
            )

        xyz = xyz.view(-1, L, 5, 3)[0].cpu().numpy()
        confs = confs[0].cpu().numpy()

        predicted_protein = protein.Protein(
            atom_positions=xyz,
            aatype=feature_dict["msa"][0].cpu().numpy(),
            atom_mask=np.ones_like(xyz)[..., 0],
            residue_index=np.arange(L) + 1,
            b_factors=np.zeros_like(xyz)[..., 0],  # no b_factors
        )
        confidence = float(confs.mean())

        return predicted_protein, confidence
