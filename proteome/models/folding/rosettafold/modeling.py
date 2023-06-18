from typing import Dict, List, Tuple

import numpy as np
import torch

from proteome import protein
from proteome.models.folding.rosettafold.config import (RoseTTAFoldConfig,
                                                        TRFoldConfig)
from proteome.models.folding.rosettafold.kinematics import xyz_to_t2d
from proteome.models.folding.rosettafold.parsers import parse_a3m
from proteome.models.folding.rosettafold.rosettafoldmodel import RoseTTAFold
from proteome.models.folding.rosettafold.trfold import TRFold
from proteome.query.pipeline import QueryPipelines
from proteome.utils import hub_utils

ROSETTAFOLD_MODEL_URLS = {
    "rosettafold_end2end": (
        "https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz",
        "weights/RoseTTAFold_e2e.pt",
    ),
}


class RoseTTAFoldForFolding:
    def __init__(
        self,
        model_name: str = "rosettafold_end2end",
        msa_pipeline: str = "alphafold_jackhmmer",
        refine: bool = True,
    ):
        self.model_name = model_name
        self.cfg = RoseTTAFoldConfig()
        self.model = RoseTTAFold(self.cfg)

        self.load_weights(*ROSETTAFOLD_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.trfold = TRFold(TRFoldConfig(), self.device)
        self.msa_pipeline = getattr(QueryPipelines, msa_pipeline)
        self.refine = refine

    def load_weights(self, weights_url: str, zip_path: str):
        """Load weights from a weights url."""
        weights = hub_utils.load_state_dict_from_tar_gz_url(weights_url, zip_path)
        msg = self.model.load_state_dict(weights["model_state_dict"], strict=True)

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

        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype="|S1").view(np.uint8)
        msa = np.array([list(s) for s in msa_list], dtype="|S1").view(np.uint8)
        for i in range(alphabet.shape[0]):
            msa[msa == alphabet[i]] = i

        # treat all unknown characters as gaps
        msa[msa > 20] = 20
        return msa

    def _featurize_input(self, msas: np.ndarray, max_msas: int = 1000):
        """Featurize the input msas for the model."""
        msas = msas[:, :max_msas]
        N, L = msas.shape

        # Template placeholders
        xyz_t = torch.full((1, L, 3, 3), np.nan).float()
        t1d = torch.zeros((1, L, 3)).float()
        t0d = torch.zeros((1, 3)).float()

        msa_tensor = torch.tensor(msas).long().view(1, -1, L).to(self.device)
        idx_pdb = torch.arange(L).long().view(1, L)
        seq = msa_tensor[:, 0]

        # template features
        xyz_t = xyz_t.float().unsqueeze(0)
        t1d = t1d.float().unsqueeze(0)
        t0d = t0d.float().unsqueeze(0)
        t2d = xyz_to_t2d(xyz_t, t0d)

        msa_tensor = msa_tensor.to(self.device)

        idx_pdb = idx_pdb.to(self.device)
        t1d = t1d[:, :10].to(self.device)
        t2d = t2d[:, :10].to(self.device)

        return {"msa": msa_tensor, "seq": seq, "idx": idx_pdb, "t1d": t1d, "t2d": t2d}

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
            prob_s, xyz, lddt = self.model(feature_dict, refine=self.refine)

        prob_trF = []
        for prob in prob_s:
            prob = prob.reshape(-1, L, L).permute(1, 2, 0).cpu().numpy()
            prob = torch.tensor(prob).permute(2, 0, 1).to(self.device)
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
            prob_trF.append(prob)

        xyz = xyz[0, :, 1]
        xyz = self.trfold.fold(xyz, prob_trF, batch=15, lr=0.1, nsteps=200)
        xyz = xyz.detach().cpu().numpy()

        xyzo = protein.add_oxygen_to_atom_positions(xyz)
        predicted_protein = protein.Protein(
            atom_positions=xyzo,
            aatype=feature_dict["seq"][0].cpu().numpy(),
            atom_mask=np.ones_like(xyzo)[..., 0],
            residue_index=feature_dict["idx"][0].cpu().numpy() + 1,
            b_factors=lddt[0].cpu().numpy()[:, None].repeat(4, axis=1),
        )
        mean_plddt = float(lddt.mean())

        return predicted_protein, mean_plddt
