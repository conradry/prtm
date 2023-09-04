import random
from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from proteome import protein
from proteome.constants import residue_constants
from proteome.models.design.eigenfold import config, schedule
from proteome.models.design.eigenfold.sampling import logp, reverse_sample
from proteome.models.design.eigenfold.score_model import ScoreModel
from proteome.models.design.eigenfold.sde import PolymerSDE
from proteome.models.folding.omegafold.modeling import OmegaFoldForFolding
from proteome.models.folding.omegafold.config import InferenceConfig as OFInferenceConfig
from torch_geometric.data import HeteroData

MODEL_URLS = {
    "model1": "https://github.com/bjing2016/EigenFold/raw/master/pretrained_model/epoch_7.pt",
}

MODEL_CONFIGS = {
    "model1": config.ScoreModelConfig(),
}
SCHEDULES = {"entropy": schedule.EntropySchedule, "rate": schedule.RateSchedule}


def _get_model_config(model_name: str) -> config.ScoreModelConfig:
    """Get the model config for a given model name."""
    return MODEL_CONFIGS[model_name]


class _OmegaFoldForGraphEmbedding(OmegaFoldForFolding):
    @torch.no_grad()
    def embed(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed a protein sequence returning a tuple of node and edge representations for the structure, respectively.
        """
        self._validate_input_sequence(sequence)

        # Prepare the input features for a given sequence and its MSAs and deletion matrices.
        list_of_feature_dict = self._featurize_input(sequence)

        res = self.model(
            list_of_feature_dict,
            predict_with_confidence=True,
            return_embeddings=True,
            fwd_cfg=OFInferenceConfig(),
        )

        return (res["node_repr"], res["edge_repr"])


class EigenFoldForFoldSampling:
    def __init__(
        self,
        model_name: str = "model1",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = ScoreModel(self.cfg)

        self.load_weights(MODEL_URLS[model_name])
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.embedder = _OmegaFoldForGraphEmbedding()

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.cache = {}

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"eigenfold_{self.model_name}.pt",
            progress=True,
            map_location="cpu",
        )["model"]
        self.model.load_state_dict(state_dict)

    def _get_dense_edges(self, n: int) -> torch.Tensor:
        atom_ids = np.arange(n)
        src, dst = np.repeat(atom_ids, n), np.tile(atom_ids, n)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_idx = np.stack([src, dst])
        return torch.tensor(edge_idx)

    def _get_score_fn(self, data: HeteroData, key: str = "resi") -> Callable:
        data = deepcopy(data)
        data.to(self.device)
        sde = data.sde

        @torch.no_grad()
        def score_fn(Y, t, k):
            data[key].pos = Y[: data["resi"].num_nodes]
            data[key].node_t = torch.ones(data.resi_sde.N, device=self.device) * t
            data.score_norm = sde.score_norm(t, k, adj=True)
            data["sidechain"].pos = Y[data["resi"].num_nodes :]
            return self.model.enn(data)

        return score_fn

    def _get_schedule(
        self,
        inference_config: config.InferenceConfig,
        sde: PolymerSDE,
        full: bool = False,
    ) -> schedule.Schedule:
        return SCHEDULES[inference_config.schedule_type](
            sde,
            Hf=inference_config.Hf,
            rmsd_max=0,
            step=inference_config.elbo_step if full else inference_config.step,
            cutoff=inference_config.cutoff,
            kmin=inference_config.kmin,
            tmin=inference_config.tmin,
            alpha=0 if full else inference_config.alpha,
            beta=1 if full else inference_config.beta,
        )

    @torch.no_grad()
    def sample_fold(
        self,
        sequence: str,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[protein.Protein, float]:
        """Design a random protein structure."""
        seqlen = len(sequence)
        if sequence in self.cache:
            node_repr, edge_repr = self.cache[sequence]
        else:
            node_repr, edge_repr = self.embedder.embed(sequence)
            self.cache = {"sequence": (node_repr, edge_repr)}

        data = HeteroData()
        data.skip = False
        seqlen = node_repr.shape[0]
        data["resi"].num_nodes = seqlen
        data["resi"].edge_index = self._get_dense_edges(seqlen)

        sde = PolymerSDE(N=seqlen, a=self.cfg.sde_a, b=self.cfg.sde_b)
        sde.make_schedule(
            Hf=inference_config.Hf,
            step=inference_config.step,
            tmin=inference_config.tmin,
        )

        data.resi_sde = data.sde = sde
        data["resi"].node_attr = node_repr
        src, dst = data["resi"].edge_index[0], data["resi"].edge_index[1]
        data["resi"].edge_attr_ = torch.cat(
            [edge_repr[src, dst], edge_repr[dst, src]], -1
        )

        sched = self._get_schedule(inference_config, sde)
        sched_full = self._get_schedule(inference_config, sde, full=True)
        score_fn = self._get_score_fn(data)

        data.Y = reverse_sample(
            score_fn,
            sde,
            sched,
            inference_config.cutoff,
            device=self.device,
        )

        data.elbo_Y = (
            logp(data.Y, score_fn, sde, sched_full, device=self.device)
            if inference_config.elbo
            else np.nan
        )

        coords = np.zeros((seqlen, 4, 3))
        coords[:, 1, :] = data.Y
        atom_mask = np.zeros((seqlen, 4))
        atom_mask[:, 1] = 1

        n = len(coords)
        structure = protein.Protein(
            atom_positions=coords,
            aatype=np.array(n * [residue_constants.restype_order_with_x["G"]]),
            atom_mask=atom_mask,
            residue_index=np.arange(0, n) + 1,
            b_factors=np.zeros((n, 4)),
            chain_index=np.zeros((n,), dtype=np.int32),
        )

        return structure, data.elbo_Y
