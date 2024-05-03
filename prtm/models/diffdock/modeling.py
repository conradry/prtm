import copy
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from prtm import protein
from prtm.models.diffdock import config
from prtm.models.diffdock.models.cg_model import CGModel
from prtm.models.diffdock.utils.diffusion_utils import (
    get_t_schedule,
    get_timestep_embedding,
)
from prtm.models.diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from prtm.models.diffdock.utils.inference_utils import InferenceDataset
from prtm.models.diffdock.utils.sampling import randomize_position, sampling
from prtm.models.diffdock.utils.visualise import PDBBlockFile
from prtm.models.esm.modeling import ESMForSequenceEmbedding
from prtm.utils.hub_utils import load_state_dict_from_zip

__all__ = ["DiffDockForLigandDocking"]

DIFFDOCK_MODEL_URLS = {
    "diffdock_v1": "https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip",
}
DIFFDOCK_MODEL_CONFIGS = {
    "diffdock_v1": config.ScoreModelConfig(),
}


def _get_diffdock_model_config(model_name: str) -> config.ScoreModelConfig:
    """Get the model config for a given model name."""
    return DIFFDOCK_MODEL_CONFIGS[model_name]


class DiffDockForLigandDocking:
    def __init__(
        self,
        model_name: str = "diffdock_v1",
    ):
        self.model_name = model_name
        self.cfg = _get_diffdock_model_config(model_name)

        t_to_sigma = partial(t_to_sigma_compl, model_cfg=self.cfg)
        timestep_emb_func = get_timestep_embedding(
            embedding_type=self.cfg.embedding_type,
            embedding_dim=self.cfg.sigma_embed_dim,
            embedding_scale=self.cfg.embedding_scale,
        )
        self.model = CGModel(t_to_sigma, timestep_emb_func, cfg=self.cfg)

        self.load_weights(DIFFDOCK_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        # LM embeddings to give to DiffDock come from ESM
        self.esm_embedder = ESMForSequenceEmbedding(model_name="esm2_650M")

    @classmethod
    @property
    def available_models(cls):
        return list(DIFFDOCK_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = load_state_dict_from_zip(
            weights_url, extract_member="score_model/best_ema_inference_epoch_model.pt"
        )

        msg = self.model.load_state_dict(state_dict, strict=True)

    def _init_trajectories(
        self, data_list: List[Batch], ligand: Chem.Mol
    ) -> List[PDBBlockFile]:
        """Initialize trajectories for the ligand docking diffusion process."""
        trajectory_list = []
        for graph in data_list:
            pdb = PDBBlockFile(ligand)
            pdb.add(ligand, 0, 0)
            pdb.add(
                (
                    graph["ligand"].pos.detach().cpu()
                    + graph.original_center.detach().cpu()
                ),
                part=1,
                order=1,
            )
            trajectory_list.append(pdb)

        return trajectory_list

    @torch.no_grad()
    def __call__(
        self,
        structure: protein.ProteinBase,
        ligand: Union[str, Chem.Mol],
        num_samples: int = 10,
        return_trajectory: bool = True,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
    ) -> Tuple[List[Chem.Mol], Dict[str, Any]]:
        """Find a docking pose for a ligand given a protein structure."""
        structure = structure.to_protein14()
        sequences = [
            structure.get_chain(chain_id).sequence() for chain_id in structure.chains
        ]
        lm_embeddings = [
            [emb.cpu() for emb in self.esm_embedder(sequences)[0]]
        ]  # list of list

        test_dataset = InferenceDataset(
            protein_structures=[structure],
            ligands=[ligand],
            lm_embeddings=lm_embeddings,
            cfg=self.cfg,
        )
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        orig_complex_graph = next(iter(test_loader))

        tr_schedule = get_t_schedule(
            inference_steps=inference_config.inference_steps,
            sigma_schedule=inference_config.sigma_schedule,
        )

        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(num_samples)]
        randomize_position(
            data_list,
            no_torsion=self.cfg.no_torsion,
            no_random=False,
            tr_sigma_max=self.cfg.tr_sigma_max,
            initial_noise_std_proportion=-1.0,
            choose_residue=False,
        )
        lig = orig_complex_graph.mol[0]
        if return_trajectory:
            trajectory_list = self._init_trajectories(data_list, lig)
        else:
            trajectory_list = None

        # run reverse diffusion
        data_list = sampling(
            data_list=data_list,
            model=self.model,
            inference_steps=inference_config.inference_steps,
            tr_schedule=tr_schedule,
            rot_schedule=tr_schedule,
            tor_schedule=tr_schedule,
            device=self.device,
            t_to_sigma=self.model.t_to_sigma,
            model_args=self.cfg,
            visualization_list=trajectory_list,
            batch_size=max(10, num_samples),
            no_final_step_noise=inference_config.no_final_step_noise,
            temp_sampling=[1.0, 1.0, 1.0],
            temp_psi=[0.0, 0.0, 0.0],
            temp_sigma_data=[0.5, 0.5, 0.5],
        )
        ligand_pos = np.asarray(
            [
                complex_graph["ligand"].pos.cpu().numpy()
                + orig_complex_graph.original_center.cpu().numpy()
                for complex_graph in data_list
            ]
        )

        final_poses = []
        for pos in ligand_pos:
            mol_pose = copy.deepcopy(lig)
            if self.cfg.remove_hs:
                mol_pose = Chem.RemoveAllHs(mol_pose)

            conf = mol_pose.GetConformer()
            for i in range(mol_pose.GetNumAtoms()):
                x, y, z = pos.astype(np.double)[i]
                conf.SetAtomPosition(i, Point3D(x, y, z))

            final_poses.append(mol_pose)

        aux_output = {}
        if return_trajectory:
            aux_output["pdb_trajectories"] = [
                trajectory.to_pdb() for trajectory in trajectory_list
            ]

        return final_poses, aux_output
