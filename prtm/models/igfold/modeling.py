import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from prtm.constants.residue_constants import restype_order
from prtm.models.antiberty.modeling import _AntiBERTyBase
from prtm.models.igfold import config
from prtm.models.igfold.model import IgFold
from prtm.protein import PDB_CHAIN_IDS, Protein5
from prtm.utils import hub_utils

__all__ = ["IgFoldForFolding"]

IGFOLD_MODEL_URLS = {
    "igfold_1": "https://files.pythonhosted.org/packages/e7/6f/49407902f1cdbb9fe3b32c4f98f9b7c425adb09772cd0207ca72a9fe4f6b/igfold-0.4.0.tar.gz",
    "igfold_2": "https://files.pythonhosted.org/packages/e7/6f/49407902f1cdbb9fe3b32c4f98f9b7c425adb09772cd0207ca72a9fe4f6b/igfold-0.4.0.tar.gz",
    "igfold_3": "https://files.pythonhosted.org/packages/e7/6f/49407902f1cdbb9fe3b32c4f98f9b7c425adb09772cd0207ca72a9fe4f6b/igfold-0.4.0.tar.gz",
    "igfold_5": "https://files.pythonhosted.org/packages/e7/6f/49407902f1cdbb9fe3b32c4f98f9b7c425adb09772cd0207ca72a9fe4f6b/igfold-0.4.0.tar.gz",
}
IGFOLD_MODEL_CONFIGS = {
    "igfold_1": config.IgFoldConfig(),
    "igfold_2": config.IgFoldConfig(),
    "igfold_3": config.IgFoldConfig(),
    "igfold_5": config.IgFoldConfig(),
}


def _get_model_config(model_name: str) -> config.IgFoldConfig:
    """Get the model config for a given model name."""
    return IGFOLD_MODEL_CONFIGS[model_name]


class _AntiBERTyForIgFoldInput(_AntiBERTyBase):
    def _prepare_sequences(
        self, sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes an amino acid sequence and returns a tokenized tensor along
        with an attention mask for the sequence.
        """
        sequences = [list(seq) for seq in sequences]
        # Replace masked residues with [MASK]
        for seq in sequences:
            for i, res in enumerate(seq):
                if res == "_":
                    seq[i] = "[MASK]"

        # Tokenize the sequence
        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        return tokens, attention_mask

    @torch.no_grad()
    def __call__(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed a single sequence."""
        tokens, attention_mask = self._prepare_sequences(sequences)
        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        # gather embeddings
        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = list(embeddings.detach())

        for i, a in enumerate(attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i][-1]

        # gather attention matrices
        attentions = outputs.attentions
        attentions = torch.stack(attentions, dim=1)
        attentions = list(attentions.detach())

        for i, a in enumerate(attention_mask):
            attentions[i] = attentions[i][:, :, a == 1]
            attentions[i] = attentions[i][:, :, :, a == 1]

        return embeddings, attentions


class IgFoldForFolding:
    def __init__(
        self,
        model_name: str = "igfold_1",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = IgFold(self.cfg)

        self.load_weights(IGFOLD_MODEL_URLS[model_name])
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.antiberty_embedder = _AntiBERTyForIgFoldInput(model_name="base")

        # Make sure model devices are the same
        self.antiberty_embedder.model = self.antiberty_embedder.model.to(self.device)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    @classmethod
    @property
    def available_models(cls):
        return list(IGFOLD_MODEL_CONFIGS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = hub_utils.load_state_dict_from_tar_gz_url(
            weights_url,
            extract_member=f"igfold-0.4.0/igfold/trained_models/IgFold/{self.model_name}.ckpt",
            model_name=f"{self.model_name}.pth",
            map_location="cpu",
        )["state_dict"]
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def __call__(
        self, sequences_dict: Dict[str, str]
    ) -> Tuple[Protein5, Dict[str, Any]]:
        """Embed a single sequence."""
        assert all([chain in ["H", "L"] for chain in sequences_dict.keys()])
        assert (
            len(sequences_dict) <= 2
        ), "Only a single heavy and light chain are supported!"
        assert len(sequences_dict) > 0, "At least one sequence must be provided!"

        embeddings, attentions = self.antiberty_embedder(list(sequences_dict.values()))
        embeddings = [e[1:-1].unsqueeze(0) for e in embeddings]
        attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in attentions]

        model_in = config.IgFoldInput(
            embeddings=embeddings,
            attentions=attentions,
            template_coords=None,
            template_mask=None,
            return_embeddings=True,
        )

        model_out = self.model(model_in)
        model_out = self.model.gradient_refine(model_in, model_out)
        score = model_out.prmsd.quantile(0.9)

        prmsd = rearrange(
            model_out.prmsd,
            "b (l a) -> b l a",
            a=4,
        )
        model_out.prmsd = prmsd

        coords = model_out.coords.squeeze(0).detach()
        res_rmsd = prmsd.square().mean(dim=-1).sqrt().squeeze(0)

        # Extract the full sequence and the chain ids
        full_seq = "".join(list(sequences_dict.values()))
        chains = list(sequences_dict.keys())
        delims = np.cumsum([len(s) for s in sequences_dict.values()]).tolist()

        # Encode the sequence as indices
        aatype = np.array([restype_order[res] for res in full_seq])

        # Prepare the protein class inputs
        coords = coords.cpu().numpy()
        atom_mask = np.ones((len(full_seq), 5), dtype=np.float32)
        b_factors = np.repeat(res_rmsd.cpu().numpy()[:, None], 5, axis=1)

        # Construct the chain delimited fields
        seq_chain_ids = list(sequences_dict.keys())
        residue_index = 1 + np.arange(0, delims[0])
        chain_index = np.array(delims[0] * [PDB_CHAIN_IDS.index(seq_chain_ids[0])])
        if len(sequences_dict) == 2:
            residue_index = np.concatenate(
                [
                    residue_index,
                    1 + np.arange(0, delims[1] - delims[0]),
                ],
                axis=0,
            )
            chain_index = np.concatenate(
                [
                    chain_index,
                    np.array(
                        (delims[1] - delims[0])
                        * [PDB_CHAIN_IDS.index(seq_chain_ids[1])]
                    ),
                ],
                axis=0,
            )

        # IgFold swaps the CB and O atom that is canonical in prtm
        atom_order = [0, 1, 2, 4, 3]

        # Atom mask and b_factors are the same for all atoms
        # so we don't need to permute the order
        structure = Protein5(
            atom_positions=coords[:, atom_order],
            aatype=aatype,
            atom_mask=atom_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            b_factors=b_factors,
        )

        return structure, {"score": score.item()}
