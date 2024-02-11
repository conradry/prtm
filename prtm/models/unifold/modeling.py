import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import pdb
from contextlib import nullcontext

from prtm.models.unifold.config import model_config
from prtm.models.unifold.data import process, protein, residue_constants, utils
from prtm.models.unifold.data.process_multimer import (
    add_assembly_features, convert_monomer_features, merge_msas,
    pair_and_merge, post_process)
from prtm.models.unifold.dataset import UnifoldDataset, make_data_config
from prtm.models.unifold.inference import automatic_chunk_size
from prtm.models.unifold.input_validation import validate_input
from prtm.models.unifold.mmseqs import get_null_template, get_template
from prtm.models.unifold.modules.alphafold import AlphaFold
from prtm.models.unifold.msa import parsers, pipeline
from prtm.models.unifold.symmetry import (
    UFSymmetry, assembly_from_prediction, uf_symmetry_config
)
from prtm.models.unifold.symmetry.dataset import get_pseudo_residue_feat
from prtm.models.unifold.symmetry.utils import get_transform
from prtm.models.unifold.utils import numpy_seed, tensor_tree_map
from prtm.protein import PDB_CHAIN_IDS
# from prtm import protein
from prtm.query.mmseqs import MMSeqs2
from prtm.utils import hub_utils

__all__ = ["UniFoldForFolding"]

UNIFOLD_MODEL_URLS = {
    "model_2_ft": "https://github.com/dptech-corp/Uni-Fold/releases/download/v2.0.0/unifold_params_2022-08-01.tar.gz",
    "multimer_ft": "https://github.com/dptech-corp/Uni-Fold/releases/download/v2.0.0/unifold_params_2022-08-01.tar.gz",
    "uf_symmetry": "https://github.com/dptech-corp/Uni-Fold/releases/download/v2.2.0/uf_symmetry_params_2022-09-06.tar.gz",
}

UNIFOLD_MODEL_CONFIGS = {
    "uf_symmetry": uf_symmetry_config(),
    "multimer_ft": model_config("multimer_ft"),
    "model_2_ft": model_config("model_2_ft"),
}


def _get_model_config(model_name: str):
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return UNIFOLD_MODEL_CONFIGS[model_name]


class UniFoldForFolding:
    def __init__(
        self,
        model_name: str = "model_2_ft",
        use_templates: bool = False,
        symmetry_group: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        if model_name != "uf_symmetry":
            assert symmetry_group in [None, "C1"], (
                "Symmetry group must be None or 'C1' for this model!"
            )
            self.model = AlphaFold(self.cfg)
        else:
            assert model_name == "uf_symmetry", (
                "To use symmetric folding set model_name='uf_symmetry'"
            )
            self.model = UFSymmetry(self.cfg)

        self.load_weights(UNIFOLD_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.model.inference_mode()
        self.symmetry_group = symmetry_group
        self.is_symmetry = symmetry_group not in [None, "C1"]
        self.use_templates = use_templates

        self.msa_caller = MMSeqs2(
            user_agent="unifold_prtm", use_templates=use_templates
        )
        self.paired_msa_caller = MMSeqs2(
            user_agent="unifold_prtm", use_templates=False, use_pairing=True
        )

        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    @classmethod
    @property
    def available_models(cls):
        return list(UNIFOLD_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        if self.model_name == "model_2_ft":
            extract_member = "monomer.unifold.pt"
        elif self.model_name == "multimer_ft":
            extract_member = "multimer.unifold.pt"
        elif self.model_name == "uf_symmetry":
            extract_member = "uf_symmetry.pt"
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        state_dict = hub_utils.load_state_dict_from_tar_gz_url(
            weights_url,
            extract_member=extract_member,
            model_name=f"unifold_{self.model_name}.pth",
            map_location="cpu",
        )["ema"]["params"]
        state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=True)

    def msa_pipeline(self, sequences: List[str]):
        """Run the MSA pipeline."""
        unique_sequences = list(set(sequences))
        unpaired_msa = self.msa_caller.query(sequences)
        if len(unique_sequences) > 1:
            paired_msa = self.paired_msa_caller.query(sequences)
        else:
            homooligomers_num = len(sequences)
            paired_a3m_lines = []
            for i in range(0, homooligomers_num):
                paired_a3m_lines.append(
                    ">" + str(self.msa_caller.N + i) + "\n" + sequences[0] + "\n"
                )
            paired_msa = {"paired_msas": "".join(paired_a3m_lines)}

        return unpaired_msa | paired_msa

    def _featurize_input(
        self, sequences: List[str], msa_templates_dict: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Prepare the input features for a given sequence and its MSAs and deletion matrices."""
        if "templates" in msa_templates_dict:
            template_features = []
            for i, template_path in enumerate(msa_templates_dict["templates"]):
                if template_path is not None:
                    template_feature = get_template(
                        msa_templates_dict["msas"][i], template_path, sequences[i]
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = get_null_template(sequences[i])
                else:
                    template_feature = get_null_template(sequences[i])
                template_features.append(template_feature)
        else:
            template_features = [get_null_template(s) for s in sequences]

        unpaired_msa = msa_templates_dict["msas"]
        paired_msa = msa_templates_dict.get("paired_msas", None)

        all_chain_features_list = []
        for chain_idx, sequence in enumerate(sequences):
            chain_name = PDB_CHAIN_IDS[chain_idx]
            sequence_features = pipeline.make_sequence_features(
                sequence=sequence, chain_name=chain_name
            )
            monomer_msa = parsers.parse_a3m(unpaired_msa[chain_idx])
            msa_features = pipeline.make_msa_features([monomer_msa])
            feature_dict = {
                **sequence_features,
                **msa_features,
                **template_features[chain_idx],
            }
            monomer_feature = convert_monomer_features(feature_dict)

            multimer_msa = parsers.parse_a3m(paired_msa[chain_idx])
            pair_feature_dict = pipeline.make_msa_features([multimer_msa])

            if len(sequences) > 1:
                pair_feature_dict = utils.convert_all_seq_feature(pair_feature_dict)
                for key in [
                    "msa_all_seq",
                    "msa_species_identifiers_all_seq",
                    "deletion_matrix_all_seq",
                ]:
                    monomer_feature[key] = pair_feature_dict[key]
            elif np.any(pair_feature_dict["msa"]):
                monomer_feature["msa"], monomer_feature["deletion_matrix"] = merge_msas(
                    monomer_feature["msa"],
                    monomer_feature["deletion_matrix"],
                    pair_feature_dict["msa"],
                    pair_feature_dict["deletion_matrix"],
                )
            all_chain_features_list.append(monomer_feature)

        all_chain_features = add_assembly_features(all_chain_features_list)

        asym_len = np.array(
            [c["seq_length"] for c in all_chain_features], dtype=np.int64
        )
        if len(sequences) == 1:
            all_chain_features = all_chain_features[0]
        else:
            all_chain_features = pair_and_merge(all_chain_features)
            all_chain_features = post_process(all_chain_features)

        all_chain_features["asym_len"] = asym_len

        num_iters = self.cfg.data.common.max_recycling_iters
        is_distillation = False

        all_chain_features["num_recycling_iters"] = int(num_iters)
        all_chain_features["use_clamped_fape"] = 1
        all_chain_features["is_distillation"] = int(is_distillation)
        if is_distillation and "msa_chains" in all_chain_features:
            all_chain_features.pop("msa_chains")

        num_res = int(all_chain_features["seq_length"])
        cfg, feature_names = make_data_config(
            self.cfg.data, 
            mode="predict", 
            num_res=num_res, 
            is_multimer=(len(sequences) > 1) or self.is_symmetry, 
            use_templates=self.use_templates,
        )

        # Conditional fixed seed context
        with (
            numpy_seed(self.random_seed, key="protein_feature") 
            if self.random_seed is not None 
            else nullcontext() as _
        ):
            all_chain_features["crop_and_fix_size_seed"] = np.random.randint(0, 63355)
            all_chain_features = utils.filter(
                all_chain_features, desired_keys=feature_names
            )
            all_chain_features = {
                k: torch.tensor(v) for k, v in all_chain_features.items()
            }
            with torch.no_grad():
                all_chain_features = process.process_features(
                    all_chain_features, self.cfg.data.common, self.cfg.data["predict"]
                )

        print("Symmetry group", self.symmetry_group)
        if self.symmetry_group is not None:
            print("Adding symmetry features")
            all_chain_features["symmetry_opers"] = torch.tensor(
                get_transform(self.symmetry_group), dtype=float
            )[None]
            all_chain_features["pseudo_residue_feat"] = torch.tensor(
                get_pseudo_residue_feat(self.symmetry_group), dtype=float
            )[None]
            all_chain_features["num_asym"] = torch.max(all_chain_features["asym_id"])[
                None
            ]

        return all_chain_features

    @torch.no_grad()
    def __call__(
        self,
        sequences_dict: Dict[str, str],
        max_recycling_iters: int,
        num_ensembles: int,
        chunk_size: int = 128,
    ):  # -> Tuple[protein.Protein37, Dict[str, Any]]:
        """Fold a protein sequence."""
        sequences_dict = validate_input(
            sequences_dict, self.symmetry_group, 6, 3000, 3000
        )[0]
        sequences = list(sequences_dict.values())

        # TODO: Actually need to update the model with these params?
        self.cfg.data.common.max_recycling_iters = max_recycling_iters
        self.cfg.globals.max_recycling_iters = max_recycling_iters
        self.cfg.data.predict.num_ensembles = num_ensembles
        self.cfg.globals.chunk_size = chunk_size
        self.cfg.data.common.use_templates = self.use_templates
        self.cfg.data.common.is_multimer = len(sequences) > 1

        batch = self._featurize_input(sequences, self.msa_pipeline(sequences))
        batch = UnifoldDataset.collater([batch])

        seq_len = batch["aatype"].shape[-1]
        chunk_size, block_size = automatic_chunk_size(
            seq_len,
            self.device,
            is_bf16=False,
        )
        self.model.globals.chunk_size = chunk_size
        self.model.globals.block_size = block_size

        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=self.device) for k, v in batch.items()
            }
            out = self.model(batch)

        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x

        # Toss out the recycling dimensions --- we don't need them anymore
        batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
        batch = tensor_tree_map(to_float, batch)
        out = tensor_tree_map(lambda t: t[0, ...], out)
        out = tensor_tree_map(to_float, out)
        batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"]
        mean_plddt = np.mean(plddt)
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        # TODO: , may need to reorder chains, based on entity_ids
        if self.symmetry_group is None:
            cur_protein = protein.from_prediction(
                features=batch, result=out, b_factors=plddt_b_factors
            )
        else:
            plddt_b_factors_assembly = np.concatenate(
                [plddt_b_factors for _ in range(batch["symmetry_opers"].shape[0])]
            )

            cur_protein = assembly_from_prediction(
                result=out,
                b_factors=plddt_b_factors_assembly,
            )

        # structure = protein.Protein37(
        #    atom_positions=res["final_atom_positions"].cpu().numpy(),
        #    aatype=feature_dict["aatype"].cpu().numpy()[:, 0],
        #    atom_mask=res["final_atom_mask"].cpu().numpy(),
        #    residue_index=feature_dict["residue_index"].cpu().numpy()[:, 0] + 1,
        #    b_factors=b_factors.cpu().numpy(),
        #    chain_index=chain_index,
        # )

        # return structure, {"mean_plddt": mean_plddt}
        return cur_protein, {"mean_plddt": mean_plddt}
