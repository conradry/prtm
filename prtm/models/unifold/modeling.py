import random
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from prtm import protein
from prtm.query.mmseqs import MMSeqs2
from prtm.utils import hub_utils
from prtm.models.unifold.config import model_config
from prtm.models.unifold.data import process, residue_constants, utils
from prtm.models.unifold.data.process_multimer import (
    add_assembly_features, 
    convert_monomer_features, 
    merge_msas,
    pair_and_merge, 
    post_process,
)
from prtm.models.unifold.dataset import make_data_config
from prtm.models.unifold.inference import automatic_chunk_size
from prtm.models.unifold.input_validation import validate_input
from prtm.models.unifold.modules.alphafold import AlphaFold
from prtm.models.unifold.symmetry import UFSymmetry, uf_symmetry_config
from prtm.models.unifold.symmetry.dataset import get_pseudo_residue_feat
from prtm.models.unifold.symmetry.utils import get_transform
from prtm.models.unifold.utils import collate_dict, numpy_seed, tensor_tree_map
from prtm.models.unifold.msa import templates, pipeline, parsers
from prtm.models.unifold.msa.tools import hhsearch

__all__ = ["UniFoldForFolding"]

UNIFOLD_MODEL_URLS = {
    # UniFold trained models
    "model_2_ft": "https://github.com/dptech-corp/Uni-Fold/releases/download/v2.0.0/unifold_params_2022-08-01.tar.gz",
    "multimer_ft": "https://github.com/dptech-corp/Uni-Fold/releases/download/v2.0.0/unifold_params_2022-08-01.tar.gz",
    "uf_symmetry": "https://github.com/dptech-corp/Uni-Fold/releases/download/v2.2.0/uf_symmetry_params_2022-09-06.tar.gz",
    # AlphaFold2 trained models
    "model_1_af2": "https://drive.google.com/uc?id=1vW1oZAI2ejeVUPQXusfN55Nf1_ldjrG6",
    "model_2_af2": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "model_3_af2": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "model_4_af2": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "model_5_af2": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "multimer_1_af2_v3": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "multimer_2_af2_v3": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "multimer_3_af2_v3": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "multimer_4_af2_v3": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
    "multimer_5_af2_v3": "https://drive.google.com/drive/folders/1eCd-fh6uf9UGp8uwx9PVxk8hFP6zDBZ6",
}

UNIFOLD_MODEL_CONFIGS = {
    # UniFold trained models
    "uf_symmetry": uf_symmetry_config(),
    "multimer_ft": model_config("multimer_ft"),
    "model_2_ft": model_config("model_2_ft"),
    # AlphaFold2 trained models
    "model_1_af2": model_config("model_1_af2"),
    "model_2_af2": model_config("model_2_af2"),
    "model_3_af2": model_config("model_3_af2"),
    "model_4_af2": model_config("model_4_af2"),
    "model_5_af2": model_config("model_5_af2"),
    "multimer_1_af2_v3": model_config("multimer_af2_v3"),
    "multimer_2_af2_v3": model_config("multimer_af2_v3"),
    "multimer_3_af2_v3": model_config("multimer_af2_v3"),
    "multimer_4_af2_v3": model_config("multimer_af2_model45_v3"),
    "multimer_5_af2_v3": model_config("multimer_af2_model45_v3"),
}


def _get_model_config(model_name: str):
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return UNIFOLD_MODEL_CONFIGS[model_name]


def get_null_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features


def get_template(
    a3m_lines: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)


class UniFoldForFolding:
    def __init__(
        self,
        model_name: str = "model_2_ft",
        use_templates: bool = False,
        symmetry_group: Optional[str] = None,
        random_seed: Optional[int] = None,
        min_sequence_length: int = 6,
        max_monomer_length: int = 3000,
        max_multimer_length: int = 3000,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        if model_name != "uf_symmetry":
            assert symmetry_group in [
                None,
                "C1",
            ], "Symmetry group must be None or 'C1' for this model!"
            self.model = AlphaFold(self.cfg)
        else:
            assert (
                model_name == "uf_symmetry"
            ), "To use symmetric folding set model_name='uf_symmetry'"
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

        self.min_sequence_length = min_sequence_length
        self.max_monomer_length = max_monomer_length
        self.max_multimer_length = max_multimer_length

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
    def available_models(cls) -> List[str]:
        return list(UNIFOLD_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        if self.model_name in ["model_2_ft", "multimer_ft", "uf_symmetry"]:
            member_keys = {
                "model_2_ft": "monomer.unifold.pt",
                "multimer_ft": "multimer.unifold.pt",
                "uf_symmetry": "uf_symmetry.pt",
            }
            extract_member = member_keys[self.model_name]
            load_fn = hub_utils.load_state_dict_from_tar_gz_url
        elif self.model_name in UNIFOLD_MODEL_URLS:
            # Must be an AlphaFold2 model
            member_keys = {
                "model_1_af2": "params_model_1.pth",
                "model_2_af2": "params_model_2.pth",
                "model_3_af2": "params_model_3.pth",
                "model_4_af2": "params_model_4.pth",
                "model_5_af2": "params_model_5.pth",
                "multimer_1_af2_v3": "params_model_1_multimer_v3.pth",
                "multimer_2_af2_v3": "params_model_2_multimer_v3.pth",
                "multimer_3_af2_v3": "params_model_3_multimer_v3.pth",
                "multimer_4_af2_v3": "params_model_4_multimer_v3.pth",
                "multimer_5_af2_v3": "params_model_5_multimer_v3.pth",
            }
            extract_member = f"unifold_converted/{member_keys[self.model_name]}"
            load_fn = hub_utils.load_state_dict_from_gdrive_zip
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        state_dict = load_fn(
            weights_url,
            extract_member=extract_member,
            name_prefix="unifold",
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
            chain_name = protein.PDB_CHAIN_IDS[chain_idx]
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

        if self.symmetry_group is not None:
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
    ) -> Tuple[protein.Protein37, Dict[str, Any]]:
        """Fold a protein sequence."""
        sequences_dict = validate_input(
            sequences_dict,
            self.symmetry_group,
            self.min_sequence_length,
            self.max_monomer_length,
            self.max_multimer_length,
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
        batch = collate_dict([batch])

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

        if self.symmetry_group is None:
            if "asym_id" in batch:
                chain_index = batch["asym_id"] - 1
            else:
                chain_index = np.zeros_like((batch["aatype"]))

            atom_positions = out["final_atom_positions"].copy()
            atom_mask = out["final_atom_mask"].copy()
            aatype = batch["aatype"]
            residue_index = batch["residue_index"] + 1
            b_factors = plddt_b_factors
        else:
            chain_index = out["expand_batch"]["asym_id"]
            atom_positions = out["expand_final_atom_positions"].copy()
            atom_mask = out["expand_final_atom_mask"].copy()
            aatype = out["expand_batch"]["aatype"]
            residue_index = out["expand_batch"]["residue_index"]
            b_factors = np.concatenate(
                [plddt_b_factors for _ in range(batch["symmetry_opers"].shape[0])]
            )

        # Surgery on the atom positions and mask to swap order of position of
        # CB and O atoms to match the prtm convention; bfactors are
        # constant for each resiude so no need to swap them
        atom_positions[:, [3, 4]] = atom_positions[:, [4, 3]]
        atom_mask = out["final_atom_mask"].copy()
        atom_mask[:, [3, 4]] = atom_mask[:, [4, 3]]
        structure = protein.Protein37(
            aatype=batch["aatype"],
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            residue_index=batch["residue_index"] + 1,
            chain_index=chain_index,
            b_factors=100 * plddt_b_factors,
        )

        return structure, {"mean_plddt": mean_plddt}
