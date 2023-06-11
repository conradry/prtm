import os
from typing import Any, Dict, List, Tuple, Optional

import ml_collections as mlc
import numpy as np
import torch

from proteome.models.folding.openfold import config, data
from proteome.models.folding.openfold.data import (
    data_pipeline,
    feature_pipeline, 
    parsers,
)
from proteome.models.folding.openfold.model import model
from proteome.models.folding.openfold.np import protein
from proteome.models.folding.openfold.np.relax import relax
from proteome.models.folding.openfold.np.relax.utils import overwrite_b_factors
from proteome.models.folding.openfold.utils.tensor_utils import tensor_tree_map
from proteome.query import jackhmmer
from proteome.utils import hub_s3

OPENFOLD_MODEL_URLS = {
    "finetuning-3": "s3://openfold/openfold_params/finetuning_3.pt",
    "finetuning-4": "s3://openfold/openfold_params/finetuning_4.pt",
    "finetuning-5": "s3://openfold/openfold_params/finetuning_5.pt",
    "finetuning_ptm-2": "s3://openfold/openfold_params/finetuning_ptm_2.pt",
    "finetuning_no_templ_ptm-1": "s3://openfold/openfold_params/finetuning_no_templ_ptm_1.pt",
}
JACKHMMER_DBS = {
    # Order of tuple is (chunk_count, z_value, db_url)
    #"uniref90": (59, 135301051, 'https://storage.googleapis.com/alphafold-colab-asia/latest/uniref90_2021_03.fasta'),
    #"smallbfd": (17, 65984053, 'https://storage.googleapis.com/alphafold-colab-asia/latest/bfd-first_non_consensus_sequences.fasta'),
    #"mgnify": (71, 304820129, 'https://storage.googleapis.com/alphafold-colab-asia/latest/mgy_clusters_2019_05.fasta'),
    "uniref90": (2, 135301051, 'https://storage.googleapis.com/alphafold-colab-asia/latest/uniref90_2021_03.fasta'),
    "smallbfd": (2, 65984053, 'https://storage.googleapis.com/alphafold-colab-asia/latest/bfd-first_non_consensus_sequences.fasta'),
    "mgnify": (2, 304820129, 'https://storage.googleapis.com/alphafold-colab-asia/latest/mgy_clusters_2019_05.fasta'),
}
MGNIFY_MAX_HITS = 501


def _placeholder_template_feats(
    num_templates: int, num_res: int
) -> Dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((num_templates, num_res, 22), dtype=np.int64),
        "template_all_atom_positions": np.zeros(
            (num_templates, num_res, 37, 3), dtype=np.float32
        ),
        "template_all_atom_mask": np.zeros(
            (num_templates, num_res, 37), dtype=np.float32
        ),
        "template_domain_names": np.zeros((num_templates,), dtype=np.float32),
        "template_sum_probs": np.zeros((num_templates, 1), dtype=np.float32),
    }


def _get_model_config(model_name: str) -> mlc.ConfigDict:
    """Get the model config for a given model name."""
    # All `finetuning` models use the same config.
    return config.model_config(name=model_name.split("-")[0])


class OpenFoldForFolding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = model.AlphaFold(self.cfg)

        self.load_weights(model, OPENFOLD_MODEL_URLS[model_name])

        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def load_weights(self, model, weights_url):
        """Load weights from a weights url."""
        weights = hub_s3.load_state_dict_from_s3_url(weights_url)
        msg = self.model.load_state_dict(weights, strict=True)

    def _validate_input_sequence(self, sequence: str) -> None:
        """Validate that the input sequence is a valid amino acid sequence."""
        sequence = sequence.translate(str.maketrans("", "", " \n\t")).upper()
        aatypes = set("ACDEFGHIKLMNPQRSTVWY")
        if not set(sequence).issubset(aatypes):
            raise ValueError(
                f"Input sequence contains non-amino acid letters: {set(sequence) - aatypes}."
            )

    def _get_msas_jackhmmer(
        self, sequence: str
    ) -> List[Tuple[str, List[Dict[str, str]]]]:
        """Get MSAs for the input sequence using Jackhmmer."""
        # TODO: It would be nice to cache results somewhere given the time it takes to get MSAs.
        # TODO: Save fasta file to a temporary directory.
        # Save the sequence to a fasta file and run jackhmmer.
        with open("target.fasta", "w") as f:
            f.write(f">query\n{sequence}")

        dbs = []
        for db_name, (chunk_count, z_value, db_url) in JACKHMMER_DBS.items():
            print(f"Running jackhmmer on {db_name} database...")
            jackhmmer_runner = jackhmmer.Jackhmmer(
                database_path=db_url,
                get_tblout=True,
                num_streamed_chunks=chunk_count,
                streaming_callback=None,
                z_value=z_value,
            )
            dbs.append((db_name, jackhmmer_runner.query("target.fasta")))

        os.remove("target.fasta")

        return dbs

    def _unpack_jackhmmer_results(
        self,
        jackhmmer_results: List[Tuple[str, List[Dict[str, str]]]],
    ) -> Tuple[List, List]:
        """Unpack the results from jackhmmer into a list of MSAs and deletion matrices."""
        msas = []
        deletion_matrices = []

        for db_name, db_results in jackhmmer_results:
            unsorted_results = []
            for i, result in enumerate(db_results):
                msa, deletion_matrix, target_names = parsers.parse_stockholm(
                    result["sto"]
                )
                e_values_dict = parsers.parse_e_values_from_tblout(result["tbl"])
                e_values = [e_values_dict[t.split("/")[0]] for t in target_names]
                zipped_results = zip(msa, deletion_matrix, target_names, e_values)
                if i != 0:
                    # Only take query from the first chunk
                    zipped_results = [x for x in zipped_results if x[2] != "query"]
                unsorted_results.extend(zipped_results)

            sorted_by_evalue = sorted(unsorted_results, key=lambda x: x[3])
            db_msas, db_deletion_matrices, _, _ = zip(*sorted_by_evalue)
            if db_msas:
                if db_name == "mgnify":
                    db_msas = db_msas[:MGNIFY_MAX_HITS]
                    db_deletion_matrices = db_deletion_matrices[:MGNIFY_MAX_HITS]

                msas.append(db_msas)
                deletion_matrices.append(db_deletion_matrices)
                msa_size = len(set(db_msas))
                print(f"{msa_size} Sequences Found in {db_name}")

        return msas, deletion_matrices

    def _featurize_input(
        self, sequence: str, msas: List[Tuple], deletion_matrices: List[Tuple]
    ) -> Dict[str, torch.Tensor]:
        """Prepare the input features for a given sequence and its MSAs and deletion matrices."""
        num_res = len(sequence)
        num_templates = 1  # dummy number --- is ignored

        feature_dict = {}
        feature_dict.update(
            data_pipeline.make_sequence_features(sequence, "test", num_res)
        )
        feature_dict.update(
            data_pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices)
        )
        feature_dict.update(_placeholder_template_feats(num_templates, num_res))

        pipeline = feature_pipeline.FeaturePipeline(self.cfg.data)  # type: ignore
        processed_feature_dict = pipeline.process_features(feature_dict, mode="predict")
        processed_feature_dict = tensor_tree_map(
            lambda t: t.to(self.device), processed_feature_dict
        )

        assert isinstance(processed_feature_dict, dict)
        return processed_feature_dict

    @torch.no_grad()
    def fold(self, sequence: str) -> Tuple[protein.Protein, float]:
        """Fold a protein sequence."""
        self._validate_input_sequence(sequence)

        # Get MSAs for the input sequence using Jackhmmer.
        jackhmmer_results = self._get_msas_jackhmmer(sequence)
        msas, deletion_matrices = self._unpack_jackhmmer_results(jackhmmer_results)

        # Prepare the input features for a given sequence and its MSAs and deletion matrices.
        feature_dict = self._featurize_input(sequence, msas, deletion_matrices)

        res = self.model(feature_dict)
        
        # Unpack to a protein and a plddt confidence metric.
        mean_plddt = float(res['plddt'].mean())
        b_factors = res['plddt'][:, None] * res['final_atom_mask']
        predicted_protein = protein.from_prediction(feature_dict, res, b_factors=b_factors)

        return predicted_protein, mean_plddt
