import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Tuple

from proteome import parsers
from proteome.query import jackhmmer


def _unpack_jackhmmer_results(
    jackhmmer_results: List[Tuple[str, List[Dict[str, str]]]],
    mgnify_max_hits: int = 501,
) -> Tuple[List, List]:
    """Unpack the results from jackhmmer into a list of MSAs and deletion matrices."""
    msas = []
    deletion_matrices = []

    for db_name, db_results in jackhmmer_results:
        unsorted_results = []
        for i, result in enumerate(db_results):
            msa, deletion_matrix, target_names = parsers.parse_stockholm(result["sto"])
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
                db_msas = db_msas[:mgnify_max_hits]
                db_deletion_matrices = db_deletion_matrices[:mgnify_max_hits]

            msas.append(db_msas)
            deletion_matrices.append(db_deletion_matrices)
            msa_size = len(set(db_msas))
            print(f"{msa_size} sequences found in {db_name}")

    return msas, deletion_matrices


def alphafold_jackhmmer_query_pipeline(sequence: str) -> Tuple[List, List]:
    """Get MSAs for the input sequence using Jackhmmer."""
    # TODO: Save fasta file to a temporary directory.
    # Save the sequence to a fasta file and run jackhmmer.
    with open("target.fasta", "w") as f:
        f.write(f">query\n{sequence}")

    jackhmmer_dbs = {
        # Order of tuple is (chunk_count, z_value, db_url)
        "uniref90": (
            2,  # 59,
            135301051,
            "https://storage.googleapis.com/alphafold-colab-asia/latest/uniref90_2021_03.fasta",
        ),
        "smallbfd": (
            2,  # 17,
            65984053,
            "https://storage.googleapis.com/alphafold-colab-asia/latest/bfd-first_non_consensus_sequences.fasta",
        ),
        "mgnify": (
            2,  # 71,
            304820129,
            "https://storage.googleapis.com/alphafold-colab-asia/latest/mgy_clusters_2019_05.fasta",
        ),
    }

    dbs = []
    for db_name, (chunk_count, z_value, db_url) in jackhmmer_dbs.items():
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
    msas, deletion_matrices = _unpack_jackhmmer_results(dbs)

    return msas, deletion_matrices


@dataclass
class QueryPipelines:
    alphafold_jackhmmer: Callable = alphafold_jackhmmer_query_pipeline
