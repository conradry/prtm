# Cropping input features of a monomer, e.g., reducing the MSA size or template number
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology
#
"""AF2Complex: protein complex structure prediction with deep learning"""
import json
import os
import pickle
import random
import re
import sys
import time
from fileinput import hook_compressed
from typing import Dict, Type

# from memory_profiler import profile

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
af_dir = os.path.join(parent_dir, "src")
sys.path.append(af_dir)

import numpy as np
from absl import app, flags, logging
from alphafold.data.complex import (initialize_template_feats,
                                    read_af2c_target_file)

# Internal import (7716).


flags.DEFINE_string(
    "target_lst_path",
    None,
    "Path to a file containing a list of targets "
    "in any monomer, homo- or hetero-oligomers "
    "configurations. For example, TarA is a monomer, TarA:2 is a dimer "
    "of two TarAs. TarA:2/TarB is a trimer of two TarA and one TarB, etc.",
)
flags.DEFINE_string(
    "output_dir", None, "Path to a directory that will " "store the results."
)
flags.DEFINE_string(
    "feature_dir",
    None,
    "Path to a directory that will " "contains pre-genearted feature in pickle format.",
)
flags.DEFINE_enum(
    "model_preset",
    "monomer_ptm",
    ["monomer", "monomer_casp14", "monomer_ptm", "multimer", "multimer_np"],
    "Choose preset model configuration - the monomer model, "
    "the monomer model with extra ensembling, monomer model with "
    "pTM head (monomer_ptm), multimer on multimer features with paired MSA generated by AF-Multimer data pipeline "
    "(multimer), and multimer model on monomer features and unpaired MSAs (mulitmer_np).",
)
flags.DEFINE_enum(
    "msa_pairing",
    None,
    ["all", "cyclic", "linear", "custom"],
    "Choose MSA pairing mode if using input features of monomers - By default no action, "
    "all - pairing as many as possible, the most dense complex MSAs, "
    "cyclic - sequentially pairing the nearest neighbor defined in the stoichoimetry, "
    "custom - use a defined list of pairs.",
)
flags.DEFINE_boolean(
    "no_template",
    False,
    "Do not use structural template. Note that "
    "this does not have an impact on models that do not use template regardless.",
)
flags.DEFINE_integer(
    "max_mono_msa_depth", 10000, "The maximum MSA depth for each monomer", lower_bound=1
)
flags.DEFINE_integer(
    "max_template_hits", 4, "The maximum PDB template for each monomer", lower_bound=0
)


FLAGS = flags.FLAGS
Flag = Type[FLAGS]


##################################################################################################


##################################################################################################
# @profile  # debugging possible memory leak with pickle load
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # read a list of target files
    target_lst = read_af2c_target_file(FLAGS.target_lst_path)

    if FLAGS.feature_dir == FLAGS.output_dir:
        print("Warning: feature input dir is the same as the output dir")

    # Predict structure for each target.
    for target in target_lst:
        target_name = target["name"]
        target_split = target["split"]
        print(f"Info: working on target {target_name}")

        seq_name = target_name
        feature_dir = os.path.join(FLAGS.feature_dir, seq_name)

        if not os.path.exists(feature_dir):
            raise SystemExit("Error: ", feature_dir, "does not exists")

        # load pre-generated features as a pickled dictionary.
        features_input_path = os.path.join(feature_dir, "features.pkl.gz")
        if not os.path.exists(features_input_path):
            features_input_path = os.path.join(feature_dir, "features.pkl")
            if not os.path.exists(features_input_path):
                raise Exception(
                    f"Error: {seq_name} could not locate feature input file under {feature_dir}"
                )

        with hook_compressed(features_input_path, "rb") as f:
            mono_feature_dict = None
            mono_feature_dict = pickle.load(f)
            N = len(mono_feature_dict["msa"])
            L = len(mono_feature_dict["residue_index"])
            T = mono_feature_dict["template_all_atom_positions"].shape[0]
            print(
                f"Info: {target['name']} found monomer {seq_name} msa_depth = {N}, seq_len = {L}, num_templ = {T}"
            )
            if N > FLAGS.max_mono_msa_depth:
                print(
                    f"Info: {seq_name} MSA size is too large, reducing to {FLAGS.max_mono_msa_depth}"
                )
                mono_feature_dict["msa"] = mono_feature_dict["msa"][
                    : FLAGS.max_mono_msa_depth, :
                ]
                mono_feature_dict["deletion_matrix_int"] = mono_feature_dict[
                    "deletion_matrix_int"
                ][: FLAGS.max_mono_msa_depth, :]
                mono_feature_dict["num_alignments"][:] = FLAGS.max_mono_msa_depth
                if "msa_species_identifiers" in mono_feature_dict:
                    mono_feature_dict["msa_species_identifiers"] = mono_feature_dict[
                        "msa_species_identifiers"
                    ][: FLAGS.max_mono_msa_depth]

            if T > FLAGS.max_template_hits:
                print(
                    f"Info: {seq_name} reducing the number of structural templates to {FLAGS.max_template_hits}"
                )
                mono_feature_dict["template_aatype"] = mono_feature_dict[
                    "template_aatype"
                ][: FLAGS.max_template_hits, ...]
                mono_feature_dict["template_all_atom_masks"] = mono_feature_dict[
                    "template_all_atom_masks"
                ][: FLAGS.max_template_hits, ...]
                mono_feature_dict["template_all_atom_positions"] = mono_feature_dict[
                    "template_all_atom_positions"
                ][: FLAGS.max_template_hits, ...]
                mono_feature_dict["template_domain_names"] = mono_feature_dict[
                    "template_domain_names"
                ][: FLAGS.max_template_hits]
                mono_feature_dict["template_sequence"] = mono_feature_dict[
                    "template_sequence"
                ][: FLAGS.max_template_hits]
                mono_feature_dict["template_sum_probs"] = mono_feature_dict[
                    "template_sum_probs"
                ][: FLAGS.max_template_hits, :]

            if (
                T == 0 or FLAGS.no_template
            ):  # deal with senario no template found, or set it to a null template if requested
                mono_template_features = initialize_template_feats(
                    1, L, is_multimer=False
                )
                mono_feature_dict.update(mono_template_features)

        output_dir = os.path.join(FLAGS.output_dir, target_name)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except (
                FileExistsError
            ):  # this could happen when multiple runs are working on the same target simultaneously
                print(f"Warning: tried to create an existing {output_dir}, ignored")

        feature_output_path = os.path.join(output_dir, "features.pkl")
        with open(feature_output_path, "wb") as f:
            print(f"Writting feature file {feature_output_path}")
            pickle.dump(mono_feature_dict, f, protocol=4)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "target_lst_path",
            "output_dir",
            "feature_dir",
        ]
    )

    app.run(main)