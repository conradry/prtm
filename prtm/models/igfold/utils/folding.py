import numpy as np
import torch
from einops import rearrange
from prtm.models.igfold.interface import IgFoldInput
from prtm.models.igfold.utils.fasta import get_fasta_chain_dict
from prtm.models.igfold.utils.general import exists
from prtm.models.igfold.utils.pdb import (
    cdr_indices,
    get_atom_coords,
    save_PDB,
    write_pdb_bfactor,
)


def get_sequence_dict(
    sequences,
    fasta_file,
):
    if exists(sequences) and exists(fasta_file):
        print("Both sequences and fasta file provided. Using fasta file.")
        seq_dict = get_fasta_chain_dict(fasta_file)
    elif not exists(sequences) and exists(fasta_file):
        seq_dict = get_fasta_chain_dict(fasta_file)
    elif exists(sequences):
        seq_dict = sequences
    else:
        exit("Must provide sequences or fasta file.")

    return seq_dict


def process_template(
    pdb_file,
    fasta_file,
    ignore_cdrs=None,
    ignore_chain=None,
):
    temp_coords, temp_mask = None, None
    if exists(pdb_file):
        temp_coords = get_atom_coords(
            pdb_file,
            fasta_file=fasta_file,
        )
        temp_coords = (
            torch.stack(
                [
                    temp_coords["N"],
                    temp_coords["CA"],
                    temp_coords["C"],
                    temp_coords["CB"],
                ],
                dim=1,
            )
            .view(-1, 3)
            .unsqueeze(0)
        )

        temp_mask = torch.ones(temp_coords.shape[:2]).bool()
        temp_mask[temp_coords.isnan().any(-1)] = False
        temp_mask[temp_coords.sum(-1) == 0] = False

        if exists(ignore_cdrs):
            cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
            if ignore_cdrs == False:
                cdr_names = []
            elif isinstance(ignore_cdrs, list):
                cdr_names = ignore_cdrs
            elif isinstance(ignore_cdrs, str):
                cdr_names = [ignore_cdrs]

            for cdr in cdr_names:
                cdr_range = cdr_indices(pdb_file, cdr)
                temp_mask[:, (cdr_range[0] - 1) * 4 : (cdr_range[1] + 2) * 4] = False
        if exists(ignore_chain) and ignore_chain in ["H", "L"]:
            seq_dict = get_fasta_chain_dict(fasta_file)
            hlen = len(seq_dict["H"])
            if ignore_chain == "H":
                temp_mask[:, : hlen * 4] = False
            elif ignore_chain == "L":
                temp_mask[:, hlen * 4 :] = False

    return temp_coords, temp_mask


def process_prediction(
    model_out,
    pdb_file,
    fasta_file,
    skip_pdb=False,
    do_refine=True,
    use_openmm=False,
    do_renum=False,
):
    prmsd = rearrange(
        model_out.prmsd,
        "b (l a) -> b l a",
        a=4,
    )
    model_out.prmsd = prmsd

    if skip_pdb:
        return model_out

    coords = model_out.coords.squeeze(0).detach()
    res_rmsd = prmsd.square().mean(dim=-1).sqrt().squeeze(0)

    seq_dict = get_fasta_chain_dict(fasta_file)
    full_seq = "".join(list(seq_dict.values()))
    chains = list(seq_dict.keys())
    delims = np.cumsum([len(s) for s in seq_dict.values()]).tolist()

    write_pdb = not do_refine or use_openmm
    pdb_string = save_PDB(
        pdb_file,
        coords,
        full_seq,
        chains=chains,
        atoms=["N", "CA", "C", "CB", "O"],
        error=res_rmsd,
        delim=delims,
        write_pdb=write_pdb,
    )

    if do_refine:
        if use_openmm:
            try:
                from prtm.models.igfold.refine import refine_openmm as refiner

                refine_input = [pdb_file]
            except:
                exit("OpenMM not installed. Please install OpenMM to use refinement.")
        else:
            try:
                from prtm.models.igfold.refine import refine_pyrosetta as refiner

                refine_input = [pdb_file, pdb_string]
            except:
                exit(
                    "PyRosetta not installed. Please install PyRosetta to use refinement."
                )

        refiner(*refine_input)

    if do_renum:
        try:
            from prtm.models.igfold.utils.abnumber_ import renumber_pdb
        except:
            exit("AbNumber not installed. Please install AbNumber to use renumbering.")

        renumber_pdb(
            pdb_file,
            pdb_file,
        )

    write_pdb_bfactor(
        pdb_file,
        pdb_file,
        bfactor=res_rmsd,
    )

    return model_out


def fold(
    antiberty,
    models,
    pdb_file,
    fasta_file=None,
    sequences=None,
    template_pdb=None,
    ignore_cdrs=None,
    ignore_chain=None,
    skip_pdb=False,
    do_refine=True,
    use_openmm=False,
    do_renum=True,
    truncate_sequences=False,
):
    seq_dict = get_sequence_dict(
        sequences,
        fasta_file,
    )

    if truncate_sequences:
        try:
            from prtm.models.igfold.utils.abnumber_ import truncate_seq
        except:
            exit("AbNumber not installed. Please install AbNumber to use truncation.")

        seq_dict = {k: truncate_seq(v) for k, v in seq_dict.items()}

    if not exists(fasta_file):
        fasta_file = pdb_file.replace(".pdb", ".fasta")
        with open(fasta_file, "w") as f:
            for chain, seq in seq_dict.items():
                f.write(
                    ">{}\n{}\n".format(
                        chain,
                        seq,
                    )
                )

    embeddings, attentions = antiberty.embed(
        seq_dict.values(),
        return_attention=True,
    )
    embeddings = [e[1:-1].unsqueeze(0) for e in embeddings]
    attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in attentions]

    temp_coords, temp_mask = process_template(
        template_pdb,
        fasta_file,
        ignore_cdrs=ignore_cdrs,
        ignore_chain=ignore_chain,
    )
    model_in = IgFoldInput(
        embeddings=embeddings,
        attentions=attentions,
        template_coords=temp_coords,
        template_mask=temp_mask,
        return_embeddings=True,
    )

    model_outs, scores = [], []
    with torch.no_grad():
        for i, model in enumerate(models):
            model_out = model(model_in)
            model_out = model.gradient_refine(model_in, model_out)
            scores.append(model_out.prmsd.quantile(0.9))
            model_outs.append(model_out)

    best_model_i = scores.index(min(scores))
    model_out = model_outs[best_model_i]
    process_prediction(
        model_out,
        pdb_file,
        fasta_file,
        skip_pdb=skip_pdb,
        do_refine=do_refine,
        use_openmm=use_openmm,
        do_renum=do_renum,
    )

    return model_out
