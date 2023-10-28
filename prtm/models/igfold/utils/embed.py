from einops import rearrange
from prtm.models.igfold.interface import IgFoldInput
from prtm.models.igfold.utils.folding import get_sequence_dict, process_template


def embed(
    antiberty,
    model,
    fasta_file=None,
    sequences=None,
    template_pdb=None,
    ignore_cdrs=None,
    ignore_chain=None,
    mask=None,
):
    seq_dict = get_sequence_dict(
        sequences,
        fasta_file,
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
        batch_mask=mask,
    )

    model_out = model(model_in)

    prmsd = rearrange(
        model_out.prmsd,
        "b (l a) -> b l a",
        a=4,
    )
    model_out.prmsd = prmsd

    return model_out
