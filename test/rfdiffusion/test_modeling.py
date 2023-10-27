from pathlib import Path

from prtm import protein
from prtm.models.rfdiffusion import config
from prtm.models.rfdiffusion.modeling import RFDiffusionForStructureDesign

from ..test_utils import _compare_structures


def test_random_length_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_random_length.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.UnconditionalSamplerConfig(
        contigmap_params=config.ContigMap(contigs=["100-200"]),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_monomer_rog_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_monomer_rog_potential.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.UnconditionalSamplerConfig(
        contigmap_params=config.ContigMap(contigs=["100-200"]),
        potentials_params=config.PotentialsParams(
            guiding_potentials=["type:monomer_ROG,weight:1,min_dist:5"],
            guide_scale=2,
            guide_decay="quadratic",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_contact_potential_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_contact_potential.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.UnconditionalSamplerConfig(
        contigmap_params=config.ContigMap(contigs=["100-120"]),
        potentials_params=config.PotentialsParams(
            guiding_potentials=["type:monomer_contacts,weight:0.05"]
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_tetrahedral_oligos_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_tetrahedral_oligos.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.UnconditionalSamplerConfig(
        contigmap_params=config.ContigMap(contigs=["84-84"]),
        symmetry_params=config.SymmetryParams(symmetry="tetrahedral"),
        potentials_params=config.PotentialsParams(
            guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"],
            olig_inter_all=True,
            olig_intra_all=True,
            guide_scale=2,
            guide_decay="quadratic",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_cyclic_oligos_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_cyclic_oligos.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.UnconditionalSamplerConfig(
        contigmap_params=config.ContigMap(contigs=["90-90"]),
        symmetry_params=config.SymmetryParams(symmetry="C6"),
        potentials_params=config.PotentialsParams(
            guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"],
            olig_intra_all=True,
            olig_inter_all=True,
            guide_scale=2.0,
            guide_decay="quadratic",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_dihedral_oligos_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_dihedral_oligos.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.UnconditionalSamplerConfig(
        contigmap_params=config.ContigMap(contigs=["60-60"]),
        symmetry_params=config.SymmetryParams(symmetry="D2"),
        potentials_params=config.PotentialsParams(
            guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.1"],
            olig_intra_all=True,
            olig_inter_all=True,
            guide_scale=2.0,
            guide_decay="quadratic",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_motif_scaffolding_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_motif_scaffolding.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "5tpn.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(contigs=["10-40/A163-181/10-40"]),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_motif_scaffolding_with_target_design():
    gt_pdb_file = (
        Path(__file__).parents[0] / f"reference_motif_scaffolding_w_target.pdb"
    )
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "1ycr.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="complex_base", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(
            contigs=["A25-109/0 0-70/B17-29/0-70"], length="70-120"
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_enzyme_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_enzyme.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "5an7.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="active_site", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(
            contigs=["10-100/A1083-1083/10-100/A1051-1051/10-100/A1180-1180/10-100"]
        ),
        potentials_params=config.PotentialsParams(
            guiding_potentials=[
                "type:substrate_contacts,s:1,r_0:8,rep_r_0:5.0,rep_s:2,rep_r_min:1"
            ],
            guide_scale=1,
            guide_decay="quadratic",
            substrate="LLK",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_nickel_motif_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_nickel_motif.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "nickel_motif.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="base_epoch8", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(
            contigs=["50/A2-4/50/0 50/A7-9/50/0 50/A12-14/50/0 50/A17-19/50/0"]
        ),
        symmetry_params=config.SymmetryParams(symmetry="C4"),
        potentials_params=config.PotentialsParams(
            guiding_potentials=["type:olig_contacts,weight_intra:1,weight_inter:0.06"],
            olig_inter_all=True,
            olig_intra_all=True,
            guide_scale=2,
            guide_decay="quadratic",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_insulin_ppi_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_insulin_ppi.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "insulin_target.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(contigs=["A1-150/0 70-100"]),
        ppi_params=config.PPIParams(hotspot_res=["A59", "A83", "A91"]),
        denoiser_params=config.DenoiserParams(noise_scale_ca=0, noise_scale_frame=0),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_insulin_ppi_beta_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_insulin_ppi_beta.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "insulin_target.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="complex_beta", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(contigs=["A1-150/0 70-100"]),
        ppi_params=config.PPIParams(hotspot_res=["A59", "A83", "A91"]),
        denoiser_params=config.DenoiserParams(noise_scale_ca=0, noise_scale_frame=0),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_sequence_inpainting():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_sequence_inpainting.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "5tpn.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(
            contigs=["10-40/A163-181/10-40"], inpaint_seq=["A163-168/A170-171/A179"]
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_partial_diffusion_without_sequence_design():
    gt_pdb_file = (
        Path(__file__).parents[0] / f"reference_partial_diffusion_wo_sequence.pdb"
    )
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "2kl8.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    diffuser_config_override = config.DiffuserConfig(partial_T=10)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(contigs=["79-79"]),
    )
    designed_structure = designer(
        sampler_config, diffuser_config_override=diffuser_config_override
    )[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_partial_diffusion_with_sequence_design():
    gt_pdb_file = (
        Path(__file__).parents[0] / f"reference_partial_diffusion_w_sequence.pdb"
    )
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "peptide_complex.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    diffuser_config_override = config.DiffuserConfig(partial_T=10)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(
            contigs=["172-172/0 34-34"], provide_seq=["172-205"]
        ),
    )
    designed_structure = designer(
        sampler_config, diffuser_config_override=diffuser_config_override
    )[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_partial_diffusion_with_multisequence_design():
    gt_pdb_file = (
        Path(__file__).parents[0] / f"reference_partial_diffusion_w_multisequence.pdb"
    )
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "peptide_complex.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    diffuser_config_override = config.DiffuserConfig(partial_T=10)
    sampler_config = config.SelfConditioningSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        contigmap_params=config.ContigMap(
            contigs=["172-172/0 34-34"], provide_seq=["172-177,200-205"]
        ),
    )
    designed_structure = designer(
        sampler_config, diffuser_config_override=diffuser_config_override
    )[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_scaffoldguided_tim_barrel_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_scaffoldguided_tim_barrel.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "1qys.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    with open(Path(__file__).parents[0] / "tim10.pdb", mode="r") as f:
        scaffold_pdb_str = f.read()

    scaffold_structure = protein.Protein14.from_pdb_string(
        scaffold_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.ScaffoldedSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        denoiser_params=config.DenoiserParams(
            noise_scale_ca=0.5, noise_scale_frame=0.5
        ),
        scaffoldguided_params=config.ScaffoldGuidedParams(
            target_structure=None,
            target_adj=False,
            target_ss=False,
            scaffold_structure_list=[scaffold_structure],
            sampled_insertion="0-5",
            sampled_N="0-5",
            sampled_C="0-5",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)


def test_scaffoldguided_ppi_design():
    gt_pdb_file = Path(__file__).parents[0] / f"reference_scaffoldguided_ppi.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.Protein14.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "1qys.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.Protein14.from_pdb_string(
        reference_pdb_str, parse_hetatom=True
    )

    with open(Path(__file__).parents[0] / "insulin_target.pdb", mode="r") as f:
        target_pdb_str = f.read()

    target_structure = protein.Protein14.from_pdb_string(
        target_pdb_str, parse_hetatom=True
    )

    with open(Path(__file__).parents[0] / "5L33.pdb", mode="r") as f:
        scaffold_pdb_str = f.read()

    scaffold_structure = protein.Protein14.from_pdb_string(
        scaffold_pdb_str, parse_hetatom=True
    )

    designer = RFDiffusionForStructureDesign(model_name="auto", random_seed=0)
    sampler_config = config.ScaffoldedSamplerConfig(
        inference_params=config.InferenceParams(
            reference_structure=reference_structure
        ),
        denoiser_params=config.DenoiserParams(noise_scale_ca=0, noise_scale_frame=0),
        ppi_params=config.PPIParams(hotspot_res=["A59", "A83", "A91"]),
        scaffoldguided_params=config.ScaffoldGuidedParams(
            target_structure=target_structure,
            target_adj=True,
            target_ss=True,
            scaffold_structure_list=[scaffold_structure],
            sampled_insertion="0-5",
            sampled_N="0-5",
            sampled_C="0-5",
        ),
    )
    designed_structure = designer(sampler_config)[0]
    designed_pdb = designed_structure.to_pdb()
    pred_structure = protein.Protein14.from_pdb_string(designed_pdb)

    _compare_structures(pred_structure, gt_structure, atol=0.1)
