from pathlib import Path

import pytest

from proteome import protein
from proteome.models.protein_generator import config
from proteome.models.protein_generator.modeling import \
    ProteinGeneratorForJointDesign

from ..test_utils import _compare_structures


def test_unconditional_design():
    expected_sequence = "GPPPLSPEEIEELRELLEELAERFGISPEELARFFEPFIRIFLEKDPEELIEELRRFLESGFTREEFVEVSIPEIERYVEKGLLSDEEVEELLEFLERLG"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_unconditional.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            contigmap_params=config.ContigMap(contigs=["100"]),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_weighted_sequence_design():
    expected_sequence = "GLPEITPEEIEELKKLWEEWKEALEPFLEWLERRGIPIGNPEFIKEFEEFIEELRKEIKNGATREEIIEFFIEEIEELVEKGLITEEEVEEFLKWIERWG"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_weighted_sequence.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            contigmap_params=config.ContigMap(contigs=["100"]),
            potentials_params=config.PotentialsParams(
                potentials=[config.AACompositionalBiasParams(aa_composition="W0.2")],
                potential_scales=[1.75],
            ),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_binder_design():
    expected_sequence = "SLLELTLSVKGETYKLRMQKSANGTVSVTILNSRGEPLAQPVSVAVTFIMLKIQAYFNETADLPCQFANSQNQSLSELVVFWQDQENLVLNEVYLGKEKFDSVHSKYMGRTSFDSDSWTLRLHNLQIKDKGLYQCIIHHKKPTGMIRIHQMNSELSVLA"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_binder.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "cd86.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.from_pdb_string(reference_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            reference_structure=reference_structure,
            contigmap_params=config.ContigMap(contigs=["B1-110/0 25-75"]),
            hotspot_params=config.HotspotParams(
                hotspot_res=["B40", "B32", "B87", "B96", "B30"]
            ),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_motif_scaffolding_design():
    expected_sequence = "SRPERVIRITPEEVNKIKSALLSTNKAVVSLDGKTIEIDRTDVIKDGEIIIDPNRTIKK"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_motif_scaffolding.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "rsv5_5tpn.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.from_pdb_string(reference_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            reference_structure=reference_structure,
            contigmap_params=config.ContigMap(contigs=["0-25/A163-181/25-30"]),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_partial_diffusion_design():
    expected_sequence = "GLSPEELREFLRREGIELRSEEELRELLERLEELRELR"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_partial_diffusion.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "design_000000.pdb", mode="r") as f:
        reference_pdb_str = f.read()

    reference_structure = protein.from_pdb_string(reference_pdb_str)
    reference_structure = protein.crop_protein_37_to_27(reference_structure)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            reference_structure=reference_structure,
            diffuser_params=config.DiffuserParams(T=50),
            contigmap_params=config.ContigMap(contigs=["38"]),
            sampling_temp=0.3,
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_secondary_structure_design():
    expected_sequence = "SLEELVRIAKRYGIPLEELISAAREIIALIRAGRKLSAAEIEAIAARFAKKFGLSPEEAREFLLELIEEVAAGGVPSAAEMVALLKALRELVEDLVAIRK"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_secondary_structure.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    secondary_structure_str = "XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX"

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            contigmap_params=config.ContigMap(contigs=["100"]),
            secondary_structure_params=config.SecondaryStructureParams(
                secondary_structure=secondary_structure_str
            ),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_secondary_structure_bias_design():
    expected_sequence = "SEEELERKKKALEQAKEELEKANRAINEARRALRELDAAQKELIALLEILKDENLSEKEREKRLEEVEEKIKEEQAKLQAEREKINAIREEVEQLLKKAK"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_secondary_structure_bias.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            contigmap_params=config.ContigMap(contigs=["100"]),
            structure_bias_params=config.StructureBiasParams(
                helix_bias=0.01, strand_bias=0.01
            ),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_secondary_structure_from_pdb_design():
    expected_sequence = "QPLELTLSGNTLTVKLPEGWSAPEVSGPTVLAYRTLPGAQPLAVAPTFVLSDGGGTVSVSPARLEPPAFVFKAELPQNAKEVEVTLTIYQQVNGKWTLLKQVVFTLTRAP"
    gt_pdb_file = (
        Path(__file__).parents[0] / f"reference_secondary_structure_from_pdb.pdb"
    )
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    with open(Path(__file__).parents[0] / "cd86.pdb", mode="r") as f:
        dssp_pdb_str = f.read()

    dssp_structure = protein.from_pdb_string(dssp_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            contigmap_params=config.ContigMap(contigs=["110"]),
            secondary_structure_params=config.SecondaryStructureParams(
                dssp_structure=dssp_structure
            ),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_sequence_conditioning_design():
    expected_sequence = "SLEELLARIEELLEELPEPSEQAKAQLEELLARIKELK"
    gt_pdb_file = Path(__file__).parents[0] / f"reference_sequence_conditioning.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            sequence="XXXXXXXXXXXXXXXXPEPSEQXXXXXXXXXXXXXXXX",
            contigmap_params=config.ContigMap(),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_sequence_partial_diffusion_design():
    expected_sequence = "GIPPLIIIRIFRIPGITLDEIINFLKNLGFENIEIERLGENYFVIRFRINGREIIIVFDKNGKILDIIFSEEDLKEILEFLKKLGINPEELEKELEKIFPN"
    gt_pdb_file = (
        Path(__file__).parents[0] / f"reference_sequence_partial_diffusion.pdb"
    )
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            sequence="SAKVEELLETAKALGISEEEVREILELLEAGFIVIEVVSLGDAVILILENKKLGKYYILKNGEIERIKKPENARELKRKIAEILNISVEEIEAIIEKLRAK",
            diffuser_params=config.DiffuserParams(T=50),
            contigmap_params=config.ContigMap(),
            sampling_temp=0.3,
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)


def test_symmetric_design():
    expected_sequence = (
        "NKIIIELTPEEIEEFLKFIKRIIEENKIIIILTPEEIEEFLKFIKRIIEENKIIIELTPEEIEEFLKFIKRIIEE"
    )
    gt_pdb_file = Path(__file__).parents[0] / f"reference_symmetric_design.pdb"
    with open(gt_pdb_file, "r") as f:
        gt_pdb_str = f.read()

    gt_structure = protein.from_pdb_string(gt_pdb_str)

    designer = ProteinGeneratorForJointDesign(model_name="auto", random_seed=0)
    designed_structure, designed_sequence = designer.design_structure_and_sequence(
        config.InferenceConfig(
            diffuser_params=config.DiffuserParams(T=50),
            contigmap_params=config.ContigMap(contigs=["25/0 25/0 25/0"]),
            symmetry_params=config.SymmetryParams(symmetry=3),
        ),
    )
    designed_pdb = protein.to_pdb(designed_structure)
    pred_structure = protein.from_pdb_string(designed_pdb)

    assert designed_sequence == expected_sequence
    _compare_structures(pred_structure, gt_structure, atol=0.01)
