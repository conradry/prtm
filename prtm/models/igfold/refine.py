import openmm
import pdbfixer

try:
    import pyrosetta

    PYROSETTA = True
except:
    # PyRosetta functions won't be imported
    PYROSETTA = False
    pass

from prtm.models.igfold.utils.general import exists

ENERGY = openmm.unit.kilocalories_per_mole
LENGTH = openmm.unit.angstroms

__all__ = ["refine_openmm"]
if PYROSETTA:
    __all__.append("refine_pyrosetta")


def refine_openmm(pdb_file, stiffness=10.0, tolerance=2.39, use_gpu=False):
    tolerance = tolerance * ENERGY
    stiffness = stiffness * ENERGY / (LENGTH**2)

    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)
    for residue in modeller.topology.residues():
        for atom in residue.atoms():
            if atom.name in ["N", "CA", "C", "CB"]:
                force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)

    with open(pdb_file, "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True,
        )


def init_pyrosetta(init_string=None, silent=True):
    if not exists(init_string):
        init_string = "-mute all -ignore_zero_occupancy false -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string, silent=silent)


def get_min_mover(
    max_iter: int = 1000,
    sf_name: str = "ref2015_cst",
    coord_cst_weight: float = 1,
    dih_cst_weight: float = 1,
) -> "pyrosetta.rosetta.protocols.moves.Mover":
    """
    Create full-atom minimization mover
    """

    sf = pyrosetta.create_score_function(sf_name)
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded,
        1,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
        0,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint,
        coord_cst_weight,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint,
        dih_cst_weight,
    )

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(False)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap,
        sf,
        "lbfgs_armijo_nonmonotone",
        0.0001,
        True,
    )
    min_mover.max_iter(max_iter)
    min_mover.cartesian(True)

    return min_mover


def get_fa_relax_mover(
    max_iter: int = 100,
) -> "pyrosetta.rosetta.protocols.moves.Mover":
    """
    Create full-atom relax mover
    """

    sf = pyrosetta.create_score_function("ref2015_cst")

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf)
    relax.max_iter(max_iter)
    relax.set_movemap(mmap)

    return relax


def get_repack_mover():
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    return packer


def refine_pyrosetta(
    out_pdb_file, pdb_string, minimization_iter=100, constrain=True, idealize=False
):
    # create new pose
    pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(
        pose,
        pdb_string,
    )

    if constrain:
        cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
        cst_mover.cst_sidechain(False)
        cst_mover.apply(pose)

    min_mover = get_min_mover(
        max_iter=minimization_iter,
        coord_cst_weight=1,
        dih_cst_weight=0,
    )
    min_mover.apply(pose)

    if idealize:
        idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
        idealize_mover.apply(pose)

    packer = get_repack_mover()
    packer.apply(pose)

    min_mover.apply(pose)

    pose.dump_pdb(out_pdb_file)
