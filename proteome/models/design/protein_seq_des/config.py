from dataclasses import dataclass


@dataclass
class SamplerConfig:
    pdb: str
    no_init_model: int = 0
    randomize: int = 1
    repack_only: int = 0
    use_rosetta_packer: int = 0
    threshold: float = 20
    symmetry: int = 0
    k: int = 4
    ala: int = 0
    val: int = 0
    restrict_gly: int = 1
    no_cys: int = 0
    no_met: int = 0
    pack_radius: float = 5.0
    var_idx: str = ""
    fixed_idx: str = ""
    resfile: str = ""
    anneal: int = 1
    step_rate: float = 0.995
    anneal_start_temp: float = 1.0
    anneal_final_temp: float = 0.0
