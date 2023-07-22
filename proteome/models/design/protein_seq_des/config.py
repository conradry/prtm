from dataclasses import dataclass


@dataclass
class SamplerConfig:
    no_init_model: bool = False
    randomize: bool = True
    repack_only: bool = False
    use_rosetta_packer: bool = False
    symmetry: bool = False
    ala: bool = False
    val: bool = False
    restrict_gly: bool = True
    no_cys: bool = False
    no_met: bool = False
    threshold: float = 20
    k: int = 4
    pack_radius: float = 5.0
    var_idx: str = ""
    fixed_idx: str = ""
    resfile: str = ""
    anneal: bool = False
    step_rate: float = 0.995
    anneal_start_temp: float = 1.0
    anneal_final_temp: float = 0.0
    is_tim: bool = False
