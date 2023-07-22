from dataclasses import dataclass
from enum import Enum


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


@dataclass
class BaselineModelConfig:
    nic: int = 6
    nf: int = 64
    momemtum: float = 0.01


@dataclass
class ConditionalModelConfig(BaselineModelConfig):
    nic: int = 28


class EnergyCalculationMethod(Enum):
    rosetta_energy: str = "rosetta_energy"  # type: ignore
    log_p_mean: str = "log_p_mean"  # type: ignore


@dataclass
class InferenceConfig:
    n_design_iters: int = 50
    sampler_config: SamplerConfig = SamplerConfig()
    energy_calculation: EnergyCalculationMethod = EnergyCalculationMethod.rosetta_energy
