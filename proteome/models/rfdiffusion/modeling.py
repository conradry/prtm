import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from proteome import protein
from proteome.models.rfdiffusion import config
from proteome.models.rfdiffusion.rosettafold_model import RoseTTAFoldModule
from proteome.models.rfdiffusion.samplers import (
    ScaffoldedSampler,
    SelfConditioningSampler,
    UnconditionalSampler,
)

RFD_MODEL_URLS = {
    "base": "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "complex_base": "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
    "complex_fold_base": "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
    "inpaint_seq": "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
    "inpaint_seq_fold": "http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt",
    "active_site": "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
    "base_epoch8": "http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt",
    "complex_beta": "http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt",
}
RFD_MODEL_CONFIGS = {
    "base": config.BaseConfig,
    "complex_base": config.ComplexBaseConfig,
    "complex_fold_base": config.ComplexFoldBaseConfig,
    "inpaint_seq": config.InpaintSeqConfig,
    "inpaint_seq_fold": config.InpaintSeqFoldConfig,
    "active_site": config.ActiveSiteConfig,
    "base_epoch8": config.Base8Config,
    "complex_beta": config.ComplexBetaConfig,
}


def _get_model_config(model_name: str) -> config.RFDiffusionModelConfig:
    """Get the model config for a given model name."""
    return RFD_MODEL_CONFIGS[model_name]


def _select_model_from_config(sampler_config: config.SamplerConfigType) -> str:
    if (
        sampler_config.contigmap_params.inpaint_seq is not None
        or sampler_config.contigmap_params.provide_seq is not None
    ):
        if hasattr(sampler_config, "scaffoldguided_params"):
            return "inpaint_seq_fold"
        else:
            return "inpaint_seq"
    elif getattr(
        getattr(sampler_config, "ppi_conf", None), "hotspot_res", None
    ) is not None and not hasattr(sampler_config, "scaffoldguided_params"):
        return "complex_base"
    elif hasattr(sampler_config, "scaffoldguided_params"):
        return "complex_fold_base"
    else:
        return "base"


def _validate_model_for_config(
    sampler_config: config.SamplerConfigType, model_name: str
):
    if (
        sampler_config.contigmap_params.inpaint_seq is not None
        or sampler_config.contigmap_params.provide_seq is not None
    ):
        if hasattr(sampler_config, "scaffoldguided_params"):
            assert model_name in ["inpaint_seq_fold"]
        else:
            assert model_name in ["inpaint_seq"]
    elif getattr(
        getattr(sampler_config, "ppi_conf", None), "hotspot_res", None
    ) is not None and not hasattr(sampler_config, "scaffoldguided_params"):
        assert model_name in ["complex_base", "complex_beta"]
    elif hasattr(sampler_config, "scaffoldguided_params"):
        assert model_name in ["complex_fold_base"]
    else:
        assert model_name in [
            "base",
            "base_epoch8",
            "active_site",
            "complex_base",
            "complex_beta",
        ]


def _get_sampler_class(sampler_config: config.SamplerConfigType):
    if isinstance(sampler_config, config.UnconditionalSamplerConfig):
        return UnconditionalSampler
    elif isinstance(sampler_config, config.SelfConditioningSamplerConfig):
        return SelfConditioningSampler
    elif isinstance(sampler_config, config.ScaffoldedSamplerConfig):
        return ScaffoldedSampler
    else:
        raise NotImplementedError


class RFDiffusionForStructureDesign:
    def __init__(
        self,
        model_name: str = "auto",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.loaded_model_name = None
        if model_name != "auto":
            self._set_model(model_name)

    def load_weights(self, model_name: str, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"rfdiffusion_{model_name}.pt",
            progress=True,
            map_location="cpu",
        )["model_state_dict"]
        self.model.load_state_dict(state_dict)

    def _set_model(self, model_name: str):
        if self.loaded_model_name == model_name:
            # Use the model that's already loaded
            return

        self.cfg = _get_model_config(model_name)
        self.model = RoseTTAFoldModule(
            d_t1d=self.cfg.preprocess.d_t1d,
            d_t2d=self.cfg.preprocess.d_t2d,
            T=self.cfg.diffuser.T,
            **asdict(self.cfg.model),
        )

        self.load_weights(model_name, RFD_MODEL_URLS[model_name])
        self.model.eval()
        self.model = self.model.to(self.device)
        self.loaded_model_name = model_name

    def design_structure(
        self,
        sampler_config: config.SamplerConfigType,
        diffuser_config_override: Optional[config.DiffuserConfig] = None,
        preprocess_config_override: Optional[config.PreprocessConfig] = None,
    ) -> protein.Protein:
        """Design a protein structure."""
        if self.model_name == "auto":
            self._set_model(_select_model_from_config(sampler_config))
        else:
            _validate_model_for_config(sampler_config, self.loaded_model_name)

        if diffuser_config_override is not None:
            diffuser_config = diffuser_config_override
        else:
            diffuser_config = self.cfg.diffuser

        if preprocess_config_override is not None:
            preprocess_config = preprocess_config_override
        else:
            preprocess_config = self.cfg.preprocess

        # Setup the sampler class
        sampler = _get_sampler_class(sampler_config)(
            self.model, diffuser_config, preprocess_config, sampler_config
        )
        x_init, seq_init = sampler.sample_init()

        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        # Loop over number of reverse diffusion time steps.
        for t in tqdm(
            range(
                int(sampler.t_step_input), sampler.inference_params.final_step - 1, -1
            )
        ):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t,
                x_t=x_t,
                seq_init=seq_t,
                final_step=sampler.inference_params.final_step,
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])

        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        plddt_stack = torch.stack(plddt_stack)

        # Get the output structure
        final_seq = seq_stack[-1]

        # Output glycines, except for motif region
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # 7 is glycine

        bfacts = torch.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0

        return protein.Protein(
            atom_positions=denoised_xyz_stack[-1].numpy(),
            aatype=final_seq.numpy(),
            atom_mask=np.ones(denoised_xyz_stack[-1].shape[:2], dtype=np.bool_),
            residue_index=np.arange(1, len(final_seq) + 1, dtype=np.int32),
            b_factors=bfacts[:, None].numpy().repeat(14, axis=-1),
            chain_index=np.array(
                [protein.PDB_CHAIN_IDS.index(char) for char in sampler.chain_idx]
            ),
        )
