import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
from proteome import protein
from proteome.models.design.protein_generator import config
from proteome.models.design.protein_generator.rosettafold_model import RoseTTAFoldModule
from proteome.models.design.protein_generator.sampler import SeqDiffSampler
from tqdm import tqdm

PROTGEN_MODEL_URLS = {
    "default": "http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_230205_dssp_hotspots_25mask_EQtasks_mod30.pt",
    "t1d_29": "http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt",
}
PROTGEN_MODEL_CONFIGS = {
    "default": config.BaseConfig,
    "t1d_29": config.ComplexConfig,
}


def _get_model_config(model_name: str) -> config.RFDiffusionModelConfig:
    """Get the model config for a given model name."""
    return PROTGEN_MODEL_CONFIGS[model_name]


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
        assert model_name in ["base", "base_epoch8", "active_site"]


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
            self.set_model(model_name)

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"{self.loaded_model_name}.pt",
            progress=True,
            map_location="cpu",
        )["model_state_dict"]
        self.model.load_state_dict(state_dict)

    def set_model(self, model_name: str):
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

        self.loaded_model_name = model_name
        self.load_weights(RFD_MODEL_URLS[model_name])
        self.model.eval()
        self.model = self.model.to(self.device)

    def design_structure(
        self,
        sampler_config: config.SamplerConfigType,
        diffuser_config_override: Optional[config.DiffuserConfig] = None,
        preprocess_config_override: Optional[config.PreprocessConfig] = None,
    ) -> protein.Protein:
        """Design a protein structure."""
        if self.model_name == "auto":
            self.set_model(_select_model_from_config(sampler_config))
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


"""
if self.cfg.checkpoint == None:
        self.cfg.checkpoint = DEFAULT_CKPT

    self.MODEL_PARAM["d_t1d"] = self.cfg.d_t1d

    # decide based on input args what checkpoint to load
    if (
        self.cfg.hotspots != None
        or self.cfg.secondary_structure != None
        or (
        self.cfg.helix_bias
        + self.cfg.strand_bias
        + self.cfg.loop_bias
        )
        > 0
        or self.cfg.dssp_pdb != None
        and self.cfg.checkpoint == DEFAULT_CKPT
    ):
        self.MODEL_PARAM["d_t1d"] = 29
        print(
        "You are using features only compatible with a newer model, switching checkpoint..."
        )
        self.cfg.checkpoint = t1d_29_CKPT

    elifself.cfg.loop_design and self.cfg.checkpoint == DEFAULT_CKPT:
        print("Switched to loop design checkpoint")
        self.cfg.checkpoint = LOOP_CHECKPOINT

    # check to make sure checkpoint chosen exists
    if not os.path.exists(self.cfg.checkpoint):
        print("WARNING: couldn't find checkpoint")

    if not os.path.exists(self.cfg.checkpoint):
        raise Exception(
        f"MODEL NOT FOUND!\nTo down load models please run the following in the main directory:\nwget http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_230205_dssp_hotspots_25mask_EQtasks_mod30.pt\nwget http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt"
        )

    self.ckpt = torch.load(self.cfg.checkpoint, map_location=self.DEVICE)

    # check to see if [loader_param, model_param, loss_param] is in checkpoint
    #   if so then you are using v2 of inference with t2d bug fixed
    self.v2_mode = False
    if "model_param" in self.ckpt.keys():
        print("You are running a new v2 model switching into v2 inference mode")
        self.v2_mode = True

        for k in self.MODEL_PARAM.keys():
        if k in self.ckpt["model_param"].keys():
            self.MODEL_PARAM[k] = self.ckpt["model_param"][k]
        else:
            print(f"no match for {k} in loaded model params")

    # make model and load checkpoint
    print("Loading model checkpoint...")
    self.model = RoseTTAFoldModule(**self.MODEL_PARAM).to(self.DEVICE)

    model_state = self.ckpt["model_state_dict"]
    self.model.load_state_dict(model_state, strict=False)
    self.model.eval()
    print("Successfully loaded model checkpoint")
"""
