import random
from enum import Enum
from typing import Optional
from dataclasses import asdict

import numpy as np
import torch
from proteome import protein
from proteome.constants import residue_constants
from proteome.models.design.rfdiffusion import config
from proteome.models.design.rfdiffusion.rosettafold_model import RoseTTAFoldModule

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


class _RFDiffusionForStructureDesign:
    def __init__(
        self,
        model_name: str,
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = RoseTTAFoldModule(
            d_t1d=self.cfg.preprocess.d_t1d,
            d_t2d=self.cfg.preprocess.d_t2d,
            T=self.cfg.diffuser.T,
            **asdict(self.cfg.model),
        )

        self.load_weights(RFD_MODEL_URLS[model_name])
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url,
            file_name=f"{self.model_name}.pt",
            progress=True,
            map_location="cpu",
        )["model_state_dict"]
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def design_structure(self, **kwargs) -> protein.Protein:
        """Design a random protein structure."""
        raise NotImplementedError


class RFDiffusionForUnconditionalStructureDesign(_RFDiffusionForStructureDesign):
    def __init__(
        self,
        model_name: str = "base",
        random_seed: Optional[int] = None,
    ):
        super().__init__(model_name, random_seed)

    @torch.no_grad()
    def design_structure(
        self,
        contig_map: config.ContigMap,
        denoiser_params: config.DenoiserParams = config.DenoiserParams(),
        symmetry_params: Optional[config.SymmetryParams] = None,
        potentials_params: Optional[config.PotentialsParams] = None,
    ) -> protein.Protein:
        """Design a random protein structure."""
        assert contig_map.inpaint_seq is None and contig_map.provide_seq is None
        return


class RFDiffusionForStructureInpainting(_RFDiffusionForStructureDesign):
    def __init__(
        self,
        model_name: str = "inpaint_seq",
        random_seed: Optional[int] = None,
    ):
        super().__init__(model_name, random_seed)

    @torch.no_grad()
    def design_structure(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
        **kwargs,
    ) -> protein.Protein:
        """Design a random protein structure."""
        return


class RFDiffusionForScaffoldGuidedStructureInpainting(_RFDiffusionForStructureDesign):
    def __init__(
        self,
        model_name: str = "inpaint_seq_fold",
        random_seed: Optional[int] = None,
    ):
        super().__init__(model_name, random_seed)

    @torch.no_grad()
    def design_structure(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
        **kwargs,
    ) -> protein.Protein:
        """Design a random protein structure."""
        return


class RFDiffusionForBinderDesign(_RFDiffusionForStructureDesign):
    def __init__(
        self,
        model_name: str = "complex_base",
        random_seed: Optional[int] = None,
    ):
        super().__init__(model_name, random_seed)

    @torch.no_grad()
    def design_structure(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
        **kwargs,
    ) -> protein.Protein:
        """Design a random protein structure."""
        return


class RFDiffusionForScaffoldGuidedBinderDesign(_RFDiffusionForStructureDesign):
    def __init__(
        self,
        model_name: str = "complex_fold_base",  # active_site
        random_seed: Optional[int] = None,
    ):
        super().__init__(model_name, random_seed)

    @torch.no_grad()
    def design_structure(
        self,
        inference_config: config.InferenceConfig = config.InferenceConfig(),
        **kwargs,
    ) -> protein.Protein:
        """Design a random protein structure."""
        return
