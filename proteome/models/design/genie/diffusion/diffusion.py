from abc import ABC, abstractmethod

import torch
from proteome.models.design.genie.model.model import Denoiser
from pytorch_lightning.core import LightningModule
from torch.optim import Adam
from tqdm import tqdm

from proteome.models.design.genie import config


class Diffusion(LightningModule, ABC):
    def __init__(self, cfg: config.GenieConfig):
        super(Diffusion, self).__init__()

        self.cfg = cfg

        self.model = Denoiser(
            self.config.model, n_timestep=self.config.diffusion.n_timestep
        )

        self.setup = False

    @abstractmethod
    def setup_schedule(self):
        """
        Set up variance schedule and precompute its corresponding terms.
        """
        raise NotImplemented

    @abstractmethod
    def transform(self, batch):
        """
        Transform batch data from data pipeline into the desired format

        Input:
                batch - coordinates from data pipeline (shape: b x (n_res * 3))

        Output: frames (shape: b x n_res)
        """
        raise NotImplemented

    @abstractmethod
    def sample_timesteps(self, num_samples):
        raise NotImplemented

    @abstractmethod
    def sample_frames(self, mask):
        raise NotImplemented

    @abstractmethod
    def q(self, t0, s, mask):
        raise NotImplemented

    @abstractmethod
    def p(self, ts, s, mask):
        raise NotImplemented

    def p_sample_loop(self, mask, verbose=True):
        if not self.setup:
            self.setup_schedule()
            self.setup = True
        ts = self.sample_frames(mask)
        ts_seq = [ts]
        for i in tqdm(
            reversed(range(self.config.diffusion["n_timestep"])),
            desc="sampling loop time step",
            total=self.config.diffusion["n_timestep"],
            disable=not verbose,
        ):
            s = torch.Tensor([i] * mask.shape[0]).long().to(self.device)
            ts = self.p(ts, s, mask)
            ts_seq.append(ts)
        return ts_seq
