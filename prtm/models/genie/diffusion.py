import math
from abc import ABC, abstractmethod

import torch
from prtm.models.genie import config
from prtm.models.genie.model import Denoiser
from prtm.models.genie.utils.affine_utils import T
from prtm.models.genie.utils.geo_utils import compute_frenet_frames
from tqdm import tqdm


def get_betas(n_timestep, schedule):
    if schedule == "linear":
        return linear_beta_schedule(n_timestep)
    elif schedule == "cosine":
        return cosine_beta_schedule(n_timestep)
    else:
        print("Invalid schedule: {}".format(schedule))
        exit(0)


def linear_beta_schedule(n_timestep, start=0.0001, end=0.02):
    return torch.linspace(start, end, n_timestep)


def cosine_beta_schedule(n_timestep):
    steps = n_timestep + 1
    x = torch.linspace(0, n_timestep, steps)
    alphas_cumprod = torch.cos((x / steps) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(torch.nn.Module, ABC):
    def __init__(self, cfg: config.GenieConfig):
        super(Diffusion, self).__init__()

        self.cfg = cfg

        self.model = Denoiser(self.cfg.model, n_timestep=self.cfg.diffusion.n_timestep)

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
        device = list(self.model.parameters())[0].device
        if not self.setup:
            self.setup_schedule()
            self.setup = True
        ts = self.sample_frames(mask)
        ts_seq = [ts]
        for i in tqdm(
            reversed(range(self.cfg.diffusion.n_timestep)),
            desc="sampling loop time step",
            total=self.cfg.diffusion.n_timestep,
            disable=not verbose,
        ):
            s = torch.Tensor([i] * mask.shape[0]).long().to(device)
            ts = self.p(ts, s, mask)
            ts_seq.append(ts)
        return ts_seq


class Genie(Diffusion):
    def setup_schedule(self):
        device = list(self.model.parameters())[0].device
        self.betas = get_betas(
            self.cfg.diffusion.n_timestep, self.cfg.diffusion.schedule
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.Tensor([1.0]).to(device), self.alphas_cumprod[:-1]]
        )
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = 1.0 / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
            self.betas * self.sqrt_alphas_cumprod_prev / self.one_minus_alphas_cumprod
        )
        self.posterior_mean_coef2 = (
            self.one_minus_alphas_cumprod_prev
            * self.sqrt_alphas
            / self.one_minus_alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * self.one_minus_alphas_cumprod_prev
            / self.one_minus_alphas_cumprod
        )

    def transform(self, batch):
        coords, mask = batch
        coords = coords.float()
        mask = mask.float()

        ca_coords = coords[:, 1::3]
        trans = ca_coords - torch.mean(ca_coords, dim=1, keepdim=True)
        rots = compute_frenet_frames(trans, mask)

        return T(rots, trans), mask

    def sample_timesteps(self, num_samples):
        device = list(self.model.parameters())[0].device
        return torch.randint(0, self.cfg.diffusion.n_timestep, size=(num_samples,)).to(
            device
        )

    def sample_frames(self, mask):
        device = list(self.model.parameters())[0].device
        trans = torch.randn((mask.shape[0], mask.shape[1], 3)).to(device)
        trans = trans * mask.unsqueeze(-1)
        rots = compute_frenet_frames(trans, mask)
        return T(rots, trans)

    def q(self, t0, s, mask):
        # [b, n_res, 3]
        device = list(self.model.parameters())[0].device
        trans_noise = torch.randn_like(t0.trans) * mask.unsqueeze(-1)
        rots_noise = (
            torch.eye(3)
            .view(1, 1, 3, 3)
            .repeat(t0.shape[0], t0.shape[1], 1, 1)
            .to(device)
        )

        trans = (
            self.sqrt_alphas_cumprod[s].view(-1, 1, 1).to(device) * t0.trans
            + self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1).to(device)
            * trans_noise
        )
        rots = compute_frenet_frames(trans, mask)

        return T(rots, trans), T(rots_noise, trans_noise)

    def p(self, ts, s, mask):
        device = list(self.model.parameters())[0].device
        # [b, 1, 1]
        w_noise = (
            (1.0 - self.alphas[s].to(device))
            / self.sqrt_one_minus_alphas_cumprod[s].to(device)
        ).view(-1, 1, 1)

        # [b, n_res]
        noise_pred_trans = ts.trans - self.model(ts, s, mask).trans
        noise_pred_rots = (
            torch.eye(3).view(1, 1, 3, 3).repeat(ts.shape[0], ts.shape[1], 1, 1)
        )
        noise_pred = T(noise_pred_rots, noise_pred_trans)

        # [b, n_res, 3]
        trans_mean = (1.0 / self.sqrt_alphas[s]).view(-1, 1, 1).to(device) * (
            ts.trans - w_noise * noise_pred.trans
        )
        trans_mean = trans_mean * mask.unsqueeze(-1)

        if (s == 0.0).all():
            rots_mean = compute_frenet_frames(trans_mean, mask)
            return T(rots_mean.detach(), trans_mean.detach())
        else:
            # [b, n_res, 3]
            trans_z = torch.randn_like(ts.trans).to(device)

            # [b, 1, 1]
            trans_sigma = self.sqrt_betas[s].view(-1, 1, 1).to(device)

            # [b, n_res, 3]
            trans = trans_mean + trans_sigma * trans_z
            trans = trans * mask.unsqueeze(-1)

            # [b, n_res, 3, 3]
            rots = compute_frenet_frames(trans, mask)

            return T(rots.detach(), trans.detach())
