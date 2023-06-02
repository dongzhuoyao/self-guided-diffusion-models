import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb

from diffusion_utils.util import exists, default

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange



# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions



# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))



# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):#around Figure 8 in VDM paper
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):##around Figure 8 in VDM paper
    return -log((torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** -2) - 1, eps = 1e-5)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MonotonicLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())


class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device

        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        x = self.net(x)

        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)


class Schedule_VDM():
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = 'cuda'#TODO. hard code temporarilly

        if self.hparams.beta_schedule == 'linear':
                self.log_snr = beta_linear_log_snr
        elif self.hparams.beta_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif self.hparams.beta_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max=log_snr_max,
                log_snr_min=log_snr_min,
                hidden_dim=self.learned_schedule_net_hidden_dim,
                frac_gradient=self.learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {self.hparams.beta_schedule}')


    def get_alpha(self, time):
        return torch.sqrt((self.log_snr(time)).sigmoid())

    def p_mean_variance(self, x, time, time_next, batch_data, condition_kwargs,denoise_fn, **kwargs):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred_noise = denoise_fn(x, batch_log_snr,  cond_scale=condition_kwargs['cond_scale'], **batch_data)

        if self.hparams.model.clip_denoised:
            x_start = (x - sigma * pred_noise) / alpha

            # in Imagen, this was changed to dynamic thresholding
            x_start.clamp_(-1., 1.)

            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)#Equation 33

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

        # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next, **kwargs):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next, **kwargs)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape,log_num_per_prog, **kwargs):

        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1., 0., self.hparams.timesteps_sampling + 1, device=self.device)

        log_num_per_prog = torch.linspace(0, self.hparams.timesteps_sampling, log_num_per_prog, dtype=torch.int).cpu().numpy().tolist()
        intermediates = []
        for i in tqdm(range(self.hparams.timesteps_sampling), desc='sampling loop time step', total=self.hparams.timesteps_sampling, disable=True):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, **kwargs)

            if i in log_num_per_prog == 0:
                intermediates.append(img.unsqueeze(0))

        #img.clamp_(-1., 1.)

        intermediates = torch.cat(intermediates, 0)  # [vis_timestep, batch, 3, W, H]

        return img, intermediates




    # training related functions - noise prediction

    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr



    def vis_schedule(self):
        _timestep = 1000
        _log_snr = self.log_snr(torch.linspace(0, 1, _timestep, dtype=torch.float64))
        wandb_dict = dict()
        ##
        data = [[x, y] for (x, y) in zip(range(_timestep), list(_log_snr))]
        table = wandb.Table(data=data, columns=["timestep", "logSNR"])
        wandb_dict.update({"logSNR_vs_time": wandb.plot.line(table, "timestep", "logSNR", title="logSNR vs time")})
        ##
        return wandb_dict


