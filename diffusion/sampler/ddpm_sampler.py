from collections import namedtuple
from functools import partial

import numpy as np
import torch
from einops import rearrange
from torch import nn
from tqdm import tqdm

from dynamic.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from diffusion_utils.util import exists, default, right_pad_dims_to, dict2obj, clip_x0_minus_one_to_one
import pytorch_lightning as pl
from diffusion_utils.taokit.wandb_utils import vis_schedule_ddpm, vis_timestep_loss


class Schedule_DDPM(nn.Module):
    """main class"""

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = dict2obj(kwargs)
        self.register_schedule(given_betas=self.hparams.given_betas, beta_schedule=self.hparams.beta_schedule, timesteps=self.hparams.num_timesteps,
                               linear_start=self.hparams.linear_start, linear_end=self.hparams.linear_end, cosine_s=self.hparams.cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, self.hparams.num_timesteps, linear_start=linear_start,
                                       linear_end=linear_end,
                                       cosine_s=cosine_s)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        if timesteps < self.hparams.num_timesteps:
            raise NotImplementedError
            c = self.hparams.num_timesteps // timesteps
            short_range_index = np.array(
                list(range(0, self.hparams.num_timesteps, c)))
            alphas = alphas[short_range_index]
            betas = betas[short_range_index]
            alphas_cumprod = alphas_cumprod[short_range_index]
            alphas_cumprod_prev = alphas_cumprod_prev[short_range_index]

        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas).to(self.hparams.device))
        self.register_buffer('alphas_cumprod', to_torch(
            alphas_cumprod).to(self.hparams.device))
        self.register_buffer('alphas_cumprod_prev', to_torch(
            alphas_cumprod_prev).to(self.hparams.device))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(
            np.sqrt(alphas_cumprod)).to(self.hparams.device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(
            np.sqrt(1. - alphas_cumprod)).to(self.hparams.device))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(
            np.log(1. - alphas_cumprod)).to(self.hparams.device))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod)).to(self.hparams.device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod - 1)).to(self.hparams.device))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.hparams.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.hparams.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(
            posterior_variance).to(self.hparams.device))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))).to(self.hparams.device))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.hparams.device))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.hparams.device))

        if self.hparams.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas).to(self.hparams.device) * (1 - self.alphas_cumprod))
        elif self.hparams.parameterization == "x0":
            lvlb_weights = 0.5 * \
                np.sqrt(torch.Tensor(alphas_cumprod)) / \
                (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights.to(
            self.hparams.device), persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

        self.register_buffer('snr_derivative', torch.zeros(1000, dtype=torch.float32).to(
            self.hparams.device))  # dummy value, will remove in the fugure
        self.register_buffer('SNR', torch.zeros(1000, dtype=torch.float32).to(
            self.hparams.device))  # dummy value, will remove in the fugure

    def time_to_sigma(self, timestep):
        _sigma2 = 1. - self.alphas_cumprod
        return torch.sqrt(_sigma2)[timestep]

    def sigma_to_time_int(self, sigma):
        _sigma = torch.sqrt(1. - self.alphas_cumprod)
        abs_delta = torch.abs(_sigma.reshape(
            1, -1) - sigma.reshape(-1, 1))  # [Batch, 1000]
        t = torch.argmin(abs_delta, dim=-1).long()
        return t

    def q_sample(self, original_sample, noise, t):
        noise = default(noise, lambda: torch.randn_like(original_sample))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, original_sample.shape) * original_sample +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, original_sample.shape) * noise)

    def q_posterior(self, original_sample, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * original_sample +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(
            1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def p_mean_variance(self, x, t, sampling_kwargs, denoise_sample_fn, denoise_sample_fn_kwargs=None, **kwargs):

        model_out = denoise_sample_fn(x, t, **denoise_sample_fn_kwargs)

        if self.hparams.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.hparams.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        x0_unclipped = x_recon.clone()

        x_recon = clip_x0_minus_one_to_one(
            pred_x0=x_recon, clip_denoised=sampling_kwargs['clip_denoised'], dtp=sampling_kwargs['dtp'])

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            original_sample=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, x_recon, x0_unclipped

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False,
                 temperature=1., noise_dropout=0., **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0, x0_unclipped = self.p_mean_variance(x=x, t=t,
                                                                                   clip_denoised=clip_denoised,
                                                                                   **kwargs)

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))

        p_sampled = model_mean + nonzero_mask * \
            (0.5 * model_log_variance).exp() * noise
        return p_sampled, x0, x0_unclipped

    @torch.no_grad()
    def sample(self, shape, sampling_kwargs=None, **kwargs):
        temperature = sampling_kwargs['temperature']
        noise_dropout = sampling_kwargs['noise_dropout']
        log_num_per_prog = sampling_kwargs['log_num_per_prog']
        timesteps = sampling_kwargs['num_timesteps']
        batch_size = shape[0]
        self.register_schedule(timesteps=timesteps,
                               given_betas=self.hparams.given_betas, beta_schedule=self.hparams.beta_schedule,
                               linear_start=self.hparams.linear_start, linear_end=self.hparams.linear_end,
                               cosine_s=self.hparams.cosine_s)

        if 'vis' in sampling_kwargs and hasattr(sampling_kwargs['vis'], 'interp') and sampling_kwargs['vis'].interp:
            img = torch.randn([1]+list(shape[1:]), device=self.hparams.device)
            img = rearrange(img, '1 c w h -> b c w h', b=shape[0])
            print(
                "ampling_kwargs['vis'].interp, using same initial z for all images in ONE batch!!!")
        else:
            img = torch.randn(shape, device=self.hparams.device)

        sampling_dict = dict(pred_x0=[], x_inter=[])
        iterator = tqdm(reversed(range(0, timesteps)),
                        total=timesteps, disable=True)
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        log_num_per_prog = torch.linspace(
            0, timesteps, log_num_per_prog, dtype=torch.int).cpu().numpy().tolist()
        for i in iterator:
            ts = torch.full((batch_size,), i,
                            device=self.hparams.device, dtype=torch.long)
            img, x0_recon, pred_x0_unclipped = self.p_sample(x=img, t=ts,
                                                             temperature=temperature[i], noise_dropout=noise_dropout,
                                                             sampling_kwargs=sampling_kwargs, **kwargs)

            if i in log_num_per_prog:
                sampling_dict['pred_x0'].append(x0_recon.unsqueeze(0))
                sampling_dict['x_inter'].append(img.unsqueeze(0))

        sampling_dict['pred_x0'] = torch.cat(
            sampling_dict['pred_x0'], 0)  # [vis_timestep, batch, 3, W, H]
        sampling_dict['x_inter'] = torch.cat(
            sampling_dict['x_inter'], 0)  # [vis_timestep, batch, 3, W, H]

        return img, sampling_dict

    def vis_schedule(self):
        wandb_dict = vis_schedule_ddpm(_betas=self.betas.cpu(), _alphas_cumprod=self.alphas_cumprod.cpu(),
                                       _snr_derivative=None)
        return wandb_dict
