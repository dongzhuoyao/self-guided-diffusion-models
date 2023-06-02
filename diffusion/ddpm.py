"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import copy

from torch import nn

from diffusion_utils.util import default, clip_unnormalize_to_zero_to_255, dict2obj
from diffusion.sampler.pndm_sampler import PNDM_Sampler
from diffusion.sampler.ddim_plms_sampler import DDIMSampler
from diffusion.sampler.ddpm_sampler import Schedule_DDPM
import torch.nn.functional as F
import torch
from diffusion.sampler.tero_sampler import Tero_Sampler

from einops import reduce


class LatentDiffusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = dict2obj(kwargs)
        self.sampler = Schedule_DDPM(**kwargs)
        self.sampler_list = {
            'native': self.sampler,
            'ddim': DDIMSampler(ddpm_num_timesteps=self.hparams.num_timesteps, device=self.hparams.device, sampler_type='ddim'),
            'plms': DDIMSampler(ddpm_num_timesteps=self.hparams.num_timesteps, device=self.hparams.device, sampler_type='plms'),
            'pndm': PNDM_Sampler(ddpm_num_timesteps=self.hparams.num_timesteps, beta_start=self.hparams.linear_start, beta_end=self.hparams.linear_end, beta_schedule=self.hparams.beta_schedule, device=self.hparams.device),
            'tero': Tero_Sampler(device=self.hparams.device)
        }

    def set_denoise_fn(self, denoise_fn, denoise_sample_fn):
        self.denoise_fn = denoise_fn

        def _denoise_sample_fn(*args, **kwargs):
            score = denoise_sample_fn(*args, **kwargs)
            return score

        self.denoise_sample_fn = _denoise_sample_fn

    def forward_tao(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def forward(self, x, *args, **kwargs):
        n_batch = len(x)
        t = torch.randint(0, self.hparams.num_timesteps,
                          (n_batch,), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def p_losses(self, x_start, t, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.sampler.q_sample(
            original_sample=x_start, t=t, noise=noise)
        model_output, _loss_inside, loss_dict_inside = self.denoise_fn(
            x_noisy, t, *args, **kwargs)

        loss_dict = {}

        prefix = 'train' if self.training else 'val'
        loss_dict.update(**{f'{prefix}/{k}': v for k,
                         v in loss_dict_inside.items()},)

        if self.hparams.parameterization == "x0":
            target = x_start
        elif self.hparams.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss = self.get_loss(model_output, target, mean=False)
        loss = reduce(loss, 'b ... -> b', 'mean')
        if prefix == 'train':
            loss_dict.update({f'{prefix}/epoch_stats_y': loss.detach()})
            loss_dict.update({f'{prefix}/epoch_stats_x': t.detach()})

        loss_weighted = loss
        loss = loss_weighted.mean()
        loss_dict.update({f'{prefix}/ddpm_loss': loss.detach()})
        loss = loss + _loss_inside
        loss_dict.update({f'{prefix}/loss': loss.detach()})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.hparams.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.hparams.loss_type == 'l2':
            if mean:
                loss = F.mse_loss(target, pred)
            else:
                loss = F.mse_loss(target, pred, reduction='none')
        elif self.hparams.loss_type == 'huber':
            if mean:
                loss = F.smooth_l1_loss(target, pred)
            else:
                loss = F.smooth_l1_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @torch.no_grad()
    def p_sample_loop(self, sampling_method, shape, sampling_kwargs, **kwargs):
        sampling_kwargs_current = copy.deepcopy(sampling_kwargs)
        # DDIM needed it,TODO more elegant impl?
        sampling_kwargs_current.update(dict(alphas_cumprod=self.sampler.alphas_cumprod,
                                       alphas_cumprod_prev=self.sampler.alphas_cumprod_prev, betas=self.sampler.betas))

        samples, intermediates = self.sampler_list[sampling_method].sample(shape=shape,
                                                                           denoise_sample_fn=self.denoise_sample_fn,
                                                                           sampling_kwargs=sampling_kwargs_current,
                                                                           **kwargs)
        samples = clip_unnormalize_to_zero_to_255(samples)
        intermediates['pred_x0'] = clip_unnormalize_to_zero_to_255(
            intermediates['pred_x0'])
        return samples, intermediates

    def vis_schedule(self):
        wandb_dict = self.sampler.vis_schedule()
        return wandb_dict
