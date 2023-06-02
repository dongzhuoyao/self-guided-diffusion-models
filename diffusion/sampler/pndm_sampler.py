"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from einops import rearrange
from diffusion_utils.util import right_pad_dims_to, clip_x0_minus_one_to_one
import math
from diffusion.sampler.utils.huggingface.scheduling_utils import SchedulerMixin, betas_for_alpha_bar, linear_beta_schedule
from diffusion.sampler.utils.huggingface.configuration_utils import ConfigMixin

class PNDMScheduler(SchedulerMixin, ConfigMixin):
    def __init__(
        self,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        tensor_format="np",
    ):
        super().__init__()
        self.register(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )
        self.timesteps = timesteps

        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        elif beta_schedule == "squaredcos_cap_v2":
            # GLIDE cosine schedule
            self.betas = betas_for_alpha_bar(
                timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        if True:#add by taohu to avoid overflow: " at = alphas_cump[t + 1].view(-1, 1, 1, 1)"
            self.alphas_cumprod = np.array((list(self.alphas_cumprod) + [0.0]),dtype=np.float32)

        self.one = np.array(1.0)

        self.set_format(tensor_format=tensor_format)

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at equations (12) and (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_residual = 0
        self.cur_image = None
        self.ets = []
        self.warmup_time_steps = {}
        self.time_steps = {}

    def get_alpha(self, time_step):
        return self.alphas[time_step]

    def get_beta(self, time_step):
        return self.betas[time_step]

    def get_alpha_prod(self, time_step):
        if time_step < 0:
            return self.one
        return self.alphas_cumprod[time_step]

    def get_warmup_time_steps(self, num_inference_steps):
        if num_inference_steps in self.warmup_time_steps:
            return self.warmup_time_steps[num_inference_steps]

        inference_step_times = list(range(0, self.timesteps, self.timesteps // num_inference_steps))

        warmup_time_steps = np.array(inference_step_times[-self.pndm_order :]).repeat(2) + np.tile(
            np.array([0, self.timesteps // num_inference_steps // 2]), self.pndm_order
        )
        self.warmup_time_steps[num_inference_steps] = list(reversed(warmup_time_steps[:-1].repeat(2)[1:-1]))

        return self.warmup_time_steps[num_inference_steps]

    def get_time_steps(self, num_inference_steps):
        if num_inference_steps in self.time_steps:
            return self.time_steps[num_inference_steps]

        inference_step_times = list(range(0, self.timesteps, self.timesteps // num_inference_steps))
        self.time_steps[num_inference_steps] = list(reversed(inference_step_times[:-3]))

        return self.time_steps[num_inference_steps]

    def step_prk(self, residual, image, t, num_inference_steps):

        warmup_time_steps = self.get_warmup_time_steps(num_inference_steps)

        t_prev = warmup_time_steps[t // 4 * 4]
        t_next = warmup_time_steps[min(t + 1, len(warmup_time_steps) - 1)]

        if t % 4 == 0:
            self.cur_residual += 1 / 6 * residual
            self.ets.append(residual)
            self.cur_image = image
        elif (t - 1) % 4 == 0:
            self.cur_residual += 1 / 3 * residual
        elif (t - 2) % 4 == 0:
            self.cur_residual += 1 / 3 * residual
        elif (t - 3) % 4 == 0:
            residual = self.cur_residual + 1 / 6 * residual
            self.cur_residual = 0

        return self.transfer(self.cur_image, t_prev, t_next, residual)

    def step_plms(self, residual, image, t, num_inference_steps):
        timesteps = self.get_time_steps(num_inference_steps)

        t_prev = timesteps[t]
        t_next = timesteps[min(t + 1, len(timesteps) - 1)]
        self.ets.append(residual)

        residual = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        return self.transfer(image, t_prev, t_next, residual)

    def transfer(self, x, t, t_next, et):


        alphas_cump = self.alphas_cumprod.to(x.device)
        at = alphas_cump[t + 1].view(-1, 1, 1, 1)
        at_next = alphas_cump[t_next + 1].view(-1, 1, 1, 1)

        x_delta = (at_next - at) * (
            (1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x
            - 1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et
        )#Eq 9) in PNDM paper.

        x_next = x + x_delta
        return x_next

    def __len__(self):
        return self.timesteps


class PNDM_Sampler(object):
    def __init__(self,  ddpm_num_timesteps, beta_start, beta_end, beta_schedule="linear", tensor_format='pt', device='cuda'):
        super().__init__()
        self.ddpm_num_timesteps = ddpm_num_timesteps
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.tensor_format = tensor_format


    @torch.no_grad()
    def sample(self,
               shape,
               sampling_kwargs,
               log_num_per_prog=100,
               **kwargs
               ):
        self.num_inference_steps = sampling_kwargs['num_timesteps']
        if self.num_inference_steps>250:
            logger.warning('according to the PNDM paper, most gains can be reaped when timestep<250, so it is not meaningful to set a timestep larger than 250')
        self.noise_scheduler = PNDMScheduler(timesteps=self.ddpm_num_timesteps, beta_start=self.beta_start,
                                             beta_end=self.beta_end, tensor_format=self.tensor_format)
        samples, intermediates = self.pndm_sampling(shape,
                                                    log_num_per_prog=log_num_per_prog,
                                                    sampling_kwargs=sampling_kwargs,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def pndm_sampling(self, shape, denoise_sample_fn,denoise_sample_fn_kwargs, log_num_per_prog, condition_kwargs=None, sampling_kwargs=None, **kwargs):
        device = self.device
        image = torch.randn(shape, device=device)
        warmup_time_steps = self.noise_scheduler.get_warmup_time_steps(self.num_inference_steps)

        iterator = tqdm(range(len(warmup_time_steps)), disable=sampling_kwargs['disable_tqdm'])
        for _id, t in enumerate(iterator):
            t_orig = warmup_time_steps[t]
            #########################
            t_orig =torch.full(fill_value=t_orig, size=(image.shape[0],)).to(image.device)
            residual = denoise_sample_fn(image, t_orig, **denoise_sample_fn_kwargs)
            ########################
            image = self.noise_scheduler.step_prk(residual, image, t, self.num_inference_steps)

            if False:
                image = clip_x0_minus_one_to_one(pred_x0=image, clip_denoised=sampling_kwargs['clip_denoised'],
                                                 dtp=sampling_kwargs['dtp'])

        timesteps = self.noise_scheduler.get_time_steps(self.num_inference_steps)
        #logger.warning(f'timesteps, {len(timesteps)}')
        iterator = tqdm(range(len(timesteps)), disable=sampling_kwargs['disable_tqdm'])
        for _id, t in enumerate(iterator):
            t_orig = timesteps[t]
            #######################
            t_orig = torch.full(fill_value=t_orig, size=(image.shape[0],)).to(image.device)
            residual = denoise_sample_fn(image, t_orig,  **denoise_sample_fn_kwargs)
            ########################
            image = self.noise_scheduler.step_plms(residual, image, t, self.num_inference_steps)

            if False:
                image = clip_x0_minus_one_to_one(pred_x0=image, clip_denoised=sampling_kwargs['clip_denoised'],
                                                   dtp=sampling_kwargs['dtp'])

        return image, dict(pred_x0=image)
