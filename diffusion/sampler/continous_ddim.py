"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from loguru import  logger
from einops import rearrange
from dynamic.diffusionmodules.util import  noise_like
from diffusion_utils.util import  right_pad_dims_to


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        logger.info(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula 16 provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        logger.info(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        logger.info(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

class DDIMSampler_Continuous(object):
    def __init__(self, num_timesteps, schedule="linear", device='cuda'):
        super().__init__()
        self.ddpm_num_timesteps = num_timesteps
        self.schedule = schedule
        self.device = device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, alpha_fn, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = alpha_fn(torch.linspace(0, 1, self.ddpm_num_timesteps, dtype=torch.float64))
        alphas_cumprod_prev= torch.concat([torch.Tensor(np.array(1.0)).reshape(-1,), alphas_cumprod[:-1]], 0)
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        ######################## ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('denoise_sample_fn', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def p_sample_loop(self,
                      shape,
                      temperature=1.,
                      noise_dropout=0.,
                      verbose=True,
                      log_num_per_prog=100,
                      alpha_fn = None,
                      # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
                      **kwargs
                      ):

        self.make_schedule(alpha_fn=alpha_fn,ddim_num_steps=kwargs['sampling_kwargs']['num_timesteps'], ddim_eta=kwargs['sampling_kwargs']['ddim_eta'], verbose=verbose)
        # sampling
        batch_size, C, H, W = shape
        size = (batch_size, C, H, W)
        #print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        samples, intermediates = self.ddimc_sampling(size,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,

                                                    log_num_per_prog=log_num_per_prog,
                                                     **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddimc_sampling(self, shape, ddim_use_original_steps=False,
                      timesteps=None, log_num_per_prog=100,
                      temperature=1., noise_dropout=0.,  **kwargs):
        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates_x_inter = []
        intermediates_pred_x0 = []
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=kwargs['sampling_kwargs']['disable_tqdm'])

        log_num_per_prog = torch.linspace(0, total_steps, log_num_per_prog, dtype=torch.int).cpu().numpy().tolist()

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            img, pred_x0 = self.p_sample_ddim(img, ts, index=index, use_original_steps=ddim_use_original_steps, temperature=temperature,
                                      noise_dropout=noise_dropout,  **kwargs)

            if index in log_num_per_prog:
                intermediates_x_inter.append(img.unsqueeze(0))
                intermediates_pred_x0.append(pred_x0.unsqueeze(0))

            #if index%10==0:
            #    logger.warning(f'{index}, min={img.min()}, max={img.max()}')

        intermediates_x_inter = torch.cat(intermediates_x_inter, 0)
        intermediates_pred_x0 = torch.cat(intermediates_pred_x0, 0)

        return img, intermediates_pred_x0

    @torch.no_grad()
    def p_sample_ddim(self, x, t, index, condition_kwargs, sampling_kwargs, batch_data, repeat_noise=False, use_original_steps=False,
                      temperature=1.,denoise_fn=None,  noise_dropout=0.,):
        b, *_, device = *x.shape, x.device

        e_t = denoise_fn(x, t,  cond_scale=condition_kwargs['cond_scale'], **batch_data)

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()##Eq 12 in DDIM paper

        ################################## clipping
        if sampling_kwargs['dtp'] < 1.0:#https://github.com/lucidrains/imagen-pytorch/blob/bfe761b52c93f53c1a961c0912bed3b33042382c/imagen_pytorch/imagen_pytorch.py#L1382
            s = torch.quantile(
                rearrange(pred_x0, 'b ... -> b (...)').abs(),
                sampling_kwargs['dtp'],
                dim=-1
            )
            s.clamp_(min=1.)#DTP only take effect if min(s)>1
            s = right_pad_dims_to(pred_x0, s)
            pred_x0 = pred_x0.clamp(-s, s) / s
        else:#if self.dtp is default 1, don't apply DTP
            if sampling_kwargs['clip_denoised']:
                pred_x0.clamp_(-1., 1.)
        ################################## clipping

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t #Eq 12 in DDIM paper
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
