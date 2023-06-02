import math
import torch
import numpy as np
from loguru import logger
from dynamic.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from functools import partial



def get_snr(beta_schedule):
    timesteps = 1000
    linear_start = 0.0015
    linear_end = 0.0155
    cosine_s = 8e-3

    betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                  cosine_s=cosine_s)

    alphas = 1. - betas

    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
    parameterization = "eps"
    v_posterior = 0

    assert alphas_cumprod.shape[0] == timesteps, 'alphas have to be defined for each timestep'

    to_torch = partial(torch.tensor, dtype=torch.float32)

    betas = to_torch(betas)
    alphas_cumprod = to_torch(alphas_cumprod)
    alphas_cumprod_prev = to_torch(alphas_cumprod_prev)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
    sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
    log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
    sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
    sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1))

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (1 - v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + v_posterior * betas
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    posterior_variance = to_torch(posterior_variance)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
    posterior_mean_coef1 = to_torch(
        betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
    posterior_mean_coef2 = to_torch(
        (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    if parameterization == "eps":
        lvlb_weights = betas ** 2 / (
                2 * posterior_variance * to_torch(alphas) * (1 - alphas_cumprod))
    elif parameterization == "x0":
        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
    else:
        raise NotImplementedError("mu not supported")
    # TODO how to choose this term
    lvlb_weights[0] = lvlb_weights[1]

    snr = np.log(alphas / (1 - alphas))
    return alphas_cumprod

class Tero_Sampler():
    def __init__(self,  deterministic=False, device='cuda'):
        if deterministic:
            self.S_tmin = 0
            self.S_tmax = 0
            self.S_churn = 80
            self.S_noise = 1.003
        elif False:#visually-pleasing
            self.S_tmin = 0.05
            self.S_tmax = 50
            self.S_churn = 50
            self.S_noise = 1.002
        else:
            self.S_tmin = 0.05
            self.S_tmax = 50
            self.S_churn = 80
            self.S_noise = 1.000
        self.sigma_max = 80
        self.sigma_min = 0.002
        self.rho = 7
        self.device = device


    def get_timestep_tero(self, i):
        return (self.sigma_max ** (1.0 / self.rho) +
         i * (self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1 / self.rho)) /
                (self.timestep - 1)) ** self.rho

    def get_gamma(self, t_i):
        if t_i>= self.S_tmin and t_i<= self.S_tmax:
            return min(self.S_churn / self.timestep, math.sqrt(2) - 1)# math.sqrt(2) - 1 = 0.414
        else:
            return  0.0

    def denoiser_func(self, x, t_i, i, denoise_sample_fn, denoise_sample_fn_kwargs, **kwargs):#https://github.com/lucidrains/imagen-pytorch/issues/37
        _sigma = t_i
        c_skip = 1
        c_out = - _sigma
        c_in = 1.0 / torch.sqrt(1 + _sigma**2)
        c_noise = torch.full((x.shape[0],), i, device=self.device, dtype=torch.float)
        D = c_skip * x + c_out * denoise_sample_fn(x=c_in * x, t=c_noise, **denoise_sample_fn_kwargs)
        return D

    def make_schedule(self, timestep):
        self.timestep = timestep
        self.t_i_list = [self.get_timestep_tero(i) for i in range(self.timestep + 1)]
        self.t_i_list = torch.tensor(self.t_i_list)
        self.time_ti_int = torch.tensor(list(reversed([i for i in range(self.timestep + 1)])))

    def sample(self, shape, sampling_kwargs, log_num_per_prog=100, **kwargs):
        self.make_schedule(sampling_kwargs['num_timesteps'])
        intermediates = []
        log_num_per_prog = torch.linspace(0, self.timestep - 1, log_num_per_prog, dtype=torch.int).cpu().numpy().tolist()

        x_i = torch.randn(shape, device=self.device) * self.t_i_list[0]
        for _timestep_int in range(self.timestep):
            t_i  = self.t_i_list[_timestep_int]
            gamma_i = self.get_gamma(t_i)
            t_i_hat = t_i + gamma_i * t_i
            epsilon_i = torch.randn(shape, device=self.device) * self.S_noise
            x_i_hat = x_i + torch.sqrt(t_i_hat ** 2 - t_i ** 2) * epsilon_i
            d_i = (x_i_hat - self.denoiser_func(x=x_i_hat, t_i=t_i_hat, i=self.time_ti_int[_timestep_int], sampling_kwargs=sampling_kwargs, **kwargs)) / (t_i_hat + 1e-20)
            ##########################
            t_i_1 = self.t_i_list[_timestep_int + 1]
            x_i_1_temp = x_i_hat + (t_i_1 - t_i_hat) * d_i

            if t_i_1.item() != 0:
                d_i_dot = (x_i_1_temp - self.denoiser_func(x=x_i_1_temp, t_i=t_i_1, i=self.time_ti_int[_timestep_int+1], sampling_kwargs=sampling_kwargs,**kwargs)) / (t_i_1 + 1e-20)
                x_i = x_i_hat + (t_i_1 - t_i_hat) * (d_i + d_i_dot) * 0.5  #update x_i
            else:
                raise
            if _timestep_int in log_num_per_prog:
                intermediates.append(x_i.unsqueeze(0))
                logger.warning(f'{_timestep_int}, t_i={t_i}, gamma={gamma_i}  min={x_i.min()}, max={x_i.max()}')

        intermediates = torch.concat(intermediates, 0)
        return x_i, intermediates


if __name__ == "__main__":
    def pred_func(x, t):
        return torch.randn(4, 3, 64, 64)


    ts = torch.randn(1000)
    sam = Tero_Sampler(denoise_sample_fn=pred_func, timestep=1000 - 1, ts=ts)
    _shape = (3, 64, 64)
    img, _ = sam.sample(_shape, batch_size=4)
    print(img.shape)