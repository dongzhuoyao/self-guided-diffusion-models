"""SAMPLING ONLY."""


import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from einops import rearrange, repeat
import torch.nn.functional as F
from dynamic.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)
from diffusion_utils.util import (
    right_pad_dims_to,
    clip_x0_minus_one_to_one,
    slerp_batch_torch,
)
from eval.papervis_utils import batch_to_conditioninterp_papervis
from eval.test_exps.common_stuff import should_exp, should_vis
from loguru import logger


class DDIMSampler(object):
    def __init__(self, ddpm_num_timesteps, device, sampler_type):
        super().__init__()
        self.ddpm_num_timesteps = ddpm_num_timesteps
        self.device = device
        self.sampler_type = sampler_type

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.device):
                attr = attr.to(torch.device(self.device))
        setattr(self, name, attr)

    def make_schedule(self, sampling_kwargs, ddim_discretize="uniform", **kwargs):
        ddim_num_steps = sampling_kwargs["num_timesteps"]
        ddim_eta = sampling_kwargs["ddim_eta"]
        if ddim_eta != 0 and self.sampler_type in ["plms"]:
            logger.warning(
                f"ddim_eta={ddim_eta}, it should be 0. resetting ddim_eta=0  in PLMS sampler."
            )
            ddim_eta = 0
            # raise ValueError('ddim_eta must be 0 for PLMS')
        alphas_cumprod, betas, alphas_cumprod_prev = (
            sampling_kwargs["alphas_cumprod"],
            sampling_kwargs["betas"],
            sampling_kwargs["alphas_cumprod_prev"],
        )

        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=False,
        )

        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        def to_torch(x): return x.clone().detach().to(
            torch.float32).to(self.device)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev",
                             to_torch(alphas_cumprod_prev))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=False,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas",
                             np.sqrt(1.0 - ddim_alphas))

    @torch.no_grad()
    def sample(self, shape, sampling_kwargs=None, **kwargs):

        self.make_schedule(sampling_kwargs=sampling_kwargs, **kwargs)
        if self.sampler_type == "ddim":
            samples, intermediates = self.ddim_sampling(
                shape, sampling_kwargs=sampling_kwargs, **kwargs
            )
        elif self.sampler_type == "plms":
            samples, intermediates = self.plms_sampling(
                shape, sampling_kwargs=sampling_kwargs, **kwargs
            )
        else:
            raise NotImplementedError
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, shape, sampling_kwargs, denoise_sample_fn_kwargs, **kwargs):
        log_num_per_prog = sampling_kwargs["log_num_per_prog"]
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        if "vis" in sampling_kwargs:
            vis = sampling_kwargs["vis"]
            if should_vis(vis, "condscale"):
                # sampling_kwargs['vis'].condscale_c.range
                _samples = vis.condscale_c.samples
                _cond_scale_num = 8
                #cond_key = 'cond'
                cond_key = 'layout'  # for STEGO
                _cond_scales = [
                    i * 3.0 / _cond_scale_num for i in range(_cond_scale_num)
                ]
                # _cond_scale = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                cond_scale = torch.tensor(
                    _cond_scales * _samples, device=self.device)
                cond_scale = rearrange(cond_scale, "c -> c 1 1 1")

                def create_dummy_data(data, cond_scale_n):
                    data = repeat(data, "b ... -> b k ...", k=cond_scale_n)
                    data = rearrange(data, "b k ... -> (b k) ...")
                    return data

                img = torch.randn(
                    [_samples] + list(shape[1:]), device=self.device
                )  # 'b c w h -> b c w h'
                img = create_dummy_data(img, cond_scale_n=len(_cond_scales))

                # 'b c w h -> b c w h'
                assert len(denoise_sample_fn_kwargs[cond_key]) >= _samples
                cond = denoise_sample_fn_kwargs[cond_key][:_samples]
                cond = create_dummy_data(cond, cond_scale_n=len(_cond_scales))

                assert len(cond_scale) == len(
                    img) and len(cond_scale) == len(cond)
                b = len(cond_scale)  # reset b
                logger.warning(
                    f"reset x, cond, cond_scale to length = {len(_cond_scales)} ")

                denoise_sample_fn_kwargs["cond_scale"] = cond_scale
                denoise_sample_fn_kwargs[cond_key] = cond

            if should_vis(vis, "interp"):
                denoise_sample_fn_kwargs["cond"] = batch_to_conditioninterp_papervis(
                    denoise_sample_fn_kwargs["cond"],
                    interp_num=vis.interp_c.n,
                    samples=vis.interp_c.samples,
                )
                img = repeat(
                    torch.randn(list(shape[1:]), device=self.device),
                    "c w h -> b c w h",
                    b=len(denoise_sample_fn_kwargs["cond"]),
                )
                b = len(img)
                print(
                    "ampling_kwargs['vis'].interp, using same initial z for all images in ONE batch!!!"
                )

            if should_vis(vis, "chainvis"):
                _samples = vis.chainvis_c.samples
                img = torch.randn(
                    [_samples] + list(shape[1:]), device=self.device)
                img = repeat(img, "k ... -> (k 2) ...")
                cond = denoise_sample_fn_kwargs["cond"][:_samples]
                denoise_sample_fn_kwargs["cond"] = repeat(
                    cond, "k ... -> (k 2) ...")
                assert len(denoise_sample_fn_kwargs["cond"]) == len(img)
                b = len(img)

                p0 = torch.tensor(
                    [1, 0], device=self.device, dtype=torch.float32
                )  # [1,0,1,0,1,0]
                p0 = repeat(p0, "c -> (k c)", k=_samples)
                denoise_sample_fn_kwargs["p0"] = p0

                logger.warning("chainvis !!!")

            if should_vis(vis, "scoremix_vis"):
                raise NotImplementedError
                _samples = vis.scoremix_vis_c.samples
                _interp = vis.scoremix_vis_c.interp
                _same_noise = vis.scoremix_vis_c.same_noise
                denoise_sample_fn_kwargs["interp"] = _interp
                if _same_noise:
                    img = torch.randn(
                        [_samples] + list(shape[1:]), device=self.device)
                    img = repeat(img, "k ... -> (k interp) ...",
                                 interp=_interp)
                else:
                    img = torch.randn(
                        [_samples * _interp] + list(shape[1:]), device=self.device
                    )
                logger.warning(
                    f"force reset batch size={len(img)} ({_samples}x{_interp}), same_noise={_same_noise}"
                )

                cond = denoise_sample_fn_kwargs["cond"][: _samples * 2]
                assert len(cond) == _samples * 2
                cond = rearrange(
                    cond, "(k cond_num) ... -> k cond_num ...", cond_num=2)
                cond = repeat(
                    cond,
                    "k cond_num ... -> (k interp cond_num) ...",
                    interp=_interp,
                    cond_num=2,
                )
                denoise_sample_fn_kwargs["cond"] = cond

                b = len(img)

                # formulate z: [batch * interp, ...]
                # formulate cond: [batch * interp * 2 ...]-> [batch, interp, 2...]

        if "exp" in sampling_kwargs:
            _exp = sampling_kwargs["exp"]
            if should_exp(_exp, "scoremix"):
                raise NotImplementedError
                _interp = _exp.scoremix_c.interp
                denoise_sample_fn_kwargs["interp"] = _interp
                _samples = shape[0] // 2
                _same_noise = _exp.scoremix_c.same_noise

                if _same_noise:
                    img = torch.randn(
                        [_samples] + list(shape[1:]), device=self.device)
                    img = repeat(img, "k ... -> (k interp) ...",
                                 interp=_interp)
                else:
                    img = torch.randn(
                        [_samples * _interp] + list(shape[1:]), device=self.device
                    )

                cond = denoise_sample_fn_kwargs["cond"][: _samples * 2]
                assert len(cond) == _samples * \
                    2, f"{len(cond)} != {_samples * 2}"

                cond = rearrange(
                    cond, "(k cond_num) ... -> k cond_num ...", cond_num=2)
                cond = repeat(
                    cond,
                    "k cond_num ... -> (k interp cond_num) ...",
                    interp=_interp,
                    cond_num=2,
                )
                denoise_sample_fn_kwargs["cond"] = cond

                b = len(img)
                logger.warning(
                    f"force reset batch size={len(img)} ({_samples}x{_interp}), cond_len={len(cond)}, same_noise={int(_same_noise)}"
                )

                # formulate z: [batch * interp, ...]
                # formulate cond: [batch * interp * 2 ...]-> [batch, interp, 2...]

            if should_exp(_exp, "condmix"):
                raise NotImplementedError
                _interp = _exp.condmix_c.interp
                _samples = shape[0] // 2
                _same_noise = _exp.condmix_c.same_noise

                if _same_noise:
                    img = torch.randn(
                        [_samples] + list(shape[1:]), device=self.device)
                    img = repeat(img, "k ... -> (k interp) ...",
                                 interp=_interp)
                else:
                    img = torch.randn(
                        [_samples * _interp] + list(shape[1:]), device=self.device
                    )

                if False:
                    cond = denoise_sample_fn_kwargs["cond"]
                    cond_permuted_index = torch.randperm(len(cond))
                    cond_permuted = cond[cond_permuted_index]
                else:
                    cond = denoise_sample_fn_kwargs["cond"][:_samples]
                    cond_permuted = denoise_sample_fn_kwargs["cond"][_samples:][
                        :_samples
                    ]
                assert len(cond_permuted) == len(cond)
                lin_w = torch.linspace(0, 1, _interp).to(cond.device)
                cond_list = []
                for _lin_w in lin_w:
                    cond_list.append(slerp_batch_torch(
                        _lin_w, cond, cond_permuted))
                cond = torch.cat(cond_list, 0)

                assert len(cond) == _samples * \
                    3, f"{len(cond)} != {_samples * 3}"
                denoise_sample_fn_kwargs["cond"] = cond

                b = len(img)
                logger.warning(
                    f"force reset batch size={len(img)} ({_samples}x{_interp}), cond_len={len(cond)}, same_noise={int(_same_noise)}"
                )

                # formulate z: [batch * interp, ...]
                # formulate cond: [batch * interp * 2 ...]-> [batch, interp, 2...]

        timesteps = self.ddim_timesteps
        total_steps = timesteps.shape[0]
        time_range = np.flip(timesteps)
        iterator = tqdm(
            time_range, desc="DDIM Sampler", total=total_steps, disable=True
        )

        sampling_dict = dict(pred_x0=[], x_inter=[])
        log_num_per_prog = (
            torch.linspace(0, total_steps, log_num_per_prog, dtype=torch.int)
            .cpu()
            .numpy()
            .tolist()
        )

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            img, pred_x0, pred_x0_unclipped = self.p_sample_ddim(
                img,
                ts,
                index=index,
                sampling_kwargs=sampling_kwargs,
                denoise_sample_fn_kwargs=denoise_sample_fn_kwargs,
                **kwargs,
            )

            # sampling_dict['pred_x0_unclipped_max'].append(pred_x0_unclipped.detach().cpu().max().reshape(1,-1))
            # sampling_dict['pred_x0_unclipped_min'].append(pred_x0_unclipped.detach().cpu().min().reshape(1,-1))
            if index in log_num_per_prog:
                sampling_dict["x_inter"].append(
                    img.detach().cpu().unsqueeze(0))
                sampling_dict["pred_x0"].append(
                    pred_x0.detach().cpu().unsqueeze(0))

        def torch_kitten(x): return torch.cat(x, 0)
        sampling_dict["x_inter"] = torch_kitten(sampling_dict["x_inter"])
        sampling_dict["pred_x0"] = torch_kitten(sampling_dict["pred_x0"])
        # sampling_dict['pred_x0_unclipped_max'] = torch_kitten(sampling_dict['pred_x0_unclipped_max'])
        # sampling_dict['pred_x0_unclipped_min'] = torch_kitten(sampling_dict['pred_x0_unclipped_min'])

        return img, sampling_dict

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        t,
        index,
        condition_kwargs,
        sampling_kwargs,
        denoise_sample_fn,
        denoise_sample_fn_kwargs,
        repeat_noise=False,
    ):
        b, *_, device = *x.shape, x.device
        e_t = denoise_sample_fn(x, t, **denoise_sample_fn_kwargs)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full_like(x, self.ddim_alphas[index], device=device)
        a_prev = torch.full_like(
            x, self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full_like(x, self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full_like(
            x, self.ddim_sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / \
            a_t.sqrt()  # Eq 12 in DDIM paper
        pred_x0_unclipped = pred_x0.clone()

        # clipping
        pred_x0 = clip_x0_minus_one_to_one(
            pred_x0=pred_x0,
            clip_denoised=sampling_kwargs["clip_denoised"],
            dtp=sampling_kwargs["dtp"],
        )

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * \
            e_t  # Eq 12 in DDIM paper
        noise = (
            sigma_t
            * noise_like(x.shape, device, repeat_noise)
            * sampling_kwargs["temperature"]
        )
        if sampling_kwargs["noise_dropout"] > 0.0:
            noise = F.dropout(noise, p=sampling_kwargs["noise_dropout"])
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0, pred_x0_unclipped.detach()

    @torch.no_grad()
    def plms_sampling(
        self,
        shape,
        sampling_kwargs,
        denoise_sample_fn,
        denoise_sample_fn_kwargs,
        **kwargs,
    ):
        log_num_per_prog = sampling_kwargs["log_num_per_prog"]
        b = shape[0]
        img = torch.randn(shape, device=self.device)
        timesteps = self.ddim_timesteps
        total_steps = timesteps.shape[0]
        time_range = np.flip(timesteps)

        old_eps = []
        log_num_per_prog = (
            torch.linspace(0, total_steps, log_num_per_prog, dtype=torch.int)
            .cpu()
            .numpy()
            .tolist()
        )

        sampling_dict = dict(pred_x0=[], x_inter=[])
        iterator = tqdm(
            time_range, desc="PLMS Sampler", total=total_steps, disable=True
        )
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            ts_next = torch.full(
                (b,),
                time_range[min(i + 1, len(time_range) - 1)],
                device=self.device,
                dtype=torch.long,
            )
            e_t = denoise_sample_fn(img, ts, **denoise_sample_fn_kwargs)

            if len(old_eps) == 0:
                # Pseudo Improved Euler (2nd order)
                x_prev, pred_x0 = self.p_sample_plms(
                    img,
                    ts,
                    e_t=e_t,
                    index=index,
                    sampling_kwargs=sampling_kwargs,
                    **kwargs,
                )
                e_t_next = denoise_sample_fn(
                    x_prev, ts_next, **denoise_sample_fn_kwargs
                )

                e_t_prime = (e_t + e_t_next) / 2
            elif len(old_eps) == 1:
                # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
                e_t_prime = (3 * e_t - old_eps[-1]) / 2
            elif len(old_eps) == 2:
                # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
                e_t_prime = (23 * e_t - 16 *
                             old_eps[-1] + 5 * old_eps[-2]) / 12
            elif len(old_eps) >= 3:
                # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
                e_t_prime = (
                    55 * e_t - 59 * old_eps[-1] + 37 *
                    old_eps[-2] - 9 * old_eps[-3]
                ) / 24

            img, pred_x0 = self.p_sample_plms(
                img,
                ts,
                e_t=e_t_prime,
                index=index,
                sampling_kwargs=sampling_kwargs,
                **kwargs,
            )

            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

            if index in log_num_per_prog:
                sampling_dict["pred_x0"].append(pred_x0.unsqueeze(0))
                sampling_dict["x_inter"].append(img.unsqueeze(0))

        sampling_dict["pred_x0"] = torch.cat(sampling_dict["pred_x0"], 0)
        sampling_dict["x_inter"] = torch.cat(sampling_dict["x_inter"], 0)

        return img, sampling_dict

    @torch.no_grad()
    def p_sample_plms(
        self,
        x,
        t,
        e_t,
        index,
        condition_kwargs,
        sampling_kwargs,
        repeat_noise=False,
    ):
        b, *_, device = *x.shape, x.device
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full_like(x, self.ddim_alphas[index], device=device)
        a_prev = torch.full_like(
            x, self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full_like(x, self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full_like(
            x, self.ddim_sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / \
            a_t.sqrt()  # Eq 12 in DDIM paper
        # clipping
        pred_x0 = clip_x0_minus_one_to_one(
            pred_x0=pred_x0,
            clip_denoised=sampling_kwargs["clip_denoised"],
            dtp=sampling_kwargs["dtp"],
        )

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * \
            e_t  # Eq 12 in DDIM paper
        noise = (
            sigma_t
            * noise_like(x.shape, device, repeat_noise)
            * sampling_kwargs["temperature"]
        )
        if sampling_kwargs["noise_dropout"] > 0.0:
            noise = F.dropout(noise, p=sampling_kwargs["noise_dropout"])
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
