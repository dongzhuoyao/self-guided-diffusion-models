from pathlib import Path
from loguru import logger
import copy
import torch

import wandb
from dataset.ds_utils.dataset_common_utils import need_to_upsample256

from diffusion_utils.util import clip_unnormalize_to_zero_to_255, delete_dir

from eval.eval_knn import get_knn_eval_dict
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision
from einops import rearrange, repeat
from PIL import Image


def img_pil_save(img_torch, save_path, pil_mode=None):
    if pil_mode == None:
        img = rearrange(
            img_torch, "c w h->w h c").to(torch.uint8).cpu().numpy()
    elif pil_mode == 'L':
        img = img_torch.to(torch.uint8).cpu().numpy()
    else:
        raise NotImplementedError
    img_pil = Image.fromarray(img, mode=pil_mode)
    img_pil.save(save_path)


def should_exp(exp, vis_exp_str):
    return hasattr(exp, vis_exp_str) and getattr(exp, vis_exp_str)


def should_vis(vis, vis_exp_str):
    return hasattr(vis, vis_exp_str) and getattr(vis, vis_exp_str)


def get_save_dir(pl_module):
    return str(Path(pl_module.trainer.logdir).expanduser().resolve())


def get_condition_scale_list(pl_module):
    # cond_scale_list = [1.25, 1, 2, 0, 4, 6, 10]
    if pl_module.hparams.cond_scale == 0:
        return [0]
    else:
        return [pl_module.hparams.cond_scale, 0]


def get_condition_scale_main(pl_module):
    return pl_module.hparams.cond_scale


def update_cfg_and_print(_kwargs, udpate_info_dict, cfg_name):
    _kwargs_current = copy.deepcopy(_kwargs)
    _kwargs_current.update(udpate_info_dict)
    logger.warning("{}: {}".format(cfg_name, _kwargs_current))
    return _kwargs_current


def sampling_str(sampling_kwargs):
    sampling_method = sampling_kwargs["sampling_method"]
    num_timesteps = sampling_kwargs["num_timesteps"]
    return f"{sampling_method}{num_timesteps}"


def cond_str(condition_kwargs):
    cond_scale = condition_kwargs["cond_scale"]

    return f"s{cond_scale}"


def sampling_cond_str(sampling_kwargs, condition_kwargs):
    return f"{sampling_str(sampling_kwargs)}_{cond_str(condition_kwargs)}"


def get_sample_fn(pl_module, sampling_cfg, condition_cfg, prefix):
    LOG_FREQUENCY = 10
    if sampling_cfg["sampling_method"] == "directimage":

        def sample_fn(batch, subset, batch_id, batch_size, prefix=""):
            batch["image"] = clip_unnormalize_to_zero_to_255(batch["image"])
            return batch["image"], None

    else:

        def sample_fn(batch, subset, batch_id, batch_size, prefix=""):
            samples, pred_x0 = pl_module.sampling_progressive(
                batch_size=batch_size,
                batch_data=batch,
                sampling_kwargs=sampling_cfg,
                condition_kwargs=condition_cfg,
            )
            return samples, pred_x0

        def log_img_wrapper(fn):
            def _fn(*args, **kwargs):
                samples, pred_x0 = fn(*args, **kwargs)
                if kwargs["batch_id"] % LOG_FREQUENCY == 0:
                    samples_wandb = [
                        wandb.Image(_sample.float()) for _sample in samples
                    ]
                    wandb_dict = {
                        f"{prefix}_eval_sample_{sampling_cond_str(sampling_cfg,condition_cfg)}": samples_wandb
                    }
                    pl_module.logger.experiment.log(wandb_dict)
                return samples, pred_x0

            return _fn

        sample_fn = log_img_wrapper(sample_fn)

    return sample_fn


def sample_and_get_fid(
    pl_module, prefix, condition_cfg, fid_cfg, sampling_cfg, debug=False
):

    from eval.eval_fid import eval_fid

    metric_dict = {"current_epoch": pl_module.trainer.current_epoch}

    vis_knn = fid_cfg["vis_knn"]
    logger.warning(fid_cfg["save_dir"])
    fid_cfg["sample_dir"] = fid_cfg["sample_dir"] + \
        f"_rank{pl_module.global_rank}"
    if pl_module.global_rank == 0:
        logger.warning("append rank id: {}".format(fid_cfg["sample_dir"]))

    sample_fn = get_sample_fn(
        pl_module=pl_module,
        sampling_cfg=sampling_cfg,
        condition_cfg=condition_cfg,
        prefix=prefix,
    )
    prepare_batch_fn = pl_module.prepare_batch
    fid_dict, fid_for_ckpt, sample_dir, gt_dir = eval_fid(
        sample_fn=sample_fn,
        prepare_batch_fn=prepare_batch_fn,
        fid_kwargs=fid_cfg,
        prefix=prefix,
        debug=debug,
    )
    metric_dict.update(fid_dict)
    if vis_knn:
        knn_dict = get_knn_eval_dict(
            sample_dir=sample_dir,
            gt_dir_4fid=gt_dir,
            fid_kwargs=fid_cfg,
            knn_k=32,
            q_num=10,
            debug=debug,
        )
        metric_dict.update(knn_dict)

    # delete_dir(sample_dir), wandb later update still need this dir
    metric_dict = {f"{prefix}_" + k: v for k, v in metric_dict.items()}
    return metric_dict, fid_for_ckpt
