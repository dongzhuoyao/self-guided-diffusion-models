import copy
from pathlib import Path
from loguru import logger

import numpy as np
import torch
import wandb
from einops import rearrange
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import Callback
from PIL import Image
import time
from torchvision.utils import make_grid
from dataset.coco14_vqdiffusion import CocoDataset
from dataset.coco17stuff27 import CocoStuffDataset
from dataset.ds_utils.dataset_common_utils import need_to_upsample256
from dataset.voc12 import VOCSegmentation

from diffusion_utils.util import batch_to_same_firstimage, batch_to_conditioninterp
from diffusion_utils.taokit.wandb_utils import wandb_scatter_fig
from dynamic_input.condition import prepare_denoise_fn_kwargs_4sampling
from dynamic_input.misc import is_need_to_run_interp_condition
from eval.test_exps.common_stuff import sampling_cond_str
from diffusion_utils.taokit.vis_utils import upsample_pt


def update_cfg_dict(_kwargs, update_info_dict, cfg_name):
    _kwargs_new = copy.deepcopy(_kwargs)
    _kwargs_new.update(update_info_dict)
    logger.warning("{}: {}".format(cfg_name, _kwargs_new))
    return _kwargs_new


def _get_wandbimg(samples):
    assert isinstance(
        samples, torch.Tensor
    )  # if it is tensor, split along first dimension
    # n_log_step, n_row, C, H, W
    denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
    denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
    wandb_list = [wandb.Image(_i.float()) for _i in denoise_grid]
    return wandb_list


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
    ):
        super().__init__()

        self.batch_frequency = batch_frequency
        self.max_images = max_images

        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        if self.check_frequency(check_idx) and self.max_images > 0:
            if pl_module.training:
                pl_module.eval()

            with torch.no_grad():
                if hasattr(pl_module, "log_images"):
                    batch = pl_module.log_images(batch, subset=split)
                    self.log_wandb(pl_module, batch=batch)
                else:
                    logger.warning(
                        "log_images function is missing in pl_module, skip the vis.."
                    )

            if pl_module.training:
                pl_module.train()

    def log_sample_and_prog(
        self, log, samples, inter_dict, _key, prog_vis_num, need_to_upsample256=False
    ):
        log[f"{_key}_max"] = samples.max().item()
        log[f"{_key}_min"] = samples.min().item()
        if need_to_upsample256:
            samples = [wandb.Image(upsample_pt(_i).float()) for _i in samples]
        else:
            samples = [wandb.Image(_i.float()) for _i in samples]
        log[_key] = samples

        # pred_x0_unclipped
        if "pred_x0_unclipped_max" in inter_dict:
            prog_pred_x0_unclipped_max = inter_dict["pred_x0_unclipped_max"]
            prog_pred_x0_unclipped_min = inter_dict["pred_x0_unclipped_min"]
            timestep = len(prog_pred_x0_unclipped_max)
            max_list = [prog_pred_x0_unclipped_max[i] for i in range(timestep)]
            min_list = [prog_pred_x0_unclipped_min[i] for i in range(timestep)]
            x_list = list(range(timestep))
            log.update(
                wandb_scatter_fig(
                    x_list=x_list, y_list=max_list, dict_key=f"{_key}_unclipped_max"
                )
            )
            log.update(
                wandb_scatter_fig(
                    x_list=x_list, y_list=min_list, dict_key=f"{_key}_unclipped_min"
                )
            )

        # progressive
        prog = inter_dict["pred_x0"]  # [timestep_vis, b, 3 ,w, h]
        prog = prog[
            :, :prog_vis_num
        ]  # [timestep_vis, b, 3 ,w, h], truncate to progressive_vis_num
        prog_wandblist = _get_wandbimg(prog)
        log[f"{_key}_prog"] = prog_wandblist
        return log

    def need_draw_mask(self, pl_module):
        return pl_module.hparams.condition_method in ["clusterlayout", "layout", "stegoclusterlayout"]

    def get_maskdata_and_label(self, pl_module, batch):
        if pl_module.hparams.condition_method in ["clusterlayout"]:
            _how = pl_module.hparams.condition.clusterlayout.how
            _stego_k = pl_module.hparams.condition.clusterlayout.stego_k
        elif pl_module.hparams.condition_method in ["stegoclusterlayout"]:
            _how = pl_module.hparams.condition.stegoclusterlayout.how
            _stego_k = pl_module.hparams.condition.stegoclusterlayout.stego_k
        elif pl_module.hparams.condition_method in ["layout"]:
            _how = pl_module.hparams.condition.layout.how
            _stego_k = pl_module.hparams.condition.layout.stego_k
        else:
            raise NotImplementedError

        if _how == "oracle":
            data_name = pl_module.hparams.data.name
            if data_name.startswith("cocostuff"):
                class_labels = CocoStuffDataset.class_names_4wandb
            elif data_name.startswith("coco"):
                class_labels = CocoDataset.class_names_4wandb
            elif data_name.startswith("voc"):
                class_labels = VOCSegmentation.class_names_4wandb
            else:
                raise ValueError(data_name)
            return batch["segmask"], class_labels

        elif _how == "lost":
            class_labels = {0: "bg", 1: "fg"}
            return batch["lostbboxmask"], class_labels

        elif _how == "stego":
            class_labels = {i: f"stego_k{i}" for i in range(_stego_k)}
            return batch["stegomask"], class_labels

        else:
            raise ValueError(_how)

    def vis_segmask_if_needed(self, _input_images, batch, pl_module):
        if self.need_draw_mask(pl_module):
            _all_wandb = []
            _mask_datas, class_labels = self.get_maskdata_and_label(
                pl_module, batch)
            for _cur_img, _cur_mask in zip(_input_images, _mask_datas):
                if len(_cur_mask.cpu().numpy()) == 1:
                    mask_data = _cur_mask.cpu().numpy().squeeze(0)
                else:
                    mask_data = np.argmax(_cur_mask.cpu().numpy(), axis=0)
                _all_wandb.append(
                    wandb.Image(
                        _cur_img,
                        masks={
                            "predictions": {
                                "mask_data": mask_data,
                                "class_labels": class_labels,
                            },
                        },
                    )
                )
        else:
            _all_wandb = [wandb.Image(_img.float()) for _img in _input_images]

        return _all_wandb

    def log_wandb(self, pl_module, batch, vis_num=16, prog_vis_num=9):
        log = dict()
        vis_num = min(len(batch["image"]), vis_num)
        log["inputs"] = batch["image"][:vis_num].to(
            pl_module.device)  # [B,3,W,H]
        log["inputs"] = self.vis_segmask_if_needed(
            log["inputs"], batch=batch, pl_module=pl_module
        )
        for k in batch:
            batch[k] = batch[k][:vis_num].to(pl_module.device)

        condition_kwargs, sampling_kwargs, fid_kwargs = pl_module.get_default_config()
        ######         update some common params #################################
        fid_kwargs.update(
            save_dir=str(Path(pl_module.trainer.logdir).expanduser().resolve())
        )
        sampling_kwargs.update(
            dict(
                num_timesteps=pl_module.hparams.model.num_timesteps_imagelogger,
                sampling_method=pl_module.hparams.model.sampling_imagelogger,
            )
        )

        with pl_module.ema_scope("Plotting Native Sampling"):
            for cond_scale in [0, pl_module.hparams.cond_scale]:

                condition_cfg_now = update_cfg_dict(
                    condition_kwargs,
                    update_info_dict=dict(cond_scale=cond_scale),
                    cfg_name="condition_kwargs",
                )
                sampling_cfg_now = update_cfg_dict(
                    sampling_kwargs,
                    update_info_dict=dict(return_inter_dict=True),
                    cfg_name="sampling_kwargs",
                )
                samples, inter_dict = pl_module.sampling_progressive(
                    batch_size=batch["image"].shape[0],
                    batch_data=batch,
                    condition_kwargs=condition_cfg_now,
                    sampling_kwargs=sampling_cfg_now,
                )

                log = self.log_sample_and_prog(
                    log=log,
                    _key=f"native_{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
                    samples=samples,
                    inter_dict=inter_dict,
                    prog_vis_num=prog_vis_num,
                    need_to_upsample256=need_to_upsample256(
                        pl_module.hparams.data.name
                    ),
                )

        if pl_module.hparams.condition_method is not None:
            with pl_module.ema_scope("Native Sampling based on sample condition"):
                condition_cfg_now = update_cfg_dict(
                    condition_kwargs,
                    update_info_dict=dict(
                        cond_scale=pl_module.hparams.cond_scale),
                    cfg_name="condition_kwargs",
                )
                sampling_cfg_now = update_cfg_dict(
                    sampling_kwargs,
                    update_info_dict=dict(return_inter_dict=True),
                    cfg_name="sampling_kwargs",
                )

                batch_dummy = batch_to_same_firstimage(batch)
                samples, inter_dict = pl_module.sampling_progressive(
                    batch_size=len(batch_dummy["image"]),
                    batch_data=batch_dummy,
                    condition_kwargs=condition_cfg_now,
                    sampling_kwargs=sampling_cfg_now,
                )
                _key = f"native_samecondition_{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}"
                log = self.log_sample_and_prog(
                    log=log,
                    _key=_key,
                    samples=samples,
                    inter_dict=inter_dict,
                    prog_vis_num=prog_vis_num,
                    need_to_upsample256=need_to_upsample256(
                        pl_module.hparams.data.name
                    ),
                )
                log[_key] = (
                    log["inputs"][0:1] + log[_key]
                )  # append first grouth truth image to better understand

        if is_need_to_run_interp_condition(pl_module):
            with pl_module.ema_scope("Native Sampling based on condition_interp"):
                _INTERP_NUM = 9
                condition_cfg_now = update_cfg_dict(
                    condition_kwargs,
                    update_info_dict=dict(
                        cond_scale=pl_module.hparams.cond_scale),
                    cfg_name="condition_kwargs",
                )
                sampling_cfg_now = update_cfg_dict(
                    sampling_kwargs,
                    update_info_dict=dict(return_inter_dict=True),
                    cfg_name="sampling_kwargs",
                )
                denoise_sample_fn_kwargs = prepare_denoise_fn_kwargs_4sampling(
                    pl_module=pl_module,
                    batch_data=batch,
                    sampling_kwargs=sampling_cfg_now,
                    cond_scale=condition_cfg_now["cond_scale"],
                )
                denoise_sample_fn_kwargs = batch_to_conditioninterp(
                    denoise_sample_fn_kwargs, interp=_INTERP_NUM
                )
                samples, inter_dict = pl_module.sampling_progressive(
                    batch_size=len(denoise_sample_fn_kwargs["cond"]),
                    batch_data=batch,
                    condition_kwargs=condition_cfg_now,
                    sampling_kwargs=sampling_cfg_now,
                    denoise_sample_fn_kwargs=denoise_sample_fn_kwargs,
                )
                log = self.log_sample_and_prog(
                    log=log,
                    _key=f"native_interp{_INTERP_NUM}_{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
                    samples=samples,
                    inter_dict=inter_dict,
                    prog_vis_num=prog_vis_num,
                    need_to_upsample256=need_to_upsample256(
                        pl_module.hparams.data.name
                    ),
                )

        #########################
        wandb_dict = dict()
        for k in list(log.keys()):
            if isinstance(
                log[k], list
            ):  # already constructed a list of wandb.Image, so directly use it here.
                wandb_dict[k] = log[k]
            elif isinstance(log[k], wandb.Image):
                wandb_dict[k] = log[k]
            else:
                wandb_dict[k] = log[k]

        wandb_dict = {f"imagelogger/{k}": v for k, v in wandb_dict.items()}
        pl_module.logger.experiment.log(wandb_dict)

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_frequency) == 0) and (
            check_idx > 0 or self.log_first_step
        ):
            return True
        else:
            return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
