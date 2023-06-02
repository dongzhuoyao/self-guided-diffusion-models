import imp
import os
from queue import Queue
import random
from functools import partial
from pathlib import Path
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from contextlib import contextmanager
from diffusion_utils.taokit.pl_utils import FIDMetrics
from dynamic_input.misc import (
    assert_check,
    assert_image_dir,
    get_default_config,
    log_range,
)
from dynamic_input.condition import (
    prepare_denoise_fn_kwargs_4sharestep,
    prepare_denoise_fn_kwargs_4sampling,
)

from eval.run_exp import run_test_and_all_exploration, run_validation
from diffusion_utils.util import (
    exists,
    default,
    mean_flat,
    count_params,
    instantiate_from_config,
    tensor_dict_copy,
)
from dynamic.ema import LitEma
from dynamic_input.clustering import prepare_cluster, vis_cluster_relatedstuff
from dynamic_input.feat import prepare_feat
from dynamic_input.image import prepare_image


from loguru import logger
import matplotlib.pyplot as plt
from diffusion_utils.taokit.wandb_utils import wandb_scatter_fig
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F


from tqdm import tqdm
from einops import rearrange, repeat, reduce

from lightning_module_common import configure_optimizers, print_best_path


class TaoDiffusion(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = instantiate_from_config(self.hparams.dynamic).to(
            self.hparams.device
        )
        count_params(self.model, verbose=True)
        if self.hparams.use_ema:
            self.model_ema = LitEma(self.model)
            logger.info(
                f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.diffusion = instantiate_from_config(self.hparams.diffusion_model)

        self.diffusion.set_denoise_fn(
            self.model.forward, self.model.forward_with_cond_scale
        )

        self.fid_metric = FIDMetrics(prefix='fidmetric_eval_val_v2')
        ##############################
        assert_check(pl_module=self)
        self.min_fid_for_ckpt = 1e10
        self.ckpt_path_has_run_first_time = False

    def get_default_config(
        self,
    ):
        condition_kwargs, sampling_kwargs, fid_kwargs = get_default_config(
            pl_module=self
        )
        return condition_kwargs, sampling_kwargs, fid_kwargs

    @contextmanager
    def ema_scope(self, context=None):
        if self.hparams.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.hparams.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    @torch.no_grad()
    def log_images(self, batch, subset):
        batch = self.prepare_batch(batch, subset)
        return batch

    def configure_optimizers(self):
        opt = configure_optimizers(pl_module=self)
        return opt

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert_image_dir(pl_module=self)
            vis_cluster_relatedstuff(pl_module=self)
            self.logger.experiment.log(
                self.diffusion.vis_schedule(), commit=False)

    def prepare_batch(
        self,
        batch_data,
        subset,
    ):
        batch_data = prepare_image(pl_module=self, batch_data=batch_data)
        batch_data = prepare_feat(pl_module=self, batch_data=batch_data)
        batch_data = prepare_cluster(
            pl_module=self, batch_data=batch_data
        )
        return batch_data

    @torch.no_grad()
    def sampling_progressive(
        self,
        batch_size,
        batch_data=None,
        sampling_kwargs=None,
        condition_kwargs=None,
        denoise_sample_fn_kwargs=None,
        **kwargs,
    ):
        _shape = (
            batch_size,
            self.hparams.data.channels,
            self.hparams.data.image_size,
            self.hparams.data.image_size,
        )
        if denoise_sample_fn_kwargs is None:
            denoise_sample_fn_kwargs = prepare_denoise_fn_kwargs_4sampling(
                pl_module=self,
                batch_data=batch_data,
                sampling_kwargs=sampling_kwargs,
                cond_scale=condition_kwargs["cond_scale"],
            )
        #############################

        result = self.diffusion.p_sample_loop(
            sampling_method=sampling_kwargs["sampling_method"],
            shape=_shape,
            denoise_sample_fn_kwargs=denoise_sample_fn_kwargs,
            sampling_kwargs=sampling_kwargs,
            condition_kwargs=condition_kwargs,
            **kwargs,
        )
        samples, intermediate_dict = result
        if sampling_kwargs["return_inter_dict"]:
            return samples, intermediate_dict
        else:
            return samples, intermediate_dict["pred_x0"]
            # return samples, intermediate_dict["x_inter"]

    @torch.no_grad()
    def sampling(self, **kwargs):
        final, inter = self.sampling_progressive(**kwargs)
        return final

    def get_loss(self, pred, target, mean=True):
        if self.hparams.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.hparams.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(
                    target, pred, reduction="none")
        elif self.hparams.loss_type == "huber":
            if mean:
                loss = torch.nn.functional.smooth_l1_loss(target, pred)
            else:
                loss = torch.nn.functional.smooth_l1_loss(
                    target, pred, reduction="none"
                )
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def shared_step(self, batch_data, subset):
        batch_data = self.prepare_batch(batch_data, subset=subset)
        log_range(self, batch_data, commit=False)
        denoise_fn_kwargs = prepare_denoise_fn_kwargs_4sharestep(
            pl_module=self, batch_data=batch_data
        )
        loss, loss_dict = self.diffusion.forward_tao(
            x=batch_data["image"], **denoise_fn_kwargs
        )
        return loss, loss_dict

    def training_step(self, batch_data, batch_idx):
        loss, loss_dict = self.shared_step(batch_data, subset="train")
        if batch_idx > 0:  # more elegant ?
            self.iters_per_sec = 1.0 / (time.time() - self.last_time)
            loss_dict.update(dict(iters_per_sec=self.iters_per_sec))
            self.last_time = time.time()

        if batch_idx == 0:
            self.last_time = time.time()
            self.epoch_stats_x = []
            self.epoch_stats_y = []

        if "train/epoch_stats_x" in loss_dict and "train/epoch_stats_y" in loss_dict:
            self.epoch_stats_x.append(loss_dict.pop("train/epoch_stats_x"))
            self.epoch_stats_y.append(loss_dict.pop("train/epoch_stats_y"))
        loss_dict.update(
            dict(
                global_step=float(self.global_step),
                img_million=float(self.global_step *
                                  len(batch_data["image"]) / 1e6),
            )
        )
        if self.hparams.optim.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            loss_dict.update({"lr_abs": lr})

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        return loss

    def training_epoch_end(self, training_step_outputs):
        if len(self.epoch_stats_x) > 0:
            _stats_x = torch.concat(
                self.epoch_stats_x, 0).cpu().numpy().tolist()
            _stats_y = torch.concat(
                self.epoch_stats_y, 0).cpu().numpy().tolist()
            wandb_dict = wandb_scatter_fig(
                x_list=_stats_x, y_list=_stats_y, dict_key="loss_vs_time"
            )
            self.logger.experiment.log(wandb_dict, commit=False)
            self.epoch_stats_x = []
            self.epoch_stats_y = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            return  # don't evaluation when very-first batch of your training
        if (self.current_epoch % self.hparams.data.fid_every_n_epoch == 0) or (self.trainer.ckpt_path is not None and not self.ckpt_path_has_run_first_time):
            if batch_idx == 0:
                if self.current_epoch == 0:
                    val_fid_num = int(self.hparams.data.val_fid_num * 0.1)
                else:
                    val_fid_num = self.hparams.data.val_fid_num
                assert_image_dir(pl_module=self)
                if True:  # used for debugging multi-gpu training
                    self.fid_for_ckpt = run_validation(
                        self,
                        wandb_rootdir="eval_val_v2",
                        val_fid_num=val_fid_num,
                        log_immediately=True,
                    )
                else:
                    self.fid_for_ckpt = 1.0/self.current_epoch
                self.ckpt_path_has_run_first_time = True

        if batch_idx == 0:
            self.log("val/fid_for_ckpt", self.fid_for_ckpt, on_epoch=True)
            if False:
                tb_metrics = {
                    **self.fid_metric.compute(self.fid_for_ckpt),
                }
                self.log_dict(tb_metrics)
            print_best_path(self)

        _, loss_dict_no_ema = self.shared_step(
            tensor_dict_copy(batch), subset="val")
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(
                tensor_dict_copy(batch), subset="val")
            loss_dict_ema = {
                key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

    @torch.no_grad()
    def validation_epoch_end(
        self,
        val_step_outputs,
    ):
        self.fid_metric.reset()

    def on_train_batch_end(self, *args, **kwargs):
        if self.hparams.use_ema:
            self.model_ema(self.model)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            if not self.hparams.profile:
                assert_image_dir(pl_module=self)
                run_test_and_all_exploration(
                    self, wandb_rootdir="eval_test_v2", log_immediately=True
                )
