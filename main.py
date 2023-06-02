import os
import sys
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.model_summary import summarize
import wandb
from diffusion_utils.util import instantiate_from_config

torch.set_num_threads(20)  


@hydra.main(config_path="config", config_name="config_base", version_base=None)
def run(cfg):

    return run_without_decorator(cfg)


def run_without_decorator(cfg, run_unittest=False):
    log_dir = cfg.log_dir
    
    if not os.path.exists(log_dir):
        logger.warning("making dir..")
        os.makedirs(log_dir)

    logger.warning("turn off cuda tf32")
    torch.backends.cuda.matmul.allow_tf32 = False

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `pl_datamodule.dm.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    seed_everything(cfg.seed)

    trainer_kwargs = dict(cfg.pl.trainer)

    trainer_kwargs["max_epochs"] = trainer_kwargs["max_epochs"] + 1
    logger.warning(
        "add onex more epoch for rounding error in evaluation of FID.")

    if run_unittest:
        if False:
            trainer_kwargs["max_epochs"] = 5000
            cfg.data.val_fid_num = 50000
            cfg.data.test_fid_num = 50000
            cfg.debug = False
        else:
            cfg.data.val_fid_num = 5
            cfg.data.test_fid_num = 5
            trainer_kwargs["max_epochs"] = 5
            trainer_kwargs["limit_train_batches"] = 32
            # should be larger than 25,
            trainer_kwargs["limit_val_batches"] = 30
            cfg.pl.callbacks.image_logger.params.batch_frequency = 100
            cfg.data.params.batch_size = 16  # at least 16
            cfg.data.fid_every_n_epoch = 1

    elif cfg.debug:
        cfg.data.val_fid_num = 5
        cfg.data.test_fid_num = 5
        trainer_kwargs["max_epochs"] = 3
        # trainer_kwargs['fast_dev_run'] = True
        trainer_kwargs["limit_train_batches"] = 32
        trainer_kwargs["limit_val_batches"] = 30  # should be larger than 25,
        cfg.pl.callbacks.image_logger.params.batch_frequency = 20
        cfg.data.params.batch_size = 4  # at least 16
        cfg.data.fid_every_n_epoch = 1

    model = instantiate_from_config(dict(cfg.sg))
    logger.info("callbacks")
    logger.info(cfg.pl.callbacks)
    logger.info("*" * 30)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(v) for _, v in cfg.pl.callbacks.items()
    ]

    wandb.finish()
    wandb.init(
        **cfg.wandb,
        config=omegaconf.OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=False,
        ),
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_logger = WandbLogger()
    # wandb_logger.watch(model)

    trainer = Trainer(logger=wandb_logger, **trainer_kwargs)
    trainer.logdir = log_dir

    data = instantiate_from_config(cfg.data)

    accumulate_grad_batches = getattr(
        cfg.pl.trainer, "accumulate_grad_batches", 1)

    logger.info(f"{accumulate_grad_batches}")

    wandb.run.summary["ckpt_path"] = str(Path(cfg.ckpt_dir))

    model_summary = summarize(model, max_depth=8)
    total_parameters = model_summary.total_parameters
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size

    wandb.run.summary["trainable_parameters"] = trainable_parameters
    wandb.run.summary["cpu_count"] = os.cpu_count()
    _model_info_prefix = "modelsize"
    log_dict = {
        f"{_model_info_prefix}/trainable_parameters": trainable_parameters,
        f"{_model_info_prefix}/total_parameters": total_parameters,
        f"{_model_info_prefix}/model_size": model_size,
    }
    wandb.log(log_dict, step=0)

    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        assert ckpt_path.exists(), f"{ckpt_path} does not exist"
        logger.warning("*" * 30)
        logger.warning(f"resume from {ckpt_path}")
        logger.warning("*" * 30)
    else:
        ckpt_path = None

    # run
    if cfg.train:
        trainer.fit(model, data, ckpt_path=ckpt_path)
    if not trainer.interrupted:
        trainer.test(model, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    run()
