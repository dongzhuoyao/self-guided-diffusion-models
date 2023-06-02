


from loguru import logger
import torch

from diffusion_utils.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR


def print_best_path(pl_module):
    if pl_module.global_rank==0:
        best_model_score = pl_module.trainer.checkpoint_callback.best_model_score
        best_model_path = pl_module.trainer.checkpoint_callback.best_model_path
        current_score = pl_module.trainer.checkpoint_callback.current_score
        last_model_path = pl_module.trainer.checkpoint_callback.last_model_path
        logger.warning(f"best_model_path(score:{best_model_score}): {best_model_path}")
        logger.warning(f"last_model_path(score:{current_score}): {last_model_path}")

def configure_optimizers(pl_module):
    params = list(pl_module.model.parameters())
    if pl_module.hparams.optim.name=='adamw':
        opt = torch.optim.AdamW(params, lr=pl_module.hparams.optim.params.lr, weight_decay=pl_module.hparams.optim.params.wd)
        if pl_module.global_rank==0:logger.warning('applied adamw')
    elif pl_module.hparams.optim.name=='adam':
        opt = torch.optim.Adam(params, lr=pl_module.hparams.optim.params.lr, eps=pl_module.hparams.optim.params.eps,
            betas=(pl_module.hparams.optim.params.beta1,pl_module.hparams.optim.params.beta2),
            weight_decay=pl_module.hparams.optim.params.wd)
        if pl_module.global_rank==0:logger.warning('applied adam')
    else:
        raise ValueError(pl_module.hparams.optim.name)
    if pl_module.hparams.optim.scheduler_config is not None:
        scheduler = instantiate_from_config(pl_module.hparams.optim.scheduler_config)
        if pl_module.global_rank==0:logger.info("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler
    return opt