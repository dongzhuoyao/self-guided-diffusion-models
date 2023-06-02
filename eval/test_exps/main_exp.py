from eval.test_exps.common_stuff import (
    get_condition_scale_list,
    get_condition_scale_main,
    sample_and_get_fid,
    sampling_cond_str,
    update_cfg_and_print,
)
from loguru import logger


def run_cond_scale_single(
    pl_module,
    condition_kwargs,
    sampling_kwargs,
    fid_kwargs,
    log_dict,
    log_immediately,
    wandb_rootdir,
    debug,
    cond_scale,
    vis_knn=False,
):
    condition_cfg_now = update_cfg_and_print(
        condition_kwargs, dict(cond_scale=cond_scale), cfg_name="condition_kwargs"
    )
    sampling_cfg_now = update_cfg_and_print(
        sampling_kwargs, dict(), cfg_name="sampling_kwargs"
    )
    fid_cfg_now = update_cfg_and_print(  # always sampling from train_dataloader for fid
        fid_kwargs,
        dict(
            dl_sample=pl_module.trainer.datamodule.train_dataloader(),
            # dl_sample=pl_module.trainer.datamodule.val_dataloader(),  # hutao
            vis_knn=vis_knn,
        ),
        cfg_name="fid_kwargs",
    )
    #logger.warning('use val_dataloader for fid')
    eval_dict, fid_for_ckpt = sample_and_get_fid(
        pl_module=pl_module,
        prefix=f"{wandb_rootdir}/{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
        condition_cfg=condition_cfg_now,
        fid_cfg=fid_cfg_now,
        sampling_cfg=sampling_cfg_now,
        debug=debug,
    )
    log_dict.update(eval_dict)

    if log_immediately:
        pl_module.logger.experiment.log(eval_dict, commit=True)
    return fid_for_ckpt


def main_cond_scale_4val(pl_module, **kwargs):
    fid_for_ckpt = run_cond_scale_single(
        cond_scale=get_condition_scale_main(pl_module),
        pl_module=pl_module,
        vis_knn=False,
        **kwargs,
    )
    return fid_for_ckpt


def main_cond_scale_4test(pl_module, **kwargs):
    if pl_module.hparams.condition_method is None:
        cond_scale_list = [0]
    else:
        cond_scale_list = get_condition_scale_list(pl_module)

    for idx, cond_scale in enumerate(cond_scale_list):
        fid_for_ckpt = run_cond_scale_single(
            cond_scale=cond_scale, pl_module=pl_module, vis_knn=False, **kwargs
        )
    return fid_for_ckpt


def main_cond_scale_ablation(pl_module, **kwargs):

    cond_scale_list = pl_module.hparams.exp.ablate_scale_list
    logger.warning('ablate_scale_list: {}'.format(cond_scale_list))
    if pl_module.hparams.condition_method is None:
        logger.warning("condition_method is None, set cond_scale_list to [0]")
        cond_scale_list = [0]
    else:
        if False:  # default
            cond_scale_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,
                               1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
        elif False:  # debugging
            cond_scale_list = [i for i in range(16)]
            cond_scale_list.reverse()

    for idx, cond_scale in enumerate(cond_scale_list):
        fid_for_ckpt = run_cond_scale_single(
            cond_scale=cond_scale, pl_module=pl_module, vis_knn=(idx == 0), **kwargs
        )
    return fid_for_ckpt
