from asyncio.log import logger
from eval.test_exps.common_stuff import (
    sample_and_get_fid,
    update_cfg_and_print,
)


def test_oracle(
    pl_module,
    condition_kwargs,
    sampling_kwargs,
    fid_kwargs,
    log_dict,
    log_immediately,
    wandb_rootdir,
    debug,
):
    condition_kwargs_current = update_cfg_and_print(
        condition_kwargs, dict(cond_scale=0), cfg_name="condition_kwargs"
    )
    sampling_kwargs_current = update_cfg_and_print(
        sampling_kwargs, dict(sampling_method="directimage"), cfg_name="sampling_kwargs"
    )
    fid_kwargs_current = update_cfg_and_print(
        fid_kwargs,
        dict(
            dl_sample=pl_module.trainer.datamodule.train_dataloader(),
            fid_num=50_0 if debug else 50_000,
            vis_knn=False,
        ),
        cfg_name="fid_kwargs",
    )
    eval_dict, fid_for_ckpt = sample_and_get_fid(
        pl_module=pl_module,
        prefix=f"{wandb_rootdir}/oracle",
        condition_cfg=condition_kwargs_current,
        fid_cfg=fid_kwargs_current,
        sampling_cfg=sampling_kwargs_current,
        debug=debug,
    )
    logger.warning(f"oracle fid = {fid_for_ckpt}")
    log_dict.update(eval_dict)
    if log_immediately:
        pl_module.logger.experiment.log(eval_dict, commit=True)
