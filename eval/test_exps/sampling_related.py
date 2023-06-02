from eval.test_exps.common_stuff import (
    get_condition_scale_main,
    sample_and_get_fid,
    sampling_cond_str,
    update_cfg_and_print,
)


def random_sample(
    pl_module,
    condition_kwargs,
    sampling_kwargs,
    fid_kwargs,
    log_dict,
    log_immediately,
    wandb_rootdir,
    debug,
):
    # Evaluate for randomsample
    if pl_module.hparams.condition_method in ["label", "cluster", "centroid"]:
        for cond_scale in [2.0]:
            condition_cfg_now = update_cfg_and_print(
                condition_kwargs,
                dict(cond_scale=cond_scale),
                cfg_name="condition_kwargs",
            )
            sampling_cfg_now = update_cfg_and_print(
                sampling_kwargs,
                dict(random_sample_condition=True),
                cfg_name="sampling_kwargs",
            )
            fid_cfg_now = update_cfg_and_print(
                fid_kwargs,
                dict(
                    dl_sample=pl_module.trainer.datamodule.train_dataloader(),
                ),
                cfg_name="fid_kwargs",
            )
            eval_dict, _ = sample_and_get_fid(
                pl_module=pl_module,
                prefix=f"{wandb_rootdir}/randomsample_{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
                condition_cfg=condition_cfg_now,
                fid_cfg=fid_cfg_now,
                sampling_cfg=sampling_cfg_now,
                debug=debug,
            )
            log_dict.update(eval_dict)
            if log_immediately:
                pl_module.logger.experiment.log(eval_dict, commit=True)


def condmix_4test(
    pl_module,
    condition_kwargs,
    sampling_kwargs,
    fid_kwargs,
    log_dict,
    log_immediately,
    wandb_rootdir,
    debug,
):

    condition_cfg_now = update_cfg_and_print(
        condition_kwargs,
        dict(cond_scale=get_condition_scale_main(pl_module)),
        cfg_name="condition_kwargs",
    )
    sampling_cfg_now = update_cfg_and_print(
        sampling_kwargs, dict(), cfg_name="sampling_kwargs"
    )
    fid_cfg_now = update_cfg_and_print(
        fid_kwargs,
        dict(
            dl_sample=pl_module.trainer.datamodule.train_dataloader(),
        ),
        cfg_name="fid_kwargs",
    )
    eval_dict, _ = sample_and_get_fid(
        pl_module=pl_module,
        prefix=f"{wandb_rootdir}/condmix_{sampling_cond_str(sampling_cfg_now,condition_cfg_now)}",
        condition_cfg=condition_cfg_now,
        fid_cfg=fid_cfg_now,
        sampling_cfg=sampling_cfg_now,
        debug=debug,
    )
    log_dict.update(eval_dict)
    if log_immediately:
        pl_module.logger.experiment.log(eval_dict, commit=True)
