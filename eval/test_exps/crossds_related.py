def crossds_test(debug):
    # cross-set evaluation
    if (not debug) and False:  # disable it now
        if (
            pl_module.hparams.condition_method is None
            or pl_module.hparams.condition_method in ["label"]
        ):
            logger.warning("skip cross-set, cross-dataset experiment")
        elif pl_module.hparams.condition_method in ["feat", "unsupervised_auglevel"]:
            for desc, dl_gt, dl_sample in [
                (
                    "gt_train_sample_train",
                    pl_module.dl_train,
                    pl_module.dl_train,
                ),
                (
                    "gt_train_sample_val",
                    pl_module.dl_train,
                    pl_module.dl_val,

                ),
                (
                    "gt_val_sample_train",
                    pl_module.dl_val,
                    pl_module.dl_train,

                ),
                ("gt_val_sample_val", pl_module.dl_val,
                 pl_module.dl_val, "val", "val"),
                (
                    "gt_val_sample_crossds",
                    pl_module.dl_val,
                    pl_module.dl_crossds,

                ),
            ]:
                for sampling_method in [
                    "directimage",
                    sampling_kwargs["sampling_method"],
                ]:
                    condition_cfg_now = update_cfg_and_print(
                        condition_kwargs,
                        dict(cond_scale=pl_module.hparams.cond_scale),
                        cfg_name="condition_kwargs",
                    )
                    sampling_cfg_now = update_cfg_and_print(
                        sampling_kwargs,
                        dict(sampling_method=sampling_method),
                        cfg_name="sampling_kwargs",
                    )
                    fid_cfg_now = update_cfg_and_print(
                        fid_kwargs,
                        dict(
                            dl_gt=dl_gt,
                            dl_sample=dl_sample,
                        ),
                        cfg_name="fid_kwargs",
                    )
                    eval_dict, _ = sample_and_get_fid(
                        pl_module=pl_module,
                        prefix=f"{wandb_rootdir}/{{cond_str(condition_kwargs_current)}}_{desc}",
                        condition_cfg=condition_cfg_now,
                        fid_cfg=fid_cfg_now,
                        sampling_cfg=sampling_cfg_now,
                        debug=debug,
                    )
                    log_dict.update(eval_dict)
                    if log_immediately:
                        pl_module.logger.experiment.log(eval_dict, commit=True)
