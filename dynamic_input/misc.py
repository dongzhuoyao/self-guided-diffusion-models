import os
from pathlib import Path
from loguru import logger
from diffusion_utils.util import get_obj_from_str



def assert_image_dir(pl_module):
    if pl_module.training:
        condition_kwargs, sampling_kwargs, fid_kwargs = pl_module.get_default_config()
        train_dir, val_dir, fid_debug_dir = (
            fid_kwargs["fid_train_image_dir"],
            fid_kwargs["fid_val_image_dir"],
            fid_kwargs["fid_debug_dir"],
        )
        logger.warning(
            f"fid_train_image_dir: {train_dir}, image_num={len(os.listdir(train_dir))}"
        )
        logger.warning(
            f"fid_val_image_dir: {val_dir}, image_num={len(os.listdir(val_dir))}"
        )
        logger.warning(
            f"fid_debug_dir: {fid_debug_dir}, image_num={len(os.listdir(fid_debug_dir))}"
        )

        assert os.path.exists(train_dir)
        assert os.path.exists(val_dir)
        assert os.path.exists(fid_debug_dir)


def assert_check(pl_module):
    condition_method = pl_module.hparams.condition_method
    ################################

    assert pl_module.hparams.parameterization in [
        "eps",
        "x0",
    ], 'currently only supporting "eps" and "x0"'
    logger.info(
        f"{pl_module.__class__.__name__}: Running in {pl_module.hparams.parameterization}-prediction mode"
    )
    ########################################################################

    if condition_method is None:
        assert pl_module.hparams.cond_dim == 0
        assert pl_module.hparams.cond_scale == 0
        assert pl_module.hparams.cond_drop_prob == 1

    elif condition_method in ["feat", "patchfeat"]:
        assert pl_module.hparams.condition.feat.feat_from is not None
        assert pl_module.hparams.data.h5_file is not None
        assert (
            pl_module.hparams.condition.feat.feat_from in pl_module.hparams.data.h5_file
        ), f"h5_file {pl_module.hparams.data.h5_file} should includes feature name. {pl_module.hparams.condition.feat.feat_from}"

    elif condition_method in ["label"]:
        if False:#dont assert now, as we have subgroup now
            assert (
                pl_module.hparams.cond_dim == pl_module.hparams.data.num_classes
            ), "{}, {}".format(
                pl_module.hparams.cond_dim, pl_module.hparams.data.num_classes
            )

    elif condition_method in ["attr", "stegoclusterlayout"]:
        pass

    elif condition_method in [
        "labelcluster",
        "cluster",
        "cluster_lookup",
        "clusterrandom",
        "clustermix",
        "centroid",
        "patchcluster",
        "labelcentroid",
        "clusterlayout",
    ]:
        assert pl_module.hparams.data.h5_file is not None

    elif condition_method in ['layout']:
        assert pl_module.hparams.data.h5_file is None

    elif condition_method in ["knn_feat"]:
        assert pl_module.hparams.data.h5_file is not None

    else:
        raise ValueError(condition_method)

    if pl_module.hparams.data.h5_file is not None:
        logger.warning(
            f"reading info from h5 file {pl_module.hparams.data.h5_file}")


def get_default_config(pl_module):

    if False:  # remove later
        fid_train_image_dir = str(
            get_obj_from_str(pl_module.hparams.data.fid_train_image_dir)
        )
        fid_val_image_dir = str(get_obj_from_str(
            pl_module.hparams.data.fid_val_image_dir))
        fid_debug_dir = str(get_obj_from_str(
            pl_module.hparams.data.fid_debug_dir))
    else:
        fid_train_image_dir = str(
            Path(pl_module.hparams.data.fid_train_image_dir).expanduser().resolve())
        fid_val_image_dir = str(Path(
            pl_module.hparams.data.fid_val_image_dir).expanduser().resolve())
        fid_debug_dir = str(
            Path(pl_module.hparams.data.fid_debug_dir).expanduser().resolve())

    condition_kwargs = dict(
        cond_scale=pl_module.hparams.cond_scale,
        condition_method=pl_module.hparams.condition_method,
    )
    fid_kwargs = dict(
        fid_num=None,
        vis_knn=False,
        fid_train_image_dir=fid_train_image_dir,
        fid_val_image_dir=fid_val_image_dir,
        fid_debug_dir=fid_debug_dir,
        sample_dir="sample",
        save_dir=None,
        vis=pl_module.hparams.vis,
        dataset_name=pl_module.hparams.data.name,
        image_size=pl_module.hparams.data.image_size,
    )
    sampling_kwargs = dict(
        sampling_method=pl_module.hparams.model.sampling,
        vis=pl_module.hparams.vis,
        num_timesteps=pl_module.hparams.model.num_timesteps,
        ddim_eta=pl_module.hparams.ddim_eta,
        log_num_per_prog=pl_module.hparams.log_num_per_prog,
        clip_denoised=pl_module.hparams.model.clip_denoised,
        dtp=pl_module.hparams.dtp,
        temperature=1.0,
        noise_dropout=0,
        random_sample_condition=False,
        return_inter_dict=False,
        disable_tqdm=True,
    )

    return condition_kwargs, sampling_kwargs, fid_kwargs


def log_range(pl_module, data_dict, commit=False):
    log_dict = dict()
    for k, _value in data_dict.items():
        log_dict[f"range/max_{k}"] = _value.max().item()
        log_dict[f"range/mean_{k}"] = _value.float().mean().item()
        log_dict[f"range/min_{k}"] = _value.float().min().item()
        log_dict[f"range/std_{k}"] = (
            _value.float().std().item()
        )  # output the std of the data  to compare with the noise
    pl_module.logger.experiment.log(log_dict, commit=commit)


def is_need_to_run_interp_condition(pl_module):
    if pl_module.hparams.condition_method in [
        "label",
        "feat",
        "cluster",
        "centroid",
        "labelcluster",
    ]:
        return True
    else:
        return False
