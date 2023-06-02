import torch
import numpy as np


def prepare_condition_kwargs(pl_module, batch_data):
    cond_dim = pl_module.hparams.cond_dim
    condition_method = pl_module.hparams.condition_method

    ######################################################

    if condition_method is not None:
        assert pl_module.hparams.cond_drop_prob > 0
        cond_drop_prob = pl_module.hparams.cond_drop_prob if pl_module.training else 1.0
    else:  # unconditional training
        cond_drop_prob = 1.0
    result_dict = dict(cond_drop_prob=cond_drop_prob)

    if condition_method is None:
        result_dict.update(cond=None)

    elif condition_method in [
        "label",
        "attr",
        "feat",
        "knn_feat",
        "patchfeat",
        "centroid",
        "labelcentroid",
        "cluster",
        "clustermix",
        "clusterrandom",

        "labelcluster",
        "patchcluster",
    ]:
        result_dict.update(cond=batch_data[condition_method])

    elif condition_method in ['cluster_lookup']:
        result_dict.update(cond=None, image_batch_ids=batch_data['id'])

    elif condition_method in ["clusterlayout"]:
        _how = pl_module.hparams.condition.clusterlayout.how

        if _how == "lost":
            result_dict.update(
                cond=batch_data["cluster"].float().to(pl_module.device),
                layout=batch_data["lostbboxmask"].float().to(pl_module.device),
            )
        elif _how == "oracle":
            result_dict.update(
                cond=batch_data["cluster"].float().to(pl_module.device),
                layout=batch_data["segmask"].float().to(pl_module.device),
            )
        elif _how == "stego":
            result_dict.update(
                cond=batch_data["cluster"].float().to(pl_module.device),
                layout=batch_data["stegomask"].float().to(pl_module.device),
            )
        else:
            raise
    elif condition_method in ["layout"]:
        _how = pl_module.hparams.condition.layout.how

        if _how == "lost":
            result_dict.update(
                layout=batch_data["lostbboxmask"].float().to(pl_module.device),
            )
        elif _how == "oracle":
            result_dict.update(
                layout=batch_data["segmask"].float().to(pl_module.device),
            )
        elif _how == "stego":
            result_dict.update(
                layout=batch_data["stegomask"].float().to(pl_module.device),
            )
        else:
            raise
    elif condition_method in ["stegoclusterlayout"]:
        result_dict.update(
            cond=batch_data["stego_attr"].float().to(pl_module.device),
            layout=batch_data["stegomask"].float().to(pl_module.device),
        )
    else:
        raise ValueError(condition_method)

    return result_dict


def prepare_denoise_fn_kwargs_4sharestep(pl_module, batch_data):
    denoise_fn_kwargs = prepare_condition_kwargs(
        pl_module=pl_module, batch_data=batch_data
    )
    return denoise_fn_kwargs


def randomsample_cond(pl_module, data_dict, random_sample_condition):
    cond_dim = pl_module.hparams.cond_dim
    condition_method = pl_module.hparams.condition_method

    if condition_method is None:
        if random_sample_condition:
            raise

    elif condition_method in ["label"]:
        if random_sample_condition:
            data_dict["label"] = data_dict["label_random"]

    elif condition_method in ["cluster"]:
        if random_sample_condition:
            data_dict["cluster"] = data_dict["cluster_random"]

    elif condition_method in ["centroid"]:
        if random_sample_condition:
            data_dict["centroid"] = data_dict["centroid_random"]

    elif condition_method in ["knn_feat"]:
        if random_sample_condition:
            data_dict["knn_feat"] = data_dict["knn_feat_random"]

    elif condition_method in [
        "feat",
        "attr",
        "labelcluster",
        "labelcentroid",
        "clusterlayout",
        "stegoclusterlayout",
        "clustermix",
        "clusterrandom",
        "layout",
        "patchcluster",
        "patchfeat",
    ]:
        if random_sample_condition:
            raise
    else:
        raise ValueError(condition_method)

    return data_dict


def prepare_denoise_fn_kwargs_4sampling(
    pl_module, batch_data, sampling_kwargs, cond_scale
):
    batch_data = randomsample_cond(
        pl_module,
        data_dict=batch_data,
        random_sample_condition=sampling_kwargs["random_sample_condition"],
    )

    denoise_sample_fn_kwargs = prepare_denoise_fn_kwargs_4sharestep(
        pl_module, batch_data
    )
    denoise_sample_fn_kwargs.update(dict(cond_scale=cond_scale))  # override
    # don't need it in testing mode. remove it to avoid bug
    denoise_sample_fn_kwargs.pop("cond_drop_prob")

    return denoise_sample_fn_kwargs
