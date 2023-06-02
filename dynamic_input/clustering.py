import os
from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from dataset.ds_utils.dataset_common_utils import ds_has_label_info

from diffusion_utils.taokit.wandb_utils import wandb_scatter_fig
from diffusion_utils.util import clip_unnormalize_to_zero_to_255


def knn_vis(pl_module, _dl):
    logger.warning("knn_vis......")
    VIS_EXAMPLE_NUM = min(16, _dl.batch_size)
    MAX_IMG_PER_KLUSTER = 16
    cluster_dict = {k: list() for k in range(VIS_EXAMPLE_NUM)}
    batch = next(iter(_dl))
    batch_nns = batch["nns"].cpu().numpy()
    for i in range(VIS_EXAMPLE_NUM):
        cur_img = clip_unnormalize_to_zero_to_255(batch["image"][i])
        cluster_dict[i].append(wandb.Image(cur_img.float()))
        for j in range(len(batch_nns[i])):
            image_i_j = clip_unnormalize_to_zero_to_255(
                _dl.dataset.__getitem__(batch_nns[i, j])["image"]
            )
            if j < VIS_EXAMPLE_NUM and len(cluster_dict[i]) < MAX_IMG_PER_KLUSTER:
                cluster_dict[i].append(wandb.Image(image_i_j.float()))

    wandb_dict = dict()
    for i in range(VIS_EXAMPLE_NUM):
        wandb_dict.update({f"nns_vis/image{i}": cluster_dict[i]})
    pl_module.logger.experiment.log(wandb_dict, commit=True)


def kmeans_vis(pl_module, _dl, cluster_ids=None):
    if pl_module.hparams.exp.kmeans_vis:
        CLUSTER_NUM_VIS = 16
        MAX_IMG_PER_KLUSTER = 16
        cluster_dict = {k: list() for k in range(CLUSTER_NUM_VIS)}
        if cluster_ids is not None:  # vis all
            cluster_dict_4papervis = {k_id: list() for k_id in cluster_ids}
        for _, batch in tqdm(
            enumerate(_dl), total=len(_dl), desc=f"Visualization Only, kmeans",
        ):
            kluster_ids = list(batch["cluster_id"].cpu().numpy())
            batch_image = clip_unnormalize_to_zero_to_255(batch["image"])
            for batch_id, kluster_id in enumerate(kluster_ids):
                if (
                    kluster_id < CLUSTER_NUM_VIS
                    and len(cluster_dict[kluster_id]) < MAX_IMG_PER_KLUSTER
                ):
                    cluster_dict[kluster_id].append(batch_image[batch_id])
                if cluster_ids is not None and kluster_id in cluster_ids:
                    cluster_dict_4papervis[kluster_id].append(
                        batch_image[batch_id])

        if cluster_ids is None:  # vis all
            wandb_dict = dict()
            for i in range(CLUSTER_NUM_VIS):
                cluster_dict[i] = [
                    wandb.Image(_img.float()) for _img in cluster_dict[i]
                ]
                wandb_dict.update({f"kmeans_vis/cluster{i}": cluster_dict[i]})
            pl_module.logger.experiment.log(wandb_dict, commit=True)
        else:  # used for papervis
            return cluster_dict_4papervis


def np_hist_to_wandb_scatter(cluster_hist_np, dict_key):
    hist, bin_edges = cluster_hist_np
    hist = hist.tolist()
    bin_edges = [int(i) for i in bin_edges.tolist()][: len(hist)]
    _dict = wandb_scatter_fig(x_list=bin_edges, y_list=hist, dict_key=dict_key)
    return _dict


def wandb_log_cluster_statistics(pl_module):
    from dataset.ds_utils.unsupervised_cluster import cal_cluster_statistics

    if pl_module.training:
        logger.warning("wandb_log_cluster_statistics...")
        _dataset_train = pl_module.trainer.datamodule.train_dataloader().dataset
        _dataset_val = pl_module.trainer.datamodule.val_dataloader().dataset
        ######################
        log_dict = dict()
        for _name, _dataset in [("train", _dataset_train), ("val", _dataset_val)]:
            cluster_eval_dict = cal_cluster_statistics(
                _dataset, nmi=True, ami=True, ari=True
            )
            cluster_eval_dict = {
                f"cluster/{_name}_{k}": v for k, v in cluster_eval_dict.items()
            }
            log_dict.update(cluster_eval_dict)
            if hasattr(_dataset, "cluster_hist"):
                log_dict.update(
                    np_hist_to_wandb_scatter(
                        _dataset.cluster_hist, dict_key=f"cluster/{_name}_cluster_hist"
                    )
                )
            if hasattr(_dataset, "class_hist"):
                log_dict.update(
                    np_hist_to_wandb_scatter(
                        _dataset.class_hist, dict_key=f"cluster/{_name}_class_hist"
                    )
                )
        pl_module.logger.experiment.log(log_dict, commit=True)


def vis_cluster_relatedstuff(pl_module):
    condition_method = pl_module.hparams.condition_method
    _train_dl = pl_module.trainer.datamodule.train_dataloader()

    if not pl_module.hparams.debug:
        # clusterlayout
        if condition_method in [
            "cluster",
            "centroid",
            "labelcluster",
            "labelcentroid",
        ]:
            if ds_has_label_info(pl_module.hparams.data.name):
                wandb_log_cluster_statistics(pl_module)

            kmeans_vis(pl_module, _dl=_train_dl)
            logger.info("KMeans done.")

        elif condition_method in ["knn_feat"]:
            knn_vis(pl_module, _dl=_train_dl)

        else:
            logger.warning("no kmeans necessary")


def prepare_cluster(pl_module, batch_data):
    ##################### Assign Kmeans label #####################
    condition_method = pl_module.hparams.condition_method
    if pl_module.hparams.condition.cluster.random:
        if condition_method in ["cluster"]:
            batch_data["cluster"] = batch_data["cluster_random"]

        elif condition_method in ["centroid"]:
            batch_data["centroid"] = batch_data["centroid_random"]

    return batch_data
