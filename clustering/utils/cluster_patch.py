import argparse
from datetime import datetime
from pathlib import Path
import shutil
from loguru import logger
from sklearn.metrics import normalized_mutual_info_score


import torch
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
import io
import os
import torch.nn.functional as F
import random
import h5py
from tqdm import tqdm
from clustering.cal_cluster_metric import cal_cluster_metric
from clustering.faiss_kmeans import run_kmeans, run_nns
from clustering.utils.cluster_standard import clustering
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def copy_json_file(feat_h5py_path, dst_h5py_path):
    json_name = str(feat_h5py_path).replace(".h5", ".json")
    assert os.path.exists(json_name), f"{json_name} not exists"
    new_json_name = str(dst_h5py_path).replace(".h5", ".json")
    shutil.copyfile(json_name, new_json_name)
    logger.warning(f"copy json from {json_name} to {new_json_name}")


def clustering_withpatches(
    src_feat_h5py_path, cluster_k, niter, minp, cluster_h5_root=None, debug=False
):

    src_feat_h5py_path = Path(src_feat_h5py_path).expanduser().resolve()
    f_feat = h5py.File(src_feat_h5py_path, mode="r")
    dset_feat = f_feat["all_attributes"]
    dataset_name = dset_feat.attrs["dataset_name"]
    feat_from = dset_feat.attrs["feat_from"]
    feat_dim = dset_feat.attrs["feat_dim"]

    if "resampled_size" in dset_feat.attrs.keys():
        resampled_size = dset_feat.attrs["resampled_size"]
        logger.warning(f"reading resampled_size={resampled_size}")
    else:
        resampled_size = 14
    logger.warning(dataset_name)
    logger.warning(feat_from)
    logger.warning(feat_dim)
    ####################

    if debug:
        token_num_all = 3
        cluster_k = 10
        niter = 30
    else:
        token_num_all = 1 + resampled_size * resampled_size
    ########################

    def get_feat(split):
        if debug:
            r = f_feat[split][:1000]
        else:
            r = f_feat[split]
        return r

    time_str = datetime.now().isoformat(timespec="hours")
    # v2: only use train feat now
    # v3, add id2name
    _h5py_path = f"~/data/sg_data/cluster/v3_{dataset_name}_cluster{cluster_k}_iter{niter}minp{minp}_{feat_from}_{time_str}_{sha[:7]}_withpatches_size{resampled_size}.h5"
    if debug:
        _h5py_path = _h5py_path.replace(".h5", "debug.h5")
    if cluster_h5_root is not None:
        _h5py_path = _h5py_path.replace(
            "~/data/sg_data/cluster", cluster_h5_root)
    _h5py_path = Path(_h5py_path).expanduser().resolve()

    logger.warning(_h5py_path)
    copy_json_file(feat_h5py_path=src_feat_h5py_path, dst_h5py_path=_h5py_path)
    logger.warning(f"begin keamns fit")

    feat_dim = get_feat("train").shape[-1]

    f = h5py.File(_h5py_path, mode="w")
    f.close()
    f = h5py.File(_h5py_path, mode="a")
    f.create_dataset(
        "train",
        data=np.ones(shape=(len(get_feat("train")),
                     token_num_all), dtype=np.int64)
        * -1,
    )
    f.create_dataset(
        "val",
        data=np.ones(shape=(len(get_feat("val")), token_num_all),
                     dtype=np.int64) * -1,
    )
    f.create_dataset(
        "centroids",
        data=np.ones(shape=(cluster_k, token_num_all, feat_dim)),
    )  # [cluster_num, token_num, feat_dim]

    dset = f.create_dataset("all_attributes", (1,))
    dset.attrs["dataset_name"] = dset_feat.attrs["dataset_name"]
    dset.attrs["feat_from"] = dset_feat.attrs["feat_from"]
    dset.attrs["cluster_k"] = cluster_k
    dset.attrs["feat_dim"] = dset_feat.attrs["feat_dim"]

    logger.info("copy metadata about id2name, name2id..")
    for key in dset_feat.attrs.keys():
        if "id2name" in key:
            dset.attrs[key] = dset_feat.attrs[key]

        if "name2id" in key:
            dset.attrs[key] = dset_feat.attrs[key]

    ######################################
    for token_id in tqdm(range(token_num_all)):  # except CLS token
        train_feat, val_feat = (
            get_feat("train")[:, token_id, :],
            get_feat("val")[:, token_id, :],
        )
        trainval = np.concatenate([train_feat, val_feat], 0)
        trainval_assigned, centroids = run_kmeans(
            feat_train=train_feat,
            feat_trainval=trainval,
            cluster_k=cluster_k,
            niter=niter,
            minp=minp,
        )
        f["train"][:, token_id] = trainval_assigned[
            : train_feat.shape[0],
        ]
        f["val"][:, token_id] = trainval_assigned[
            train_feat.shape[0]:,
        ]
        f["centroids"][:, token_id, :] = centroids

    ##################
    if "train_labels" in f_feat:
        train_labels, val_labels = get_feat(
            "train_labels"), get_feat("val_labels")
        trainval_labels = np.concatenate([train_labels, val_labels], 0)
        train_size = train_feat.shape[0]
        cluster_dict = cal_cluster_metric(
            pred_np=trainval_assigned[:train_size], gt_np=trainval_labels[:train_size]
        )
        logger.warning(f"Train set cluster result: {cluster_dict}")
        cluster_dict = cal_cluster_metric(
            pred_np=trainval_assigned[train_size:
                                      ], gt_np=trainval_labels[train_size:]
        )
        logger.warning(f"Val set cluster result: {cluster_dict}")

    f.close()
    f_feat.close()
    logger.warning(f"saving {_h5py_path}")
    logger.warning("*" * 66)
