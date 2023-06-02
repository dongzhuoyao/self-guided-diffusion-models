
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
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def copy_json_file(feat_h5py_path, dst_h5py_path):
    json_name = str(feat_h5py_path).replace(".h5", ".json")
    assert os.path.exists(json_name), f"{json_name} not exists"
    new_json_name = str(dst_h5py_path).replace(".h5", ".json")
    shutil.copyfile(json_name, new_json_name)
    logger.warning(f"copy json from {json_name} to {new_json_name}")


def clustering(
    feat_h5_path, nns, cluster_k, niter, minp, cluster_h5_root=None, debug=False
):
    feat_h5_path = str(Path(feat_h5_path).expanduser().resolve())
    f_feat = h5py.File(feat_h5_path, mode="r")
    dset_feat = f_feat["all_attributes"]
    dataset_name = dset_feat.attrs["dataset_name"]
    feat_from = dset_feat.attrs["feat_from"]
    feat_dim = dset_feat.attrs["feat_dim"]
    is_grey = int(dset_feat.attrs["is_grey"]
                  ) if "is_grey" in dset_feat.attrs else 0
    logger.warning(dataset_name)
    logger.warning(feat_from)
    logger.warning(feat_dim)
    ####################

    def get_feat(split):
        return f_feat[split][:1000] if debug else f_feat[split][:]

    if debug:
        cluster_k, niter = 10, 30
    ########################

    time_str = datetime.now().isoformat(timespec="hours")

    # v2: only use train feat now
    # v3: add nns information
    # v4, following feat version as v4
    _cluster_h5_path = f"~/data/sg_data/cluster/v4_{dataset_name}_cluster{cluster_k}_iter{niter}minp{minp}_nns{nns}_{feat_from}_grey{is_grey}_{time_str}_{sha[:7]}.h5"
    if debug:
        _cluster_h5_path = _cluster_h5_path.replace(".h5", "debug.h5")
    if cluster_h5_root is not None:
        _cluster_h5_path = _cluster_h5_path.replace(
            "~/data/sg_data/cluster", cluster_h5_root)
    _cluster_h5_path = Path(_cluster_h5_path).expanduser().resolve()

    logger.warning(_cluster_h5_path)
    copy_json_file(feat_h5py_path=feat_h5_path, dst_h5py_path=_cluster_h5_path)
    logger.warning(f"begin kmeans fit")

    f = h5py.File(_cluster_h5_path, mode="w")
    f.close()
    f = h5py.File(_cluster_h5_path, mode="a")
    f.create_dataset(
        "train", data=np.ones(shape=(len(get_feat("train")),), dtype=np.int64) * -1
    )
    f.create_dataset(
        "val", data=np.ones(shape=(len(get_feat("val")),), dtype=np.int64) * -1
    )
    # f.create_dataset('centroids', data=np.ones(cluster_k,), dtype=np.int64)*-1)
    dset = f.create_dataset("all_attributes", (1,))
    dset.attrs["dataset_name"] = dset_feat.attrs["dataset_name"]
    dset.attrs["feat_from"] = dset_feat.attrs["feat_from"]
    dset.attrs["cluster_k"] = cluster_k
    dset.attrs["feat_dim"] = dset_feat.attrs["feat_dim"]
    dset.attrs["is_grey"] = is_grey

    ######################################
    train_feat, val_feat = get_feat("train"), get_feat("val")
    trainval_feat = np.concatenate([train_feat, val_feat], 0)
    trainset_size = len(train_feat)

    if nns > 0:
        logger.warning("creating nns dataset....!")
        sample_nns, sample_nn_radius_all = run_nns(
            np.array(train_feat),
            features_trainval=trainval_feat,
            k_nn=nns,
            faiss_lib=True,
            gpu=True,
        )
        f.create_dataset("train_feat", data=train_feat)
        f.create_dataset(
            "train_nns",
            data=sample_nns[
                :trainset_size,
            ],
        )
        f.create_dataset(
            "train_nns_radius",
            data=sample_nn_radius_all[
                :trainset_size,
            ],
        )
        f.create_dataset(
            "val_nns",
            data=sample_nns[
                trainset_size:,
            ],
        )
        f.create_dataset(
            "val_nns_radius",
            data=sample_nn_radius_all[
                trainset_size:,
            ],
        )
    else:
        logger.warning("skipping nns....!")

    if "train_attentions" in list(f_feat.keys()):
        logger.warning("copy attentions to dataset")
        f.create_dataset("train_attentions", data=f_feat["train_attentions"])
        f.create_dataset("val_attentions", data=f_feat["val_attentions"])

    trainval_assigned, centroids = run_kmeans(
        feat_train=train_feat,
        feat_trainval=trainval_feat,
        cluster_k=cluster_k,
        niter=niter,
        minp=minp,
    )
    f["train"][:] = trainval_assigned[
        :trainset_size,
    ]
    f["val"][:] = trainval_assigned[
        trainset_size:,
    ]
    f.create_dataset("centroids", data=centroids)

    ##################
    if "train_labels" in f_feat:
        train_labels, val_labels = get_feat(
            "train_labels"), get_feat("val_labels")
        train_val_labels = np.concatenate([train_labels, val_labels], 0)
        trainset_size = len(train_feat)
        cluster_dict = cal_cluster_metric(
            pred_np=trainval_assigned[:trainset_size],
            gt_np=train_val_labels[:trainset_size],
        )
        logger.warning(f"Train set cluster result: {cluster_dict}")

        cluster_dict = cal_cluster_metric(
            pred_np=trainval_assigned[trainset_size:],
            gt_np=train_val_labels[trainset_size:],
        )
        logger.warning(f"Val set cluster result: {cluster_dict}")

    f.close()
    f_feat.close()
    logger.warning(f"saving {_cluster_h5_path}")
    logger.warning("*" * 66)
