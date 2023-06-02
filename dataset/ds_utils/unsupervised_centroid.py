import json
from pathlib import Path

import numpy as np
import torch
import h5py
import torch.nn.functional as F
from dataset.ds_utils.dataset_common_utils import (
    ds_has_label_info,
    normalize_featnp,
    skip_id2name,
)

from dataset.ds_utils.unsupervised_cluster import cal_cluster_statistics


def set_centroid_info(dl, h5_file):

    dl.centroid_file = Path(h5_file).expanduser().resolve()
    dl.cluster_k = h5py.File(dl.centroid_file, "r")["all_attributes"].attrs["cluster_k"]
    dl.centroid_list = h5py.File(dl.centroid_file, "r")["centroids"]  # [N, K]

    dl.cluster_list = h5py.File(dl.centroid_file, "r")[dl.split_name]  # [N, 1]
    dl.cluster_list_random = np.random.randint(
        0, dl.cluster_k, size=dl.cluster_list.shape
    )  # [N, 1]

    if ds_has_label_info(dl.dataset_name):
        cal_cluster_statistics(dl)


def get_centroid_by_index(dl, index):

    if skip_id2name(dl.dataset_name):
        id_in_h5 = index
    else:
        filename = dl.id2name(index)
        id_in_h5 = int(dl.filename2id[filename])

    cluster_id = dl.cluster_list[id_in_h5]
    centroid = dl.centroid_list[cluster_id]
    cluster_onehot = F.one_hot(
        torch.tensor(cluster_id).long(), num_classes=dl.cluster_k
    )  # [cluster_k]

    centroid_random = dl.centroid_list[dl.cluster_list_random[id_in_h5]]

    # centroid = normalize_featnp(centroid)
    # entroid_random = normalize_featnp(centroid_random)
    # centroid don't need feature normalization here, as it has been done in faiss.kmeans of  cluster_on_feat.py files.

    return centroid, cluster_id, cluster_onehot, centroid_random
