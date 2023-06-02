import json
import os
from pathlib import Path
import h5py
import torch
from dataset.ds_utils.dataset_common_utils import ds_has_label_info, skip_id2name
from dataset.ds_utils.unsupervised_clustermix import get_clustermix_by_index, set_clustermix_info
from .supervised_label import get_labelinfo_by_index, set_label_info
from dataset.ds_utils.unsupervised_centroid import (
    get_centroid_by_index,
    set_centroid_info,
)

from dataset.ds_utils.unsupervised_cluster import get_cluster_by_index, set_cluster_info
from dataset.ds_utils.unsupervised_feat import get_feat_by_index, set_feat_info
from dataset.ds_utils.unsupervised_knn import get_knnfeat_by_index, set_knnfeat_info
from dataset.ds_utils.unsupervised_layout import get_clusterlayout_by_index
from dataset.ds_utils.unsupervised_patchcluster import (
    get_patchcluster_by_index,
    set_patchcluster_info,
)
from dataset.ds_utils.unsupervised_patchfeat import get_patchfeat_by_index
from loguru import logger


def add_prefix(root_global, h5_file):
    if h5_file is not None:
        h5_file = os.path.join(root_global, h5_file)
    return h5_file


def set_cond(dl, h5_file, h5_file2=None):
    assert hasattr(dl, "split_name")
    assert hasattr(dl, "dataset_name")
    assert hasattr(dl, "id2name")
    assert hasattr(dl, "root_global")

    root_global = dl.root_global
    h5_file = add_prefix(root_global, h5_file)
    h5_file2 = add_prefix(root_global, h5_file2)

    def load_name2id(dataset_name):
        if skip_id2name(dataset_name):
            return None
        else:
            json_path = str(Path(h5_file).expanduser().resolve()
                            ).replace(".h5", ".json")
            logger.warning(f"load_name2id from {json_path}")
            return json.load(open(json_path, "r",))["name2id"]

    if ds_has_label_info(dl.dataset_name):
        set_label_info(dl)

    if dl.condition_method is None:
        pass

    elif dl.condition_method in ["label", "attr", "layout", "stegoclusterlayout"]:
        pass

    elif dl.condition_method in ["feat", "patchfeat"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_feat_info(dl, h5_file)

    elif dl.condition_method in ["cluster", "cluster_lookup", "clusterrandom"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_cluster_info(dl, h5_file)

    elif dl.condition_method in ["clustermix"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        assert h5_file2 is not None
        set_clustermix_info(dl, h5_file, h5_file2)

    elif dl.condition_method in ["labelcluster"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_label_info(dl)
        set_cluster_info(dl, h5_file)

    elif dl.condition_method in ["clusterlayout"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_cluster_info(dl, h5_file)

    elif dl.condition_method in ["labelcentroid"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_label_info(dl)
        set_centroid_info(dl, h5_file)

    elif dl.condition_method in ["centroid"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_centroid_info(dl, h5_file)

    elif dl.condition_method in ["patchcluster"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_patchcluster_info(dl, h5_file)

    elif dl.condition_method in ["knn_feat"]:
        dl.filename2id = load_name2id(dl.dataset_name)
        set_knnfeat_info(dl, h5_file)

    else:
        raise ValueError(dl.condition_method)


def get_cond(dl, condition_method, index, dataset_name, mask=None, attr_nhot=None):

    result = dict()
    condition = dl.condition

    if ds_has_label_info(dataset_name):
        label_info = get_labelinfo_by_index(dl, index)
        result.update(label_info)

    if condition_method is None:
        pass

    elif condition_method in ["attr"]:
        assert not ds_has_label_info(dataset_name)

    elif condition_method in ["label"]:
        assert ds_has_label_info(dataset_name)

    elif condition_method in ["feat"]:
        feat = get_feat_by_index(dl, index)
        result.update(feat=feat)

    elif condition_method in ["layout", "stegoclusterlayout"]:
        pass

    elif condition_method in ["patchfeat"]:
        patchfeat = get_patchfeat_by_index(dl, index)
        result.update(patchfeat=patchfeat)

    elif condition_method in ["cluster", "cluster_lookup", "clusterlayout", 'clusterrandom']:
        cluster_onehot, cluster_id, cluster_random = get_cluster_by_index(
            dl, index)
        result.update(
            cluster=cluster_onehot, clusterrandom=cluster_random, cluster_id=cluster_id
        )

    elif condition_method in ["clustermix"]:
        cluster_onehot, cluster_random = get_clustermix_by_index(
            dl, index)
        result.update(
            clustermix=cluster_onehot, clustermix_random=cluster_random,
        )

    elif condition_method in ["knn_feat"]:
        knn_feat, knn_feat_random, nns = get_knnfeat_by_index(dl, index)
        result.update(knn_feat=knn_feat,
                      knn_feat_random=knn_feat_random, nns=nns)

    elif condition_method in ["patchcluster"]:
        patchcluster = get_patchcluster_by_index(dl, index)
        result.update(patchcluster=patchcluster)

    elif condition_method in ["labelcentroid"]:
        label_onehot = get_labelinfo_by_index(dl, index)["label_onehot"]
        if not ds_has_label_info(dataset_name):
            raise ValueError(condition_method)
        centroid, cluster_id, cluster_onehot, centroid_random = get_centroid_by_index(
            dl, index
        )
        labelcentroid = torch.cat(
            [label_onehot, torch.from_numpy(centroid)], 0)
        result.update(labelcentroid=labelcentroid, cluster_id=cluster_id)

    elif condition_method in ["labelcluster"]:
        label_onehot = get_labelinfo_by_index(dl, index)["label_onehot"]
        if not ds_has_label_info(dataset_name):
            raise ValueError(condition_method)

        cluster_onehot, cluster_id, cluster_random = get_cluster_by_index(
            dl, index)

        labelcluster = torch.cat([label_onehot, cluster_onehot], 0)
        result.update(
            labelcluster=labelcluster,
            cluster_id=cluster_id,
        )

    elif condition_method in ["centroid"]:
        centroid, cluster_id, cluster_onehot, centroid_random = get_centroid_by_index(
            dl, index
        )
        result.update(
            centroid=centroid, centroid_random=centroid_random, cluster_id=cluster_id
        )

    else:
        raise ValueError(condition_method)

    return result
