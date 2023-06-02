import imp
import json
from pathlib import Path
from loguru import logger
import numpy as np
import h5py
import torch
from clustering.cal_cluster_metric import cal_cluster_metric
import torch.nn.functional as F

from dataset.ds_utils.dataset_common_utils import ds_has_label_info


def cal_cluster_statistics(dl, nmi=True, ami=False, ari=False):
    dl.cluster_hist = np.histogram(dl.cluster_list, bins=dl.cluster_k)
    dl.cluster_ids = np.array(dl.cluster_list)
    logger.warning(
        f"cluster_k ={dl.cluster_k}, max={np.max(dl.cluster_list)}, min={np.min(dl.cluster_list)}, cal_cluster_metric...ING.."
    )
    dl.cluster_eval_dict = cal_cluster_metric(
        pred_np=dl.cluster_list,
        gt_np=dl.labels,
        need_nmi=nmi,
        need_ami=ami,
        need_ari=ari,
    )
    logger.warning(f"Train={dl.train}, NMI against label: {dl.cluster_eval_dict}")
    return dl.cluster_eval_dict


def get_clusterlayout_by_index(dl, index):
    filename = dl.id2name(index)
    id_in_h5 = int(dl.filename2id[filename])

    if False:
        dl.layout_list = h5py.File(dl.cluster_file, "r")[dl.split_name]
        attentions = h5py.File(dl.cluster_file, "r")[f"{dl.split_name}_attentions"]
        attentions = torch.from_numpy(attentions[id_in_h5])
        _layout = get_attention_layout_by_thres(
            attentions=attentions,
            threshold=dl.condition.clusterlayout.threshold,
            output_size=dl.size,
        )
    else:
        _layout = None

    dl.cluster_list = h5py.File(dl.cluster_file, "r")[dl.split_name]
    cluster_id = dl.cluster_list[id_in_h5]
    cluster_onehot = F.one_hot(
        torch.tensor(cluster_id).long(), num_classes=dl.cluster_k
    )  # [cluster_k]
    cluster_random_onehot = F.one_hot(
        torch.tensor(dl.cluster_list_random[id_in_h5]).long(), num_classes=dl.cluster_k
    )  # [cluster_k]
    return cluster_onehot, cluster_id, cluster_random_onehot, _layout


def get_attention_layout_by_thres(attentions, threshold=0.8, output_size=32):
    w_featmap, h_featmap = 14, 14
    attentions = torch.mean(attentions, dim=0, keepdim=True)  # 6 heads to 1 head
    nh, _ = attentions.shape
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            torch.nn.functional.interpolate(
                th_attn.unsqueeze(0), size=output_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

    return th_attn
