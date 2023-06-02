
import imp
import json
from pathlib import Path
from loguru import logger
import numpy as np
import h5py
import torch 
from clustering.cal_cluster_metric import cal_cluster_metric
import torch.nn.functional as F
from dataset.ds_utils.dataset_common_utils import ds_has_label_info, skip_id2name

def cal_cluster_statistics(dl, nmi=True, ami=False, ari=False):
    return dict()
    dl.cluster_hist = np.histogram(dl.cluster_list, bins=dl.cluster_k)
    dl.cluster_ids = np.array(dl.cluster_list)
    logger.warning(f'cluster_k ={dl.cluster_k}, max={np.max(dl.cluster_list)}, min={np.min(dl.cluster_list)}, cal_cluster_metric...ING..')
    dl.cluster_eval_dict = cal_cluster_metric(pred_np=dl.cluster_list, gt_np=dl.label_list, need_nmi=nmi, need_ami=ami, need_ari=ari)
    logger.warning(f'Train={dl.train}, NMI against label: {dl.cluster_eval_dict}') 
    return dl.cluster_eval_dict


def set_cluster_info(dl, h5_file):
    
    dl.cluster_file = Path(h5_file).expanduser().resolve()
    dl.cluster_k = h5py.File(dl.cluster_file, 'r')['all_attributes'].attrs['cluster_k']
    dl.cluster_list = h5py.File(dl.cluster_file, 'r')[dl.split_name] #[N, 1]
    dl.cluster_list_random = np.random.randint(0, dl.cluster_k, size = dl.cluster_list.shape) #[N, 1], prepare ready
    if ds_has_label_info(dl.dataset_name) and False:
        cal_cluster_statistics(dl)


def get_cluster_by_index(dl, index):

    if skip_id2name(dl.dataset_name):
        id_in_h5 = index 
    else:
        filename = dl.id2name(index)
        id_in_h5 = int(dl.filename2id[filename])

    cluster_id = dl.cluster_list[id_in_h5]

    cluster_onehot = F.one_hot(torch.tensor(cluster_id).long(), num_classes=dl.cluster_k)#[cluster_k]
    cluster_random_onehot = F.one_hot(torch.tensor(dl.cluster_list_random[id_in_h5]).long(), num_classes=dl.cluster_k)#[cluster_k]
    
    return cluster_onehot, cluster_id, cluster_random_onehot


   



    