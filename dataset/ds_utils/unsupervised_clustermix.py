
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
    raise NotImplementedError
    dl.cluster_hist = np.histogram(dl.cluster_list, bins=dl.cluster_k)
    dl.cluster_ids = np.array(dl.cluster_list)
    logger.warning(f'cluster_k ={dl.cluster_k}, max={np.max(dl.cluster_list)}, min={np.min(dl.cluster_list)}, cal_cluster_metric...ING..')
    dl.cluster_eval_dict = cal_cluster_metric(pred_np=dl.cluster_list, gt_np=dl.label_list, need_nmi=nmi, need_ami=ami, need_ari=ari)
    logger.warning(f'Train={dl.train}, NMI against label: {dl.cluster_eval_dict}') 
    return dl.cluster_eval_dict


def set_clustermix_info(dl, h5_file, h5_file_tomix):
    
    def set_attr(h5_file, split_name):
        cluster_file = Path(h5_file).expanduser().resolve()
        cluster_k = h5py.File(cluster_file, 'r')['all_attributes'].attrs['cluster_k']
        cluster_list = h5py.File(cluster_file, 'r')[split_name] #[N, 1]
        cluster_list_random = np.random.randint(0, cluster_k, size=cluster_list.shape) #[N, 1], prepare ready
        return cluster_file, cluster_k, cluster_list, cluster_list_random

    dl.cluster_file, dl.cluster_k, dl.cluster_list, dl.cluster_list_random = set_attr(h5_file, dl.split_name)
    dl.cluster_file_tomix, dl.cluster_k_tomix, dl.cluster_list_tomix, dl.cluster_list_random_tomix = set_attr(h5_file_tomix, dl.split_name)
    



def get_clustermix_by_index(dl, index):

    if skip_id2name(dl.dataset_name):
        id_in_h5 = index 
    else:
        filename = dl.id2name(index)
        id_in_h5 = int(dl.filename2id[filename])

    def get_cluster_attr(id_in_h5, cluster_list, cluster_list_random, cluster_k):
    
        cluster_id = cluster_list[id_in_h5]

        cluster_onehot = F.one_hot(torch.tensor(cluster_id).long(), num_classes=cluster_k)#[cluster_k]
        cluster_random_onehot = F.one_hot(torch.tensor(cluster_list_random[id_in_h5]).long(), num_classes=cluster_k)#[cluster_k]
        
        if len(cluster_onehot.shape)==2 and cluster_onehot.shape[1]>1:#For PCA group
            cluster_onehot = cluster_onehot.reshape(-1)
            cluster_random_onehot = cluster_random_onehot.reshape(-1)

        return  cluster_onehot, cluster_random_onehot
    
    cluster_onehot, cluster_random_onehot = get_cluster_attr(id_in_h5, dl.cluster_list, dl.cluster_list_random, dl.cluster_k)
    cluster_onehot_tomix, cluster_random_onehot_tomix = get_cluster_attr(id_in_h5, dl.cluster_list_tomix, dl.cluster_list_random_tomix, dl.cluster_k_tomix)

    cluster_onehot = torch.cat([cluster_onehot, cluster_onehot_tomix], dim=0)
   
    cluster_random_onehot = torch.cat([cluster_random_onehot, cluster_random_onehot_tomix], dim=0)

    return cluster_onehot, cluster_random_onehot


   



    