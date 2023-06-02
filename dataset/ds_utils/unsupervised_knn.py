
import imp
import json
from pathlib import Path
from loguru import logger
import numpy as np
import h5py
import torch 
import torch.nn.functional as F

from dataset.ds_utils.dataset_common_utils import normalize_featnp




def set_knnfeat_info(dl,h5_file):
    assert dl.condition.knn_feat.knn_k is not None 
    dl.knn_file = Path(h5_file).expanduser().resolve()
    dl.feat_list = h5py.File(h5_file, 'r')['train_feat']#feat always from trainset 
    dl.nns_list = h5py.File(dl.knn_file, 'r')[f'{dl.split_name}_nns'] #[N, k]
    dl.nns_radius_list = h5py.File(dl.knn_file, 'r')[f'{dl.split_name}_nns_radius'] #[N, k]
    dl.nns_list_random = np.random.randint(0, len(dl.feat_list), size=len(dl.nns_list)) #[N, 1]

    
def get_knnfeat_by_index(dl, index):

    filename = dl.id2name(index)
    id_in_h5 = int(dl.filename2id[filename])

    nns=dl.nns_list[id_in_h5]
    knn_k = dl.condition.knn_feat.knn_k
    assert knn_k<=len(nns)
    _nn_index = int(nns[np.random.randint(0,knn_k,(1))])
    _nn_feat = dl.feat_list[_nn_index]

    _nn_feat_random = dl.feat_list[dl.nns_list_random[id_in_h5]]

    _nn_feat = normalize_featnp(_nn_feat)
    _nn_feat_random = normalize_featnp(_nn_feat_random)

    return _nn_feat, _nn_feat_random, nns
