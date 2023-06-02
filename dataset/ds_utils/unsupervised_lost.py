
import imp
from pathlib import Path
from loguru import logger
import numpy as np
import h5py
import torch 
from clustering.cal_cluster_metric import cal_cluster_metric
import torch.nn.functional as F

from dataset.ds_utils.dataset_common_utils import ds_has_label_info



def set_lost_info(dl,lost_file):
    dl.lost_dict = h5py.File(Path(lost_file).expanduser().resolve(), 'r')
   

def get_lostbbox_by_index(dl, index):
    image_name = dl.get_imagename_by_index(index) 
    _bbox = dl.lost_dict[f'{image_name}_bbox']
    _bbox = np.array(_bbox)
    #_clusterid = dl.lost_dict[f'{image_name}_clusterid']
    #_cluster_k = dl.lost_dict['all_attributes'].attrs['cluster_k']
    return _bbox


