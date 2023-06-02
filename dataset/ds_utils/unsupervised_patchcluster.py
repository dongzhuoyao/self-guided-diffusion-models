

from pathlib import Path
import h5py
import torch
import torch.nn.functional as F

def set_patchcluster_info(dl, h5_file):

    dl.cluster_file = Path(h5_file).expanduser().resolve()
    dl.cluster_k = h5py.File(dl.cluster_file, 'r')['all_attributes'].attrs['cluster_k']

    
def get_patchcluster_by_index(dl, index):

    filename = dl.id2name(index)
    id_in_h5 = int(dl.filename2id[filename])

    dl.cluster_list = h5py.File(dl.cluster_file, 'r')[dl.split_name]
    patchcluster_onehot=dl.cluster_list[id_in_h5]#[patches, 1]
    patchcluster_onehot = F.one_hot(torch.tensor(patchcluster_onehot).long(), num_classes=dl.cluster_k) #[patches, cluster_K],[197, 1600]
    return patchcluster_onehot