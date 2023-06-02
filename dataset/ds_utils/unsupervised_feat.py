import json
from pathlib import Path
import h5py

from dataset.ds_utils.dataset_common_utils import normalize_featnp, skip_id2name


def set_feat_info(dl, h5_file):

    dl.feat_file = Path(h5_file).expanduser().resolve()


def get_feat_by_index(dl, index):

    if skip_id2name(dl.dataset_name):
        id_in_h5 = index
    else:
        filename = dl.id2name(index)
        id_in_h5 = int(dl.filename2id[filename])

    feat = h5py.File(dl.feat_file, "r")[dl.split_name][id_in_h5]
    feat = normalize_featnp(feat)

    return feat
