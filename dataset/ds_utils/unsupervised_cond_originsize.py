import h5py
import torch


from dataset.ds_utils.unsupervised_layout import (
    get_clusterlayout_by_index,
)
from dataset.ds_utils.unsupervised_lost import get_lostbbox_by_index, set_lost_info


def set_lostbbox_in_originsize(dl, lost_file=None):
    assert hasattr(dl, "split_name")
    assert hasattr(dl, "dataset_name")

    if dl.condition_method is None:
        pass

    elif dl.condition_method in ["clusterlayout", "layout"]:
        if dl.condition.layout.how in ['oracle', 'stego']:
            pass
        else:
            set_lost_info(dl, lost_file)


def get_lostbbox_originsize(dl, condition_method, index):
    condition = dl.condition

    if condition_method is None:
        return None

    elif condition_method in ["clusterlayout", "layout"]:
        if condition_method in ['clusterlayout']:
            _how = condition.clusterlayout.how
        elif condition_method in ['layout']:
            _how = condition.layout.how
        else:
            raise NotImplementedError

        if _how in ["oracle", "stego"]:
            pass
        elif _how == "dinoseg":
            raise ValueError("there is a bug here about spatial alignment.")
            (
                cluster_onehot,
                cluster_id,
                cluster_random,
                _layout,
            ) = get_clusterlayout_by_index(dl, index)
            result.update(
                cluster=cluster_onehot,
                cluster_random=cluster_random,
                cluster_id=cluster_id,
                layout=_layout,
            )
        elif _how == "lost":
            bbox = get_lostbbox_by_index(dl, index)
            return bbox
        else:
            raise ValueError(_how)
