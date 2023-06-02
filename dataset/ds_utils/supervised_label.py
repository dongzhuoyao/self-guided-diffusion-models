import numpy as np

import torch
from loguru import logger


def set_label_info(dl):
    dl.label_list = np.array(dl.label_list)
    assert len(np.unique(dl.label_list)) == dl.label_num
    if dl.label_list.min() == 1:
        dl.label_list = dl.label_list - 1
        logger.warning("label index started from 1, changed to start from 0 now...")

    dl.label_list_random = np.random.randint(
        0, dl.label_num, size=dl.label_list.shape
    )  # [N, 1], prepare ready

    if dl.condition is not None:
        noise_ratio = dl.condition.label.noise_ratio
        if noise_ratio > 0:
            is_noise_mask = (
                np.random.uniform(0, 1, size=dl.label_list.shape) < noise_ratio
            )
            dl.label_list = (
                is_noise_mask * dl.label_list_random
                + (1 - is_noise_mask) * dl.label_list
            )
            logger.warning(f"mix noise into label with ratio {noise_ratio}")


def get_labelinfo_by_index(dl, index):
    label_id = dl.label_list[index]
    label_onehot = torch.nn.functional.one_hot(
        torch.tensor(label_id), num_classes=dl.label_num
    )
    label_random_id = dl.label_list_random[index]
    label_random_onehot = torch.nn.functional.one_hot(
        torch.tensor(label_random_id), num_classes=dl.label_num
    )
    return dict(label_id=label_id, label_onehot=label_onehot, label=label_onehot,label_random=label_random_onehot)
