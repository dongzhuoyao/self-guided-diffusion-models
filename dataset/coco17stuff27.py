from pathlib import Path
import pickle
import random
from loguru import logger

import torch
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
from dataset.ds_utils.unsupervised_cond import set_cond
from dataset.ds_utils.unsupervised_cond_originsize import (
    set_lostbbox_in_originsize,
)
from dataset.transforms.complex_ds_common_util import (
    RandomScaleCrop,
    get_item_complex,
    set_stego,
)


class CocoStuffDataset(Dataset):
    class_names = [f'cls{i}' for i in range(182)]
    class_names_4wandb = dict()
    for i, c in enumerate(class_names):
        class_names_4wandb[i] = c

    def __init__(
        self,
        size,
        root,
        root_global,
        root_coco17_annos,
        h5_file=None,
        attr_num=182,
        split="train",
        debug=False,
        lost_file=None,
        img_save_path=None,
        condition_method=None,
        condition=None,
        size4cluster=320,
        size4crop=224,
    ):
        super().__init__()

        if img_save_path is None:
            split = 'train'
            logger.warning("force split to train now!")
        else:
            logger.warning(f'img_save_path is not None, so split is {split}')

        self.dataset_name = "cocostuff64"

        self.attr_num = attr_num
        assert self.attr_num in [182, 27]
        self.size4crop = size4crop
        self.size4cluster = size4cluster
        self.split_name = split
        self.train = True if split == "train" else False
        self.condition_method = condition_method
        self.condition = condition
        self.lost_file = lost_file
        self.img_save_path = img_save_path
        self.debug = debug
        self.size = size
        self.label_num = self.attr_num
        self.root_global = root_global

        self.transform = RandomScaleCrop(base_size=size4crop, resize_size=size)
        ##################

        if self.attr_num == 27:
            pkl_path = "dataset/data_files/fine_to_coarse_dict.pickle"
            logger.warning("use 27 attrs")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.attr182_to_attr27 = data['fine_index_to_coarse_index']
        else:
            logger.warning("use 182 attrs, sure??")

        set_stego(dl=self, condition_method=condition_method,
                  condition=condition)
        set_cond(self, h5_file)
        set_lostbbox_in_originsize(self, lost_file=self.lost_file)

        self.root = os.path.join(
            Path(root).expanduser().resolve(), split)
        logger.info(f"loading dataset from {self.root}, is this a ssd?")
        root_coco17_annos = str(Path(root_coco17_annos).expanduser().resolve())

        self.root_annotation = os.path.join(
            root_coco17_annos, split+'2017')
        assert len(os.listdir(self.root_annotation)) > 0
        self.image_ids = os.listdir(self.root)
        print(f"image_length={len(self.image_ids)}")
        self.image_ids.sort()

        if self.is_stego:
            assert len(os.listdir(self.stego_mask_dir)) > 0
            self.stego_paths = []
            new_image_ids = []
            for image_id in self.image_ids:
                image_name = os.path.basename(image_id).replace(".jpg", ".png")
                full_stego_path = os.path.join(self.stego_mask_dir, image_name)
                if os.path.isfile(full_stego_path):
                    self.stego_paths.append(
                        full_stego_path
                    )
                    new_image_ids.append(image_id)

            self.image_ids = new_image_ids
            print(f"new image_length={len(self.image_ids)}")
        self.num = len(self.image_ids)

    def get_imagename_by_index(self, index):
        return str(self.image_ids[index])

    def read_lost_pickle(self, filename):
        with open(filename, "rb") as f:
            preds_dict = pickle.load(f)
        print("Predictions saved at %s" % filename)
        return preds_dict

    def id2name(self, index):
        return self.image_ids[index]

    def load_img(self, image_id):
        filepath = os.path.join(self.root, image_id)
        img = Image.open(filepath).convert("RGB")
        return img

    def _read_img_segmask(self, index):
        image_id = self.image_ids[index]
        image = self.load_img(image_id)

        filepath = os.path.join(
            self.root_annotation, image_id.replace(".jpg", ".png"))
        mask = Image.open(filepath)

        return image, mask

    def __len__(self):
        if self.debug:
            return 120
        return self.num

    def __getitem__(self, index):
        result = get_item_complex(dl=self, index=index)
        return result


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
