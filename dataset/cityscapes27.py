from cmath import log
from genericpath import isfile
from pathlib import Path
import pickle
import random
from loguru import logger

import torch
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import torch.nn.functional as F
import json
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

from torchvision import transforms
import cv2


class CityscapesDataset(Dataset):
    # https://github.com/ultralytics/yolov5/blob/4d8d84b0ea7147aca64e7c38ce1bdb5fbb9c5a53/data/coco.yaml#L18
    class_names = [
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]  # class names, first 'person' is actually background?

    class_names_4wandb = dict()
    for i, c in enumerate(class_names):
        class_names_4wandb[i] = c

    def __init__(
        self,
        size,
        root,
        h5_file=None,
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

        split = 'train'
        logger.warning("force split to train now!")

        self.size4crop = size4crop
        self.size4cluster = size4cluster
        self.dataset_name = "cocostuff64"
        self.split_name = split

        if split == 'train':
            self.train = True
            root = os.path.join(root, 'train_extra_images')
        else:
            self.train = False
            root = os.path.join(root, 'val_images')

        root = Path(root).expanduser().resolve()
        logger.info(f"loading dataset from {root}, is this a ssd?")
        self.root = root

        self.condition_method = condition_method
        self.condition = condition
        self.lost_file = lost_file
        self.img_save_path = img_save_path
        self.debug = debug
        self.size = size
        self.label_num = 27

        self.transform = RandomScaleCrop(base_size=size4crop, resize_size=size)
        ##################

        set_stego(dl=self, condition_method=condition_method,
                  condition=condition)
        set_cond(self, h5_file)
        set_lostbbox_in_originsize(self, lost_file=self.lost_file)

        self.image_ids = os.listdir(self.root)
        self.image_ids.sort()

        logger.warning(f"image_length={len(self.image_ids)}")
        

        if self.is_stego:
            self.stego_paths = []
            new_image_ids = []
            for image_id in self.image_ids:
                image_name = os.path.basename(image_id)
                full_stego_path = os.path.join(self.stego_mask_dir, image_name)
                if os.path.isfile(full_stego_path):
                    self.stego_paths.append(
                        full_stego_path
                    )
                    new_image_ids.append(image_id)
            self.image_ids = new_image_ids
            assert len(self.image_ids)>0, "no stego images found!"
            print(f"new image_length={len(self.image_ids)}")
        self.num = len(self.image_ids)

    def get_imagename_by_index(self, index):
        return str(self.image_ids[index])

    def read_lost_pickle(self, filename):
        raise NotImplementedError
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

        mask = None  # image.copy()
        return image, mask

    def __len__(self):
        if self.debug:
            return 120
        return self.num

    def __getitem__(self, index):
        return get_item_complex(dl=self, index=index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
