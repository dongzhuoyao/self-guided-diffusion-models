# https://github.com/sndnyang/imagenet64x64/blob/master/imagenet_64x64.ipynb

import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import torchvision
from PIL import Image
from einops import rearrange
from torchvision.transforms import transforms
import h5py
import random

import torch.nn.functional as F
import torch
import os
import pickle
import numpy as np
from loguru import logger
from tqdm import tqdm
from dataset.ds_utils.unsupervised_cond import get_cond, set_cond
from diffusion_utils.util import normalize_to_neg_one_to_one, make_clean_dir


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo)
    return dict


# another impl can be here: https://github.com/ActiveVisionLab/BNInterpolation/blob/master/imagenet32_dataset.py
class ImageNet_Pickle(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        root_global,
        train,
        image_size=32,
        h5_file=None,
        h5_file2=None,
        condition_method=None,
        condition=None,
        debug=False,
        img_save_path=None,
        data_ratio=1.0,
        corruption = 0,
        subgroup=1,
    ) -> None:

        super().__init__()
        self.dataset_name = "inp"
        self.train = train
        self.condition_method = condition_method
        self.condition = condition
        self.split_name = "train" if self.train else "val"
        self.img_save_path = img_save_path
        self.debug = debug
        self.size = image_size
        self.root_global = root_global
        self.label_num = 1000
        self.data_ratio = data_ratio
        self.corruption = corruption
        self.subgroup = subgroup
        logger.info(f"data_ratio {self.data_ratio} in ImageNet_Pickle")

        if False:
            self.pickle_to_h5(root)

        self.set_images_labels(root=root, train=train)
        set_cond(self, h5_file, h5_file2)

    def read_raw_pickle(self, root, train):
        if train:
            logger.info("reading pickle file, pls wait..")
            _data, _labels = [], []
            for _i in range(1, 11):
                d = unpickle(os.path.join(root, f"train_data_batch_{_i}"))
                _data.append(d["data"])
                _labels.extend(d["labels"])

                if self.debug and self.size == 64:
                    break  # save memory in case in64

            data, labels = np.concatenate(_data, 0), _labels
        else:
            d = unpickle(os.path.join(root, "val_data"))
            data, labels = d["data"], d["labels"]

        # indexing from 0 instead of 1
        labels = np.array([i - 1 for i in labels])
        return data, labels

    def read_image_data(self, root, train):
        if self.size == 64:
            h5_ds_path = (
                Path(os.path.join(root, "in64pickle.h5")).expanduser().resolve()
            )
            h5_dataset = h5py.File(h5_ds_path, mode="r")
            data, labels = (
                h5_dataset[f"data_{self.split_name}"],
                h5_dataset[f"labels_{self.split_name}"],
            )
            print(f"reading from h5 file now. {h5_ds_path}")
            return data, labels

        data, labels = self.read_raw_pickle(root, train)
        return data, labels

    def id2name(self, index):
        file_name = str(index) + ".jpg"  # fake it
        return file_name

    def get_root_path(self, root):
        root = Path(root).expanduser().resolve()
        logger.info(f"loading imagenet from {root}, is this a ssd?")
        if self.size == 32:
            root = os.path.join(root, "size32")
        elif self.size == 64:
            root = os.path.join(root, "size64")
        else:
            raise ValueError(self.size)
        return root

    def pickle_to_h5(self, root):
        root = self.get_root_path(root)

        dest_h5_path = os.path.join(root, "in64pickle.h5")
        print(f"transfer pickle to h5...,{dest_h5_path}")

        train_data, train_labels = self.read_raw_pickle(root, train=True)
        val_data, val_labels = self.read_raw_pickle(root, train=False)

        f = h5py.File(dest_h5_path, mode="w")
        f.close()
        f = h5py.File(dest_h5_path, mode="a")
        f.create_dataset("data_train", data=train_data, dtype=train_data.dtype)
        f.create_dataset("labels_train", data=train_labels,
                         dtype=train_labels.dtype)

        f.create_dataset("data_val", data=val_data, dtype=val_data.dtype)
        f.create_dataset("labels_val", data=val_labels, dtype=val_labels.dtype)
        f.close()
        print("pickle2h5 done...")

    def set_images_labels(self, root, train):

        root = self.get_root_path(root)
        ##############################
        self.data, self.label_list = self.read_image_data(
            root=root, train=train)

        if self.data_ratio < 1:
            train_indices = np.arange(len(self.data))
            np.random.shuffle(train_indices, seed=666)
            selected_indices = train_indices[:int(
                len(self.data)*self.data_ratio)]
            self.data = self.data[selected_indices]
            self.label_list = self.label_list[selected_indices]
            logger.warning(
                f"using {self.data_ratio} of data, first 10 indices {selected_indices[:10]}, determinstic by seed 666")

        elif self.corruption >0:
            logger.warning(f"corrupting {self.corruption} of data")
            assert self.condition_method == 'cluster'
            if False:
                train_indices = np.arange(len(self.data))
                np.random.shuffle(train_indices, seed=666)#inplace
                self.data = self.data[train_indices]
                self.label_list = self.label_list[train_indices]

            
            corrupted_num = int(len(self.label_list) * self.corruption)
            shuffled_index = torch.randperm(corrupted_num)
            shuffled_index = shuffled_index.numpy()
            self.label_list = np.array(self.label_list)
            self.label_list[:corrupted_num] = self.label_list[shuffled_index]

        elif self.subgroup >1:
            logger.warning(f"using subgroup {self.subgroup} of data")
            assert self.condition_method == 'label'
            CLS_NUM_ORIGIN = 1000
            sub_label_counter = {k: 0 for k in range(CLS_NUM_ORIGIN)}
            def sub_label_func(label):
                factor = sub_label_counter[label]
                sub_label_counter[label] = (1 + sub_label_counter[label]) % self.subgroup #recurrent 
                return label + CLS_NUM_ORIGIN * factor
            self.label_list = [sub_label_func(label) for label in self.label_list]
            self.label_list = np.array(self.label_list)
            self.label_num = self.label_num*self.subgroup
        else:
            logger.warning(f"no tricks, using all data")

        
        self.class_hist = np.histogram(
            self.label_list, bins=len(self.label_list))
        logger.info(
            f"train={self.train}, image_num = {len(self.data)}, class_num = {len(np.unique(self.label_list))}"
        )

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(3, self.size, self.size).transpose((1, 2, 0))
        imgnp_4unsup = torch.from_numpy(np.copy(img))
        imgnp_4unsup = rearrange(imgnp_4unsup, "w h c -> c w h")
        img_4save = np.copy(img)
        img = Image.fromarray(img)

        if self.img_save_path is not None:
            img.save(os.path.join(self.img_save_path, f"{index}.png"))

        img = transforms.ToTensor()(img)
        img = normalize_to_neg_one_to_one(img)  # [0,1]->[-1,1]

        result = dict(image=img, img4unsup=imgnp_4unsup,
                      img_4save=img_4save, id=index)
        cond_dict = get_cond(
            dl=self,
            condition_method=self.condition_method,
            index=index,
            dataset_name=self.dataset_name,
        )
        result.update(cond_dict)
        return result

    def __len__(self) -> int:
        if self.debug:
            return 1200
        return len(self.data)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # feat_file = '~/data/sg_data/feat/v1_in32_feat_dino_vitb16_2022-07-06T22.h5'
    image_size = 64
    h5_file = "~/data/sg_data/cluster/v1_in32_cluster8000_dino_vits16_2022-07-06T23.h5"
    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = (
            Path(
                f"~/dataaaaaa/in{image_size}_{subset}_50k").expanduser().resolve()
        )
        ds = ImageNet_Pickle(
            root="~/data/imagenet_small",
            image_size=image_size,
            train=is_train,
            h5_file=h5_file,
            img_save_path=None,
            debug=False,
        )
        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        make_clean_dir(img_save_path)
        total_num = 50_000
        for index, d in enumerate(tqdm(dataloader)):
            if index == total_num - 1:
                break
