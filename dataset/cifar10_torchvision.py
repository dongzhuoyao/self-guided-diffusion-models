import os.path
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torchvision
from PIL import Image
from einops import rearrange
from torch.utils.data import DataLoader

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import h5py
from loguru import logger

from diffusion_utils.util import normalize_to_neg_one_to_one, make_clean_dir


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            feat_file=None,
            cluster_file=None,
            centroid_file=None,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            debug: bool = False,
            img_save_path=None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_name = 'cifar10'
        self.train = train  # training set or test set
        self.split_name = 'train' if self.train else 'val'

        if feat_file is not None:
            self.feat_file = Path(feat_file).expanduser().resolve()
        else:
            self.feat_file = None
        if cluster_file is not None:
            self.cluster_file = Path(cluster_file).expanduser().resolve()
        else:
            self.cluster_file = None
        assert not (
            self.cluster_file is not None and self.feat_file is not None)
        self.feat_list = None
        self.cluster_list = None
        if self.cluster_file is not None:
            self.cluster_k = h5py.File(self.cluster_file, 'r')[
                'all_attributes'].attrs['cluster_k']
            self.cluster_list = h5py.File(self.cluster_file, 'r')[
                self.split_name]  # [N, 1]
            self.cluster_hist = np.histogram(
                self.cluster_list, bins=self.cluster_k)

        ##################

        self.img_save_path = img_save_path
        self.debug = debug

        if self.transform is None:
            self.transform = transforms.ToTensor()

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.debug:
            self.data, self.targets = self.data[:1200], self.targets[:1200]

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i,
                             _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img4unsup = rearrange(torch.from_numpy(np.copy(img)), 'w h c -> c w h')
        assert img4unsup.dtype == torch.uint8
        img = Image.fromarray(img)
        if self.img_save_path is not None:
            img.save(os.path.join(self.img_save_path, f'{index}.png'))
            print(f'save {index}')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = normalize_to_neg_one_to_one(img)  # [0,1]->[-1,1]
        label_onehot = F.one_hot(torch.tensor(target), num_classes=10)

        result = dict(image=img, label=label_onehot,
                      img4unsup=img4unsup, id=index)

        # raise
        raise
        return result

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


if __name__ == '__main__':
    if False:  # generate images
        for subset, is_train in [('train', True), ('val', False)]:
            img_save_path = Path(
                f'~/data/cifar10_{subset}').expanduser().resolve()
            ds = CIFAR10(root='~/data', train=is_train,
                         img_save_path=img_save_path)
            dataloader = DataLoader(
                ds, batch_size=1, shuffle=True, num_workers=0)
            make_clean_dir(img_save_path)
            for d in dataloader:
                pass
    else:
        ds = CIFAR10(root='~/data', train=True,
                     feat_file='~/data/sg_data/feat/cifar10_feat_vitbase_2022-07-04T23.h5')
        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        for d in dataloader:
            print(d['feat'].shape)
