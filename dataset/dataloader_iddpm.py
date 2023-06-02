import os
from pathlib import Path
from PIL import Image
import blobfile as bf
from einops import rearrange
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import torch.nn.functional as F
from tqdm import tqdm
from dataset.ds_utils.unsupervised_cond import get_cond, set_cond
from diffusion_utils.util import make_clean_dir


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageNetDataset_iDDPM(Dataset):
    def __init__(
        self,
        image_size,
        root,
        train=True,
        h5_file=None,
        condition_method=None,
        condition=None,
        debug=False,
        img_save_path=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()

        self.TOTAL_CLASS_NUM = 1000
        self.debug = debug
        self.img_save_path = img_save_path
        self.train = train
        self.split_name = "train" if self.train else "val"
        root = Path(root).expanduser().resolve()
        logger.info(f"loading imagenet from {root}, is this a ssd?")
        # set image_paths, and classes
        if train:
            data_dir = os.path.join(root, "train")
        else:
            data_dir = os.path.join(root, "val")
        if not data_dir:
            raise ValueError("unspecified data directory")
        image_paths = _list_image_files_recursively(data_dir)

        class_names = [path.split("/")[-2] for path in image_paths]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

        self.resolution = image_size
        self.images = image_paths[shard:][::num_shards]
        self.label_list = None if classes is None else classes[shard:][::num_shards]
        self.label_num = len(set(self.label_list))
        logger.info(
            f"train={self.train}, image_num = {len(self.images)}, class_num = {self.label_num}"
        )

        self.class_hist = np.histogram(self.label_list, bins=self.label_num)

        #########################
        self.dataset_name = "in_from224"
        self.condition_method = condition_method
        self.condition = condition
        set_cond(self, h5_file)
        #########################

    def id2name(self, index):
        file_name = os.path.basename(self.images[index])
        return file_name

    def __len__(self):
        if self.debug:
            return 1200
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]

        if self.img_save_path is not None:
            pil_to_save = Image.fromarray(arr)
            pil_to_save.save(os.path.join(self.img_save_path, f"{index}.png"))

        imgnp_4unsup = rearrange(torch.from_numpy(np.copy(arr)), "w h c -> c w h")

        arr = arr.astype(np.float32) / 127.5 - 1  # [0, 255]->[-1, 1]
        image = np.transpose(arr, [2, 0, 1])  # [C,W,H]

        result = dict(image=image, img4unsup=imgnp_4unsup, id=index)
        cond_dict = get_cond(
            dl=self,
            condition_method=self.condition_method,
            index=index,
            dataset_name=self.dataset_name,
        )
        result.update(cond_dict)
        return result


if __name__ == "__main__":
    # ds = ImageNetDataset_iDDPM(image_size=32,train=False, root='~/data/imagenet')
    # dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = Path(f"~/data/in32from224_{subset}_50k").expanduser().resolve()
        ds = ImageNetDataset_iDDPM(
            root="~/data/imagenet",
            image_size=32,
            train=is_train,
            img_save_path=img_save_path,
            debug=False,
        )
        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
        make_clean_dir(img_save_path)
        total_num = 50_000
        for index, d in enumerate(tqdm(dataloader)):
            if index == total_num - 1:
                break
