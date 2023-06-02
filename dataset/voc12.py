# https://discuss.pytorch.org/t/loading-voc-2012-dataset-with-dataloaders/805/7
from __future__ import print_function, division
import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from loguru import logger
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F

from dataset.ds_utils.unsupervised_cond import set_cond
from dataset.ds_utils.unsupervised_cond_originsize import (
    set_lostbbox_in_originsize,
)
from dataset.transforms.complex_ds_common_util import (
    get_item_complex,
    set_stego,
    RandomScaleCrop,
)
from dynamic.attention_ldm import log


class VOCSegmentation(Dataset):
    LABELS = [  # https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    class_names_4wandb = dict()
    for i, l in enumerate(LABELS):
        class_names_4wandb[i] = l

    def __init__(
        self,
        root,
        split,
        size,
        root_global="~/data",
        h5_file=None,
        debug=False,
        lost_file=None,
        img_save_path=None,
        condition_method=None,
        condition=None,
        size4cluster=300,
        size4crop=224,
    ):
        super().__init__()

        ########
        self.dataset_name = "voc"
        self.size4crop = size4crop
        self.size4cluster = size4cluster
        self._base_dir = Path(root).expanduser().resolve()

        self.size = size
        self.root_global = root_global
        self.label_num = 21
        self.debug = debug
        self.img_save_path = img_save_path
        self.condition_method = condition_method
        self.condition = condition
        self.h5_file = h5_file
        self.lost_file = lost_file
        self.split_name = split

        self.transform = RandomScaleCrop(
            base_size=self.size4crop, resize_size=size)

        set_stego(dl=self, condition_method=condition_method,
                  condition=condition)
        set_cond(self, h5_file)
        set_lostbbox_in_originsize(self, lost_file=self.lost_file)

        self.set_image_list()
        logger.warning(f"Number of images in {split}: {len(self.image_paths)}")

    def set_image_list(
        self,
    ):
        self._image_dir = os.path.join(self._base_dir, "JPEGImages")
        self._segmask_dir = os.path.join(
            self._base_dir, "SegmentationClassAug")

        self.image_ids, self.image_paths, self.mask_paths = [], [], []
        self.stego_paths = []

        for iddd, image_name in enumerate(os.listdir(self._image_dir)):
            image_name = image_name.replace(".jpg", "").replace(".png", "")
            _image_path = os.path.join(self._image_dir, image_name + ".jpg")
            _mask_path = os.path.join(self._segmask_dir, image_name + ".png")
            assert os.path.isfile(_image_path), _image_path
            if not os.path.isfile(_mask_path):
                #logger.warning(f"Mask not found, skip: {_mask_path}")
                continue

            self.image_ids.append(image_name)
            self.image_paths.append(_image_path)
            self.mask_paths.append(_mask_path)

            if self.is_stego:
                self.stego_paths.append(
                    os.path.join(self.stego_mask_dir, image_name + ".png")
                )
        logger.warning(
            f"Number of images in {self.split_name}: {len(self.image_paths)}, old len: {iddd}")

    def get_imagename_by_index(self, index):
        return str(self.image_ids[index])

    def id2name(self, index):
        file_name = self.image_ids[index] + ".jpg"
        return file_name

    def __len__(self):
        return len(self.image_paths)

    def _read_img_segmask(self, index):
        _img = Image.open(self.image_paths[index]).convert("RGB")
        _mask = Image.open(self.mask_paths[index])
        return _img, _mask

    def __getitem__(self, index):
        return get_item_complex(dl=self, index=index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    lost_file = os.path.join(
        Path(__file__).parent.resolve(), f"data_files/lost/voc1207/preds.pkl"
    )
    voc_train = VOCSegmentation(size=128, lost_file=lost_file, split="train")
    bs = 5
    dataloader = DataLoader(voc_train, batch_size=bs,
                            shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(bs):
            img = sample["image"].numpy()
            gt = sample["label"].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset="pascal")
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # img_tmp *= (0.229, 0.224, 0.225)
            # img_tmp += (0.485, 0.456, 0.406)
            # img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title("display")
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
