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
from diffusion_utils.util import (
    clip_unnormalize_to_zero_to_255,
    make_clean_dir,
)
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2


def visualize_predictions(image, pred, vis_folder, im_name):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0),
        2,
    )

    if ".jpg" in im_name:
        im_name = im_name.replace(".jpg", "")
    pltname = f"{vis_folder}/LOST_{im_name}.png"
    Image.fromarray(image).save(pltname)
    print(f"Predictions saved at {pltname}.")


class CocoDataset(Dataset):
    coco_category_ids = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
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
        root_global="~/data",
        root="~/data/coco14",
        h5_file=None,
        split="train",
        debug=False,
        is_20k=True,  # always true now
        lost_file=None,
        img_save_path=None,
        condition_method=None,
        condition=None,
        size4cluster=300,
        size4crop=224,

    ):
        super().__init__()

        ########
        if is_20k:
            split = "train"
            print("is_20k=true, override split=train......")
        else:
            raise

        self.root_global = root_global
        self.size4crop = size4crop
        self.size4cluster = size4cluster
        self.dataset_name = "coco"
        self.split_name = split
        self.train = True if split == "train" else False
        self.condition_method = condition_method
        self.condition = condition
        self.is_20k = is_20k
        self.lost_file = lost_file
        self.img_save_path = img_save_path
        self.debug = debug
        self.size = size
        self.label_num = len(self.class_names)

        self.transform = RandomScaleCrop(base_size=size4crop, resize_size=size)
        ##################

        root = Path(root).expanduser().resolve()
        logger.info(f"loading dataset from {root}, is this a ssd?")
        self.root = os.path.join(root, split)

        instances_file = os.path.join(
            root, "annotations", "instances_" + split + "2014.json"
        )
        self.json_file = json.load(open(instances_file, "r"))
        self.coco = COCO(instances_file)

        self.imgid2cats = dict()
        category_id_set = set()
        for ann in self.json_file["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            category_id_set.add(category_id)
            area = ann["area"]
            if image_id in self.imgid2cats:
                self.imgid2cats[image_id].append(category_id)
            else:
                self.imgid2cats[image_id] = [category_id]

        self.category_id_list = list(category_id_set)

        self.image_prename = "COCO_" + split + "2014_"
        self.folder_path = os.path.join(root, split + "2014")
        self.image_ids = list(self.imgid2cats.keys())

        if self.is_20k:
            print("filtering 20k...")
            with open(
                os.path.join(
                    Path(__file__).parent.resolve(
                    ), "data_files/coco_20k_filenames.txt"
                ),
                "r",
            ) as f:
                names = f.readlines()
            names = [name.strip() for name in names]
            final_image_ids = []
            for image_id in self.image_ids:
                full_path = (
                    "train2014/" + self.image_prename +
                    str(image_id).zfill(12) + ".jpg"
                )
                if full_path in names:
                    final_image_ids.append(image_id)
            self.image_ids = final_image_ids
            print(f"new 20k image_length={len(self.image_ids)}")
        else:
            print(f"image_length={len(self.image_ids)}")

        self.image_ids.sort()
        self.num = len(self.image_ids)

        set_stego(dl=self, condition_method=condition_method,
                  condition=condition)
        set_cond(self, h5_file)
        set_lostbbox_in_originsize(self, lost_file=self.lost_file)

    def get_imagename_by_index(self, index):
        return str(self.image_ids[index])

    def read_lost_pickle(self, filename):
        with open(filename, "rb") as f:
            preds_dict = pickle.load(f)
        print("Predictions saved at %s" % filename)
        return preds_dict

    def id2name(self, index):
        image_id = self.image_ids[index]
        file_name = str(image_id).zfill(12) + ".jpg"
        return file_name

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)

        for instance in target:
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in self.coco_category_ids:
                c = self.coco_category_ids.index(cat)
            else:
                raise
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(
                    np.uint8
                )
        return mask

    def load_segmask(self, img_id):
        _metadata = self.coco.loadImgs(img_id)[0]
        coco_annos = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        _seg_mask = self._gen_seg_mask(
            coco_annos, _metadata["height"], _metadata["width"]
        )
        return Image.fromarray(_seg_mask)

    def load_img(self, image_id):
        filepath = os.path.join(
            self.folder_path, self.image_prename +
            str(image_id).zfill(12) + ".jpg"
        )
        img = Image.open(filepath).convert("RGB")
        return img

    def _read_img_segmask(self, index):
        image_id = self.image_ids[index]
        image = self.load_img(image_id)
        mask = self.load_segmask(image_id)
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

    if False:
        coco_val = CocoDataset(size=128, split="train")
        dataloader = DataLoader(coco_val, batch_size=1,
                                shuffle=True, num_workers=0)
        for ii, sample in enumerate(dataloader):
            for jj in range(sample["image"].size()[0]):
                sample["image"] = rearrange(
                    sample["image"], "b c w h-> b w h c")
                sample["image"] = clip_unnormalize_to_zero_to_255(
                    sample["image"])
                img = sample["image"].numpy()
                img_tmp = img[jj].astype(np.uint8)
                plt.figure()
                plt.title("display")
                plt.subplot(111)
                plt.imshow(img_tmp)
                plt.show(block=True)
                plt.close()

            if ii == 0:
                break
    elif False:
        for subset, is_train in [("train", True), ("val", False)]:
            img_save_path = Path(
                f"~/data/coco64v2_{subset}_10k").expanduser().resolve()
            make_clean_dir(img_save_path)
            ds = CocoDataset(size=64, split=subset,
                             img_save_path=img_save_path)
            bs = 128
            dataloader = DataLoader(
                ds, batch_size=bs, shuffle=False, num_workers=0)
            total_num = 10_000
            for index, d in enumerate(tqdm(dataloader)):
                if index * bs > total_num - 1:
                    break
    else:
        for subset, is_train in [("train", True), ("val", False)]:
            img_save_path = Path(
                f"~/data11/coco64v2_{subset}").expanduser().resolve()
            make_clean_dir(img_save_path)
            lost_pickle = os.path.join(
                Path(__file__).parent.resolve(),
                f"data_files/lost/COCO20k_train/LOST-vit_small16_k/preds.pkl",
            )
            ds = CocoDataset(
                size=64,
                split=subset,
                img_save_path=None,
                is_20k=True,
                lost_file=lost_pickle,
            )
            dataloader = DataLoader(
                ds, batch_size=128, shuffle=False, num_workers=0)
            for index, d in enumerate(tqdm(dataloader)):
                pass
