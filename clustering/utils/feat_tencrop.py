import colorsys
import json
import os
from pathlib import Path
import random
import cv2
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from clustering.utils.common_utils import has_attention_map

from dataset.ds_utils.dataset_common_utils import ds_has_label_info, get_train_val_dl
from self_sl.ssl_backbone import get_ssl_backbone
import h5py
import os

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torchvision
import numpy as np
from PIL import Image
from diffusion_utils.taokit.color_util import random_colors



def extract_feat_tencrop(
    feat_name,
    h5py_path,
    bs,
    dataset_name,
    image_size,
    version,
    debug=False,
    dataset_root=None,
    attention_map=False,
    is_grey=False,
    _crop_num=10
):

    if debug:
        h5py_path = h5py_path.replace(".h5", "debug.h5")
    h5py_path = str(Path(h5py_path).expanduser().resolve())
    logger.warning(h5py_path)
    json_name = h5py_path.replace(".h5", ".json")

    _feat_backbone = get_ssl_backbone(
        feat_name, dataset_name, image_size, is_grey=is_grey)
    dataloader_train, dataloader_val = get_train_val_dl(
        dataset_name=dataset_name,
        bs=bs,
        image_size=image_size,
        debug=debug,
        dataset_root=dataset_root,
    )

    f = h5py.File(h5py_path, mode="w")
    f.close()
    f = h5py.File(h5py_path, mode="a")
    f.create_dataset(
        "train",
        shape=(len(dataloader_train.dataset),
               _crop_num,  _feat_backbone.feat_dim),
        dtype="float32",
    )
    f.create_dataset(
        "val",
        shape=(len(dataloader_val.dataset),
               _crop_num,  _feat_backbone.feat_dim),
        dtype="float32",
    )
    if ds_has_label_info(dataset_name):
        f.create_dataset(
            "train_labels", shape=(len(dataloader_train.dataset),), dtype="float32"
        )
        f.create_dataset(
            "val_labels", shape=(len(dataloader_val.dataset),), dtype="float32"
        )

    if True:
        dset = f.create_dataset("all_attributes", (1,))
        dset.attrs["dataset_name"] = dataset_name
        dset.attrs["feat_from"] = feat_name
        dset.attrs["feat_dim"] = _feat_backbone.feat_dim
        dset.attrs["version"] = version
        dset.attrs["is_grey"] = int(is_grey)

    id2name_dict = dict()
    name2id_dict = dict()
    for split, split_labels, dl in [
        ("train", "train_labels", dataloader_train),
        ("val", "val_labels", dataloader_val),
    ]:
        for batch_id, batch_data in enumerate(tqdm(dl)):
            with torch.no_grad():
                batch_transformed = _feat_backbone.transform_batch(
                    batch_data["img4unsup"].to("cuda"))

                backbone_dict = _feat_backbone.batch_encode_feat(
                    batch_transformed)
                feat = backbone_dict["feat"]

                ids = batch_data["id"].cpu().numpy()
                f[split][ids] = feat.detach().cpu().numpy()

                for id in ids:
                    _name = dl.dataset.id2name(id)
                    id2name_dict[str(id)] = _name
                    name2id_dict[_name] = str(id)

                if ds_has_label_info(dataset_name):
                    labels = batch_data["label"].argmax(-1).cpu().numpy()
                    f[split_labels][ids] = labels

                if "simclr" in feat_name:
                    assert feat.min() >= 0, feat.min()

        for i in range(len(dl.dataset)):
            assert np.linalg.norm(f[split][i]) > 0, i

    with open(json_name, "w") as outfile:
        json.dump(dict(id2name=id2name_dict, name2id=name2id_dict), outfile)
        print(f"dump json file, {json_name}")

    f.close()
    logger.warning(f"saving {h5py_path}")
