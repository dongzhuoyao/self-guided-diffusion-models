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


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def display_instances(
    image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(fname)
    plt.show()
    print(f"{fname} saved.")
    fig.clf()
    plt.close()
    return


def vis_attention(
    attentions, threshold, patch_size, output_dir, img, w_featmap, h_featmap
):
    nh, _ = attentions.shape
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            torch.nn.functional.interpolate(
                th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        torch.nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    os.makedirs(output_dir, exist_ok=True)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(output_dir, "img.png"),
    )
    if False:
        for j in range(nh):
            fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format="png")
            print(f"{fname} saved.")

    # save attentions heatmaps
    attn_all = sum(attention * 1 / len(attentions) for attention in attentions)
    plt.imsave(
        fname=os.path.join(output_dir, "attn-allhead.png"),
        arr=attn_all,
        cmap="inferno",
        format="png",
    )
    plt.show()
    plt.close()

    if threshold is not None:
        th_attn_all = sum(_th_attn * 1 / len(th_attn) for _th_attn in th_attn)
        image = skimage.io.imread(os.path.join(output_dir, "img.png"))
        display_instances(
            image,
            th_attn_all,
            fname=os.path.join(
                output_dir, "mask_th" + str(threshold) + "_headall" + ".png"
            ),
            blur=False,
        )

    if False:
        if threshold is not None:
            image = skimage.io.imread(os.path.join(output_dir, "img.png"))
            for j in range(nh):
                display_instances(
                    image,
                    th_attn[j],
                    fname=os.path.join(
                        output_dir,
                        "mask_th" + str(threshold) + "_head" + str(j) + ".png",
                    ),
                    blur=False,
                )


def vis_attention_wrapper(_feat_backbone, img, batch_transformed, attention_map):
    backbone_dict = _feat_backbone.batch_encode_feat(
        batch_transformed, attention_map=True
    )
    feat = backbone_dict["feat"]
    attentions = backbone_dict["attentions"]
    bs, nh, _, _ = attentions.shape  # number of head
    # we keep only the output patch attention
    attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)
    if attention_map:
        attentions = attentions[0]  # only vis first image
        patch_size = 16
        w_featmap = batch_transformed.shape[-2] // patch_size
        h_featmap = batch_transformed.shape[-1] // patch_size
        img = torch.nn.functional.interpolate(img, size=224)[0].float()
        vis_attention(
            attentions=attentions,
            threshold=0.5,
            output_dir=Path(__file__).parent.resolve(),
            img=img,
            patch_size=patch_size,
            w_featmap=w_featmap,
            h_featmap=h_featmap,
        )
    return attentions


def extract_feat(
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
        shape=(len(dataloader_train.dataset), _feat_backbone.feat_dim),
        dtype="float32",
    )
    f.create_dataset(
        "val",
        shape=(len(dataloader_val.dataset), _feat_backbone.feat_dim),
        dtype="float32",
    )
    if ds_has_label_info(dataset_name):
        f.create_dataset(
            "train_labels", shape=(len(dataloader_train.dataset),), dtype="float32"
        )
        f.create_dataset(
            "val_labels", shape=(len(dataloader_val.dataset),), dtype="float32"
        )

    if has_attention_map(feat_name):
        head_num = 6
        patch_nums = 196
        f.create_dataset(
            "train_attentions",
            shape=(len(dataloader_train.dataset), head_num, patch_nums),
            dtype="float32",
        )
        f.create_dataset(
            "val_attentions",
            shape=(len(dataloader_val.dataset), head_num, patch_nums),
            dtype="float32",
        )
        logger.warning("creating dataset attentions..")
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
                    batch_data["img4unsup"].to("cuda")
                )
                if has_attention_map(feat_name):
                    attentions = vis_attention_wrapper(
                        _feat_backbone=_feat_backbone,
                        img=batch_data["img4unsup"].clone(),
                        batch_transformed=batch_transformed,
                        attention_map=attention_map,
                    )

                backbone_dict = _feat_backbone.batch_encode_feat(
                    batch_transformed)
                feat = backbone_dict["feat"]

                ids = batch_data["id"].cpu().numpy()
                f[split][ids] = feat.detach().cpu().numpy()

                for id in ids:
                    _name = dl.dataset.id2name(id)
                    id2name_dict[str(id)] = _name
                    name2id_dict[_name] = str(id)

                if has_attention_map(feat_name):
                    f[f"{split}_attentions"][ids] = attentions.detach().cpu().numpy()

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
