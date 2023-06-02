import os
import random
from PIL import Image
import torch
import numpy as np
from einops import rearrange
from dataset.ds_utils.unsupervised_cond import get_cond

from dataset.ds_utils.unsupervised_cond_originsize import get_lostbbox_originsize
from diffusion_utils.util import normalize_to_neg_one_to_one
from loguru import logger
from torchvision import transforms
from pathlib import Path


class RandomScaleCrop(object):
    def __init__(self, base_size, resize_size, fill=0):
        self.base_size = base_size
        self.crop_size = base_size
        self.resize_size = resize_size
        self.fill = fill

    def __call__(self, img, mask, bboxmask=None, stegomask=None):
        # random scale (short edge)
        # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        short_size = random.randint(
            int(self.base_size * 1.05), int(self.base_size * 1.25)
        )  
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((ow, oh), Image.NEAREST)

        if bboxmask is not None:
            bboxmask = bboxmask.resize((ow, oh), Image.NEAREST)
        if stegomask is not None:
            stegomask = stegomask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            raise  # dongzhuoyao, should not possible
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(
                0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if mask is not None:
            mask = mask.crop(
                (x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if bboxmask is not None:
            bboxmask = bboxmask.crop(
                (x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if stegomask is not None:
            stegomask = stegomask.crop(
                (x1, y1, x1 + self.crop_size, y1 + self.crop_size)
            )

        img = torch.from_numpy(
            np.array(img.resize((self.resize_size, self.resize_size)))
        )

        if mask is not None:
            mask = torch.from_numpy(
                np.array(
                    mask.resize(
                        (self.resize_size, self.resize_size), resample=Image.NEAREST
                    )
                )
            )
        if bboxmask is not None:
            bboxmask = torch.from_numpy(
                np.array(
                    bboxmask.resize(
                        (self.resize_size, self.resize_size), resample=Image.NEAREST
                    )
                )
            )
        if stegomask is not None:
            stegomask = torch.from_numpy(
                np.array(
                    stegomask.resize(
                        (self.resize_size, self.resize_size), resample=Image.NEAREST
                    )
                )
            )
        img = rearrange(img, "w h c -> c w h")
        return img, mask, bboxmask, stegomask


def segmask_to_onehotmask(dl, segmask):
    segmask[segmask == 255] = 0  # set 255 as bg
    if hasattr(dl, 'attr182_to_attr27'):
        segmask_clone = segmask.clone()
        attr_list = torch.unique(segmask_clone).long()
        for _attr in attr_list.tolist():
            segmask_clone[segmask == _attr] = dl.attr182_to_attr27[_attr]
        segmask = segmask_clone

    segmask = torch.nn.functional.one_hot(
        segmask.long(), num_classes=dl.label_num)
    segmask = rearrange(segmask, "w h c-> c w h")
    return segmask


def stego_to_onehotmask(segmask, num_classes):
    segmask[segmask == 255] = 0  # set 255 as bg
    segmask = torch.nn.functional.one_hot(
        segmask.long(), num_classes=num_classes)
    segmask = rearrange(segmask, "w h c-> c w h")
    return segmask


def stegomask_to_attr_nhot(stegomask, stego_k):
    attr_list = torch.unique(stegomask).long()

    attr_nhot = torch.nn.functional.one_hot(
        attr_list, num_classes=stego_k)
    assert len(attr_nhot.shape) == 2
    attr_nhot = attr_nhot.sum(0)
    return attr_nhot


def segmask_to_attr_nhot(dl, segmask):
    attr_list = torch.unique(segmask).long()
    if hasattr(dl, 'attr182_to_attr27'):
        attr_list_clone = attr_list.clone()
        for _attr in attr_list.tolist():
            attr_list_clone[attr_list == _attr] = dl.attr182_to_attr27[_attr]
        attr_list = attr_list_clone

    attr_nhot = torch.nn.functional.one_hot(
        attr_list, num_classes=dl.label_num)
    assert len(attr_nhot.shape) == 2
    attr_nhot = attr_nhot.sum(0)
    return attr_nhot


def get_lostbboxmask(dl, segmask, condition_method, index):
    bbox = get_lostbbox_originsize(
        dl=dl, condition_method=condition_method, index=index
    )
    if bbox is not None:
        bboxmask = np.zeros_like(np.asarray(segmask))
        bboxmask[bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        bboxmask = Image.fromarray(bboxmask)
    else:
        bboxmask = None

    return bboxmask, bbox


def set_stego(dl, condition_method, condition):
    if condition_method in ["clusterlayout"]:
        if condition.clusterlayout.how == "stego":
            dl.is_stego = True
            dl.stego_mask_dir = str(
                Path(condition.clusterlayout.stego_dir).expanduser().resolve())
            dl.stego_cluster_num = condition.clusterlayout.stego_k
        else:
            dl.is_stego = False
            dl.stego_mask_dir = None
            dl.stego_cluster_num = -1

    elif condition_method in ["stegoclusterlayout"]:
        if condition.stegoclusterlayout.how == "stego":
            dl.is_stego = True
            dl.stego_mask_dir = str(
                Path(condition.stegoclusterlayout.stego_dir).expanduser().resolve())
            dl.stego_cluster_num = condition.stegoclusterlayout.stego_k
        else:
            dl.is_stego = False
            dl.stego_mask_dir = None
            dl.stego_cluster_num = -1

    elif condition_method == "layout":
        if condition.layout.how == "stego":
            dl.is_stego = True
            dl.stego_mask_dir = str(
                Path(condition.layout.stego_dir).expanduser().resolve())
            dl.stego_cluster_num = condition.layout.stego_k
        else:
            dl.is_stego = False
            dl.stego_mask_dir = None
            dl.stego_cluster_num = -1
    else:
        dl.is_stego = False
        dl.stego_mask_dir = None
        dl.stego_cluster_num = -1

    if dl.is_stego:
        logger.warning(
            f"set is_stego to True, {dl.stego_mask_dir}, k={dl.stego_cluster_num}"
        )

    else:
        logger.warning("set is_stego to False")


def get_item_complex(dl, index):
    result = dict()
    image, segmask = dl._read_img_segmask(index)

    img4unsup = transforms.ToTensor()(image)*255.0
    img4unsup = transforms.Resize(
        (dl.size4cluster, dl.size4cluster))(img4unsup)

    stegomask = Image.open(dl.stego_paths[index]) if dl.is_stego else None
    lostbboxmask, lostbbox4d = get_lostbboxmask(
        dl=dl,
        segmask=segmask,
        condition_method=dl.condition_method,
        index=index,
    )

    image, segmask, lostbboxmask, stegomask = dl.transform(
        image, segmask, bboxmask=lostbboxmask, stegomask=stegomask
    )

    if dl.img_save_path is not None:
        if False:
            image_pil = Image.fromarray(
                image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            )
            image_pil.save(os.path.join(
                dl.img_save_path, f"{index}.png"))
        else:
            result.update(
                dict(img_save=image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)))

    if lostbboxmask is not None:
        lostbboxmask = rearrange(lostbboxmask, "w h-> 1 w h")
        result.update(dict(lostbboxmask=lostbboxmask))
        # visualize_predictions(image=np.asarray(image), pred=bbox, vis_folder='./lost_vis_voc2107',im_name=f'{index}')

    if stegomask is not None:
        stego_attr = stegomask_to_attr_nhot(
            stegomask=stegomask, stego_k=dl.stego_cluster_num)
        stegomask = stego_to_onehotmask(
            stegomask, num_classes=dl.stego_cluster_num)
        result.update(dict(stegomask=stegomask, stego_attr=stego_attr))

    image = normalize_to_neg_one_to_one(image / 255.0)  # [0,1]->[-1,1]

    if segmask is not None:
        segmask_onehot = segmask_to_onehotmask(dl, segmask)
        attr_nhot = segmask_to_attr_nhot(dl, segmask)
        result.update(dict(segmask=segmask_onehot,
                      attr=attr_nhot))

    else:
        attr_nhot = None

    result.update(
        dict(
            image=image,
            img4unsup=img4unsup,
            id=index,
        )
    )

    cond_dict = get_cond(
        dl=dl,
        condition_method=dl.condition_method,
        index=index,
        mask=segmask,
        attr_nhot=attr_nhot,
        dataset_name=dl.dataset_name,
    )
    result.update(cond_dict)

    return result
