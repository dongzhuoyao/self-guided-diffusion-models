import argparse
import os
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
from dataset.cityscapes27 import CityscapesDataset
from dataset.coco14_vqdiffusion import CocoDataset
from dataset.coco17stuff27 import CocoStuffDataset
from dataset.dataloader_iddpm import ImageNetDataset_iDDPM
from PIL import Image
import blobfile as bf
from einops import rearrange
from loguru import logger
import numpy as np
import torch
from dataset.ffhq_dataset_v2 import FFHQ_v2

from dataset.voc12 import VOCSegmentation

from diffusion_utils.util import make_clean_dir
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cleanfid import fid as clean_fid

TMP_DIR = "~/data/sg_fid_eval_tmp"
num_workers = 8
root_global = "~/data"


def cycle(dl):
    '''
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/8c3609a6e3c216264e110c2019e61c83dafad9f5/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L34
    '''
    while True:
        for data in dl:
            yield data


def generate_in(size=32, debug=True, shuffle=True, bs=1, total_num=50_000):

    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = (
            Path(os.path.join(TMP_DIR, f"in{size}from224_{subset}_50k"))
            .expanduser()
            .resolve()
        )
        ds = ImageNetDataset_iDDPM(
            image_size=size,
            train=is_train,
            root="~/data/imagenet",
            h5_file=None,
            img_save_path=img_save_path,
            debug=debug,
        )
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        print(img_save_path)


def generate_in_pickle_longtail(size=32, debug=True, shuffle=True, total_num=50_000):
    from dataset.imagenet_pickle import ImageNet_Pickle
    _counter = dict()
    for i in range(1000):
        _counter[i] = int(i * 50.0 / 1000)
    _counter_all = 0
    for subset, is_train in [("val", False)]:
        img_save_path = (
            Path(os.path.join(TMP_DIR, f"longtail_in{size}_{subset}"))
            .expanduser()
            .resolve()
        )
        ds = ImageNet_Pickle(
            root="~/data/imagenet_small",
            image_size=size,
            train=is_train,
            h5_file=None,
            img_save_path=None,
            debug=debug, root_global=root_global
        )
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        for index, d in enumerate(tqdm(dataloader)):
            if index == total_num - 1:
                break
            img_4save = d["img_4save"][0].cpu().numpy()
            label_id = d["label_id"].item()
            if _counter[label_id] > 0:
                img = Image.fromarray(img_4save)
                img.save(os.path.join(img_save_path, f"{index}.png"))
                _counter_all += 1
                _counter[label_id] -= 1

        print(f"image num = {_counter_all}")
        print(img_save_path)


def generate_in_pickle(size=32, debug=True, shuffle=True, bs=1, total_num=50_000):
    from dataset.imagenet_pickle import ImageNet_Pickle
    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = (
            Path(os.path.join(TMP_DIR, f"in{size}v2_{subset}_{total_num}"))
            .expanduser()
            .resolve()
        )
        ds = ImageNet_Pickle(
            root="~/data/imagenet_small",
            image_size=size,
            train=is_train,
            h5_file=None,
            img_save_path=img_save_path,
            debug=debug,
        )
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        print(img_save_path)


def generate_in_pickle_inp_original(size=32, debug=True, shuffle=True, bs=1, total_num=50_000, _version='v1'):
    from dataset.imagenet_pickle_v2 import ImageNet_Pickle_Original
    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = (
            Path(os.path.join(
                TMP_DIR, f"inp{size}originalv2_{_version}_{subset}_{total_num}"))
            .expanduser()
            .resolve()
        )
        ds = ImageNet_Pickle_Original(
            root="~/data/imagenet_official_np",
            image_size=size,
            train=is_train,
            h5_file=None,
            img_save_path=img_save_path,
            debug=debug,
        )
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        print(img_save_path)


def generate_COCO(size=64, debug=True, shuffle=True, bs=1, total_num=50_000, _version='v4'):
    REAL_DIR_4FID = str(
        Path("~/data/sg_fid_eval/coco64v3_train_10k").expanduser().resolve()
    )
    assert os.path.exists(REAL_DIR_4FID)
    assert str(size) in REAL_DIR_4FID

    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = str(
            Path(os.path.join(
                TMP_DIR, f"coco{size}{_version}_{subset}_{total_num}"))
            .expanduser()
            .resolve()
        )
        ds = CocoDataset(
            size=size, split=subset, img_save_path=img_save_path, debug=debug, root_global=root_global,
        )
        dataloader = DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        result = clean_fid.compute_fid(img_save_path, REAL_DIR_4FID)
        print(result)
        print("*" * 100)
        print(img_save_path)


def generate_CocoStuff(size=64, debug=True, compute_fid=True, shuffle=True, total_num=50_000, bs=128, version="v4"):
    if compute_fid:
        REAL_DIR_4FID = str(
            Path("~/data/sg_fid_eval/cocostuff64_train_10k_tmp").expanduser().resolve())
        assert os.path.exists(REAL_DIR_4FID)
        assert str(size) in REAL_DIR_4FID

    for subset, is_train in [("val", False), ("train", True)]:
        img_save_path = str(
            Path(os.path.join(
                TMP_DIR, f"cocostuff{size}{version}_{subset}_{total_num}"))
            .expanduser()
            .resolve()
        )
        ds = CocoStuffDataset(
            root="~/data/cocostuff27/images",
            root_coco17_annos="~/data/stego_data/cocostuff/annotations",
            size=size, split=subset, img_save_path=img_save_path, debug=debug
        )
        dataloader = DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        if compute_fid:
            result = clean_fid.compute_fid(img_save_path, REAL_DIR_4FID)
            print(result)
        print("*" * 100)
        print(img_save_path)


def generate_cityscapes(size=64, debug=True, compute_fid=False, shuffle=True, bs=128, total_num=10_000):
    if compute_fid:
        REAL_DIR_4FID = str(
            Path("~/data/sg_fid_eval/cocostuff64_train_10k_tmp").expanduser().resolve())
        assert os.path.exists(REAL_DIR_4FID)
        assert str(size) in REAL_DIR_4FID

    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = str(
            Path(os.path.join(
                TMP_DIR, f"cityscapes{size}_{subset}_{total_num}"))
            .expanduser()
            .resolve()
        )
        ds = CityscapesDataset(
            root="~/data/cs320",
            size=size, split=subset, img_save_path=img_save_path, size4cluster=240, debug=debug
        )
        dataloader = DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        if compute_fid:
            result = clean_fid.compute_fid(img_save_path, REAL_DIR_4FID)
            print(result)
        print("*" * 100)
        print(img_save_path)


def generate_VOC(size=64, debug=True, shuffle=True, bs=128, total_num=10_000, _version='v3'):
    REAL_DIR_4FID = str(
        Path("~/data/sg_fid_eval/voc64v2_train_10k").expanduser().resolve()
    )
    assert os.path.exists(REAL_DIR_4FID)
    assert str(size) in REAL_DIR_4FID

    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = str(
            Path(os.path.join(
                TMP_DIR, f"voc{size}{_version}_{subset}_{total_num}"))
            .expanduser()
            .resolve()
        )
        ds = VOCSegmentation(
            root="~/data/pascalvoc12_07/VOCdevkit/VOC2012",
            size=size,
            split=subset,
            img_save_path=img_save_path,
            debug=debug,
        )
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers)
        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1

        result = clean_fid.compute_fid(img_save_path, REAL_DIR_4FID)
        print(result)
        print("*" * 100)
        print(img_save_path)


def generate_ffhq(size=64, debug=True, shuffle=True, bs=128, total_num=10_000):

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # feat_file = '~/data/sg_data/feat/v1_in32_feat_dino_vitb16_2022-07-06T22.h5'
    h5_file = "~/data/sg_data/cluster/v1_in32_cluster8000_dino_vits16_2022-07-06T23.h5"
    for subset, is_train in [("train", True), ("val", False)]:
        img_save_path = (
            Path(os.path.join(TMP_DIR, f"ffhq{size}_{subset}_10k"))
            .expanduser()
            .resolve()
        )
        ds = FFHQ_v2(
            root="~/data/ffhq/thumbnails128x128",
            split=is_train,
            img_save_path=img_save_path,
            size=size,
            debug=debug,
        )
        dataloader = DataLoader(
            ds, batch_size=1, shuffle=shuffle, num_workers=num_workers)

        make_clean_dir(img_save_path)

        sample_iter = cycle(dataloader)
        global_id = 0
        for batch_id in tqdm(range((total_num//bs)+1)):
            data_dict = next(sample_iter)
            img_saves = data_dict['img_save']
            for img_save in img_saves:
                img = Image.fromarray(img_save.cpu().numpy())
                img.save(os.path.join(img_save_path, f"{global_id}.png"))
                global_id += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ds",
        type=str,
        default="inp_lt",
        choices=["inp", "inp_original", "inp_lt", "in", "coco",
                 "cocostuff", "voc", "cityscapes",  "ffhq"],
        help="the output image",
    )
    p.add_argument("--size", type=int, default=64, help="the output image")
    p.add_argument("--debug", type=int, default=0, help="the output image")
    args = p.parse_args()

    if args.ds in ["inp"]:
        generate_in_pickle(size=args.size, debug=args.debug)
    elif args.ds in ["inp_original"]:
        generate_in_pickle_inp_original(size=args.size, debug=args.debug)
    elif args.ds in ["inp_lt"]:
        generate_in_pickle_longtail(size=args.size, debug=args.debug)
    elif args.ds in ["in"]:
        generate_in(size=args.size, debug=args.debug)
    elif args.ds in ["coco"]:
        generate_COCO(size=args.size, debug=args.debug)
    elif args.ds in ["cocostuff"]:
        generate_CocoStuff(size=args.size, debug=args.debug)
    elif args.ds in ["cityscapes"]:
        generate_cityscapes(size=args.size, debug=args.debug)
    elif args.ds in ["ffhq"]:
        generate_ffhq(size=args.size, debug=args.debug)
    elif args.ds in ["voc"]:
        generate_VOC(size=args.size, debug=args.debug)
