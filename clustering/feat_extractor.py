import git
import argparse
from datetime import datetime
from loguru import logger

from PIL import Image
import os
from clustering.utils.feat_nospatial import extract_feat
from clustering.utils.feat_tencrop import extract_feat_tencrop
import torch

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

torch.set_num_threads(4)  # can drastically boost the speed in IvI cluster.

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    return img


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feat",
        type=str,
        default="timm_resnet50_random",
        choices=[
            "simclr",

            "timm_resnet50",
            "timm_resnet50_random"
            "timm_densenet121",
            "timm_vgg19",
            "timm_vit_small_patch16_224",
            "timm_vit_base_patch16_224",
            "timm_vit_base_patch16_224_in21k",
            "timm_vit_base_patch16_224_dino",

            "mae_vitbase",
            "mae_vitlarge",
            "dino_vits16",
            "dino_vits8",
            "dino_vitb16",
            "dino_xcit_m24_p8",
            "dino_vitb8",
            "dino_resnet50",

            "msn_vitb16",
            "msn_vitl16",
            "msn_vits16",
            "msn_vitb4",
            "msn_vitl7"
        ],
        help="the output image",
    )
    p.add_argument(
        "--ds",
        type=str,
        default="in32p",
        choices=[
            "cifar10",
            "in32p",
            "in32p_original",
            "in64p",
            "in224",
            "coco64",
            'cocostuff27_64'
            "voc64",
            "ffhq64",
            "ffhq128",
        ],
        help="the output image",
    )
    p.add_argument(
        "--v",
        type=str,
        # v2: add label info, v3. add mapping info. v4. update img4sup(use original large image in voc and coco)
        default="v4",
        help="the output image",
    )
    p.add_argument("--bs", type=int, default=8, help="the output image")
    p.add_argument("--image_size", type=int,
                   default=32, help="the output image")
    p.add_argument("--debug", type=int, default=1, help="the output image")
    p.add_argument(
        "--patch", type=int, default=0, help="the output image"
    )
    p.add_argument(
        "--vis", type=int, default=0, help="the output image"
    )
    p.add_argument(
        "--is_grey", type=int, default=0, help="the output image"
    )
    p.add_argument(
        "--att", type=int, default=0, help="attention map vis"
    )
    p.add_argument(
        "--tencrop", type=int, default=0, help="the output image"
    )
    p.add_argument(
        "--resize", type=int, default=4, help="the output image"
    )

    p.add_argument(
        "--ds_root",
        type=str,
        default=None,  # v2: add label info
        help="the output image",
    )
    p.add_argument(
        "--h5_root",
        type=str,
        default=None,  # v2: add label info
        help="the output image",
    )
    args = p.parse_args()

    dataset_name = args.ds
    feat_name = args.feat

    assert str(args.image_size) in str(args.ds)

    time_str = datetime.now().isoformat(timespec="hours")
    h5py_path = f"~/data/sg_data/feat/{args.v}_{dataset_name}_feat_{feat_name}_grey{args.is_grey}_{time_str}_{sha[:7]}.h5"

    if args.h5_root is not None:
        h5py_path = h5py_path.replace("~/data/sg_data/feat", args.h5_root)

    if args.tencrop:
        extract_feat_tencrop(
            dataset_name=dataset_name,
            bs=args.bs,
            h5py_path=h5py_path,
            version=args.v,
            image_size=args.image_size,
            feat_name=feat_name,
            debug=args.debug,
            dataset_root=args.ds_root,
            attention_map=args.att,
            is_grey=args.is_grey,
        )
    else:
        extract_feat(
            dataset_name=dataset_name,
            bs=args.bs,
            h5py_path=h5py_path,
            version=args.v,
            image_size=args.image_size,
            feat_name=feat_name,
            debug=args.debug,
            dataset_root=args.ds_root,
            attention_map=args.att,
            is_grey=args.is_grey,
        )
