from matplotlib import image
import numpy as np


from torch.utils.data import DataLoader


def normalize_featnp(features):
    assert len(features.shape) == 1
    result = features / np.linalg.norm(features, axis=0, keepdims=True)
    return result


def ds_has_label_info(dataset_name):
    if dataset_name.startswith("coco"):
        return False
    elif dataset_name.startswith("voc"):
        return False
    elif dataset_name.startswith("ffhq"):
        return False
    else:
        return True


def skip_id2name(dataset_name):
    if "ffhq" in dataset_name:
        return True
    else:
        return False


def need_to_upsample256(dataset_name):
    if dataset_name in ["ffhq64", "coco64", "voc64", "cocostuff64"]:
        return True
    else:
        return False


def get_train_val_dl(
    dataset_name,
    bs,
    image_size,
    dataset_root=None,
    debug=False,
    num_workers=3,
    shuffle=False,
    root_global='~/data'
):
    def get_torch_loader(_ds):
        return DataLoader(_ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers)

    if dataset_name == "cifar10":
        from dataset.cifar10_torchvision import CIFAR10

        root = "~/data" if dataset_root is None else dataset_root
        ds_train = CIFAR10(root=root, train=True, debug=debug)
        ds_val = CIFAR10(root=root, train=False, debug=debug)

    elif dataset_name == "in32p":
        from dataset.imagenet_pickle import ImageNet_Pickle

        root = "~/data/imagenet_small" if dataset_root is None else dataset_root
        ds_train = ImageNet_Pickle(
            root=root, train=True, image_size=32, debug=debug, root_global=root_global)
        ds_val = ImageNet_Pickle(
            root=root, train=False, image_size=32, debug=debug, root_global=root_global)

    elif dataset_name == "in32p_original":
        from dataset.imagenet_pickle_v2 import ImageNet_Pickle_Original

        root = "~/data/imagenet_official_np" if dataset_root is None else dataset_root
        root_original = "~/data/imagenet_official"
        ds_train = ImageNet_Pickle_Original(
            root=root, train=True, image_size=32, debug=debug, get_backbone_feat=True, root_original=root_original, root_global=root_global)
        ds_val = ImageNet_Pickle_Original(
            root=root, train=False, image_size=32, debug=debug, get_backbone_feat=True, root_original=root_original, root_global=root_global)

    elif dataset_name == "in64p":
        from dataset.imagenet_pickle import ImageNet_Pickle

        root = "~/data/imagenet_small" if dataset_root is None else dataset_root
        ds_train = ImageNet_Pickle(
            root=root, train=True, image_size=64, debug=debug, root_global=root_global)
        ds_val = ImageNet_Pickle(
            root=root, train=False, image_size=64, debug=debug, root_global=root_global)

    elif dataset_name == "in32v2":
        from dataset.dataloader_iddpm import ImageNetDataset_iDDPM

        root = "~/data/imagenet" if dataset_root is None else dataset_root
        ds_train = ImageNetDataset_iDDPM(
            image_size=32, root=root, train=True, debug=debug, root_global=root_global)
        ds_val = ImageNetDataset_iDDPM(
            image_size=32, root=root, train=False, debug=debug, root_global=root_global)

    elif dataset_name == "in224":
        root = "~/data/imagenet" if dataset_root is None else dataset_root
        ds_train = ImageNetDataset_iDDPM(
            image_size=224, root=root, train=True, debug=debug, root_global=root_global)
        ds_val = ImageNetDataset_iDDPM(
            image_size=224, root=root, train=False, debug=debug, root_global=root_global)

    elif dataset_name == "coco64":
        from dataset.coco14_vqdiffusion import CocoDataset

        root = "~/data/coco14" if dataset_root is None else dataset_root
        ds_train = CocoDataset(
            root=root, split="train", size=64, debug=debug, root_global=root_global)
        ds_val = CocoDataset(root=root, split="val",
                             size=64, debug=debug, root_global=root_global)

    elif dataset_name == "cocostuff27_64":
        from dataset.coco17stuff27 import CocoStuffDataset

        root = "~/data/cocostuff27/images" if dataset_root is None else dataset_root
        root_coco17_annos = "~/data/stego_data/cocostuff/annotations"
        ds_train = CocoStuffDataset(
            root=root, root_coco17_annos=root_coco17_annos, split="train", size=64, size4cluster=320, debug=debug, root_global=root_global)
        ds_val = CocoStuffDataset(
            root=root, root_coco17_annos=root_coco17_annos,  split="val", size=64, size4cluster=320, debug=debug, root_global=root_global)

    elif dataset_name == "voc64":
        from dataset.voc12 import VOCSegmentation

        root = (
            "~/data/pascalvoc12_07/VOCdevkit/VOC2012"
            if dataset_root is None
            else dataset_root
        )
        ds_train = VOCSegmentation(
            root=root, split="train", size=64, debug=debug, root_global=root_global)
        ds_val = VOCSegmentation(root=root, split="val",
                                 size=64, debug=debug, root_global=root_global)

    elif dataset_name == "ffhq64":
        from dataset.ffhq_dataset import FFHQ

        root = "~/data/ffhq/thumbnails128x128" if dataset_root is None else dataset_root
        att_root = "~/data/ffhq-features-dataset/json"
        ds_train = FFHQ(
            root=root, att_root=att_root, split="train", size=64, debug=debug, root_global=root_global)
        ds_val = FFHQ(root=root, att_root=att_root,
                      split="val", size=64, debug=debug, root_global=root_global)

    elif dataset_name == "ffhq128":
        from dataset.ffhq_dataset import FFHQ

        root = "~/data/ffhq/thumbnails128x128" if dataset_root is None else dataset_root
        att_root = "~/data/ffhq-features-dataset/json"
        ds_train = FFHQ(
            root=root, att_root=att_root, split="train", size=128, debug=debug, root_global=root_global)
        ds_val = FFHQ(root=root, att_root=att_root,
                      split="val", size=128, debug=debug, root_global=root_global)

    else:
        raise ValueError(dataset_name)

    dataloader_train = get_torch_loader(ds_train)
    dataloader_val = get_torch_loader(ds_val)

    assert hasattr(dataloader_train.dataset, "id2name")
    assert hasattr(dataloader_val.dataset, "id2name")

    return dataloader_train, dataloader_val
