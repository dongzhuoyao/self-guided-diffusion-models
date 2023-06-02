from pathlib import Path
import random
from cv2 import repeat

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from diffusion_utils.util import count_params
from loguru import logger
from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization,
)
from self_sl.msn.logistic_eval import init_model
from self_sl.mae import models_mae
from torchvision import transforms as pth_transforms

from self_sl.timm_backbone import timm_4sg


class simclr_4sg:  # https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr
    def __init__(
        self,
        dataset_name,
        image_size,
        device="cuda",
        weight_path="https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt",
    ):
        self.dataset_name = dataset_name
        self.device = device

        simclr_resnet50 = SimCLR.load_from_checkpoint(
            weight_path, strict=False)
        simclr_resnet50.freeze()
        simclr_resnet50 = simclr_resnet50.encoder
        simclr_resnet50.eval()

        self.model = simclr_resnet50.to(device)
        self.feat_dim = 2048
        self.normalize_transform = transforms.Compose(
            [imagenet_normalization()])

    def transform_image(self, x):
        # x is Image.PIL object or torch.tensor, output a [3, 32, 32] dimension tensor
        assert isinstance(x, torch.Tensor)
        x = Image.fromarray(rearrange(x.to(torch.uint8),
                            "c w h->w h c").cpu().numpy())
        x_transformed = transforms.PILToTensor()(x)
        assert x_transformed.min() >= 0
        x_transformed = x_transformed / 255.0
        x_normalized = self.normalize_transform(x_transformed)
        return x_normalized

    def transform_batch(self, xs, *args, **kwargs):
        assert len(xs.shape) == 4 and xs.shape[1] == 3
        return torch.cat(
            [self.transform_image(_x, *args, **kwargs).unsqueeze(0)
             for _x in xs],
            0,
        )

    @torch.no_grad()
    def batch_encode_feat(self, x_transformed):
        result_list = self.model(x_transformed.to(self.device))
        assert len(result_list) == 1
        feat = result_list[0].detach()
        return dict(feat=feat)


class dino_4sg:  # https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr
    def __init__(
        self,
        dataset_name,
        arch_name,
        is_grey=False,
        device="cuda",
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.is_grey = is_grey
        if is_grey:
            logger.warning("dino_4sg is_grey is True!!!!!!")

        self.input_size_official_required = 224  # TODO,
        if "dino_vits16" == arch_name:
            self.model = torch.hub.load(
                "facebookresearch/dino:main", "dino_vits16"
            )  # 80M
            self.feat_dim = 384  # CLS Token dim

        elif "dino_vits8" == arch_name:
            self.model = torch.hub.load(
                "facebookresearch/dino:main", "dino_vits8"
            )  # ?M
            self.feat_dim = 384  # CLS Token dim

        elif "dino_vitb16" == arch_name:
            self.model = torch.hub.load(
                "facebookresearch/dino:main", "dino_vitb16"
            )  # 350M
            self.feat_dim = 768  # CLS Token dim

        elif "dino_vitb8" == arch_name:
            self.model = torch.hub.load(
                "facebookresearch/dino:main", "dino_vitb8"
            )  # 350M
            self.feat_dim = 768  # CLS Token dim

        elif "dino_xcit_m24_p8" == arch_name:
            self.model = torch.hub.load(
                "facebookresearch/dino:main", "dino_xcit_medium_24_p8"
            )
            self.feat_dim = 512  # CLS Token dim

        elif "dino_resnet50" == arch_name:
            self.model = torch.hub.load(
                "facebookresearch/dino:main", "dino_resnet50"
            )  # 90M
            self.feat_dim = 2048  # CLS Token dim

        else:
            raise ValueError(arch_name)
        self.arch_name = arch_name
        self.model.eval()
        self.model.to(device)

        class Identity():
            def __init__(self):
                super().__init__()

            def __call__(self, x):
                return x

        # https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/visualize_attention.py#L165
        self.transform = pth_transforms.Compose(
            [
                pth_transforms.ToPILImage(),
                pth_transforms.RandomGrayscale(
                    p=1.0) if is_grey else Identity(),  # https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/main_dino.py#L427
                pth_transforms.Resize(self.input_size_official_required),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def transform_image(self, x):
        return self.transform(x)

    def transform_batch(self, xs, *args, **kwargs):
        assert len(xs.shape) == 4 and xs.shape[1] == 3
        return torch.cat(
            [self.transform_image(_x, *args, **kwargs).unsqueeze(0)
             for _x in xs],
            0,
        )

    @torch.no_grad()
    def batch_encode_feat(
        self, x_transformed, return_cls_token=True, attention_map=False
    ):
        x_transformed = x_transformed.to(self.device)
        if "xcit" in self.arch_name or 'resnet' in self.arch_name:
            feat = self.model(x_transformed)
            return dict(feat=feat.detach())

        else:
            intermediate_layers = self.model.get_intermediate_layers(
                x_transformed, n=1)
            last_layer = intermediate_layers[-1].detach()  # [bs, 1+196, 384]
            if return_cls_token:
                if not attention_map:
                    return dict(feat=last_layer[:, 0, :])
                else:
                    attentions = self.model.get_last_selfattention(
                        x_transformed)
                    return dict(feat=last_layer[:, 0, :], attentions=attentions)
            else:
                return dict(
                    feat=last_layer[:, :, :]
                )  # return all patches plus CLS token


class mae_4sg:
    # https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb#scrollTo=4573e6be-935a-4106-8c06-e467552b0e3d
    # https://github.com/facebookresearch/mae
    def __init__(
        self,
        dataset_name,
        arch_name="mae_vit_large_patch16",
        device="cuda",
        input_size_dataset=32,
    ):
        self.device = device
        self.dataset_name = dataset_name
        self.input_size_official_required = 224
        self.input_size_dataset = input_size_dataset

        """
        wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth &&
        wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
        """
        # load model
        if (
            arch_name == "mae_vitlarge"
        ):  # https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
            ckpt_dir = "~/data/sg_data/mae_ckpt/mae_visualize_vit_large.pth"
            self.feat_dim = 1024
            model = getattr(models_mae, "mae_vit_large_patch16")()

        elif (
            arch_name == "mae_vitbase"
        ):  # https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
            ckpt_dir = "~/data/sg_data/mae_ckpt/mae_pretrain_vit_base.pth"
            self.feat_dim = 768
            model = getattr(models_mae, "mae_vit_base_patch16")()
        else:
            raise NotImplementedError(arch_name)
        checkpoint = torch.load(
            Path(ckpt_dir).expanduser().resolve(), map_location="cpu"
        )
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        logger.warning(msg)
        model.eval()
        model.to(self.device)
        self.model = model
        self.transform = transforms.Compose([pth_transforms.Resize(
            self.input_size_official_required), imagenet_normalization()])

    def transform_image(self, img):
        # img is a tensor object shaped [3,32,32]
        assert isinstance(img, torch.Tensor)
        img = img / 255.0
        img = self.transform(img)
        return img

    def transform_batch(self, xs, *args, **kwargs):
        assert len(xs.shape) == 4 and xs.shape[1] == 3
        return torch.cat(
            [self.transform_image(_x, *args, **kwargs).unsqueeze(0)
             for _x in xs],
            0,
        )

    @torch.no_grad()
    def batch_encode_feat(self, x_transformed, mask_ratio=0.0):
        # x_transformed = torch.nn.functional.interpolate(x_transformed,size=224)
        latent, mask, ids_restore = self.model.forward_encoder(
            x_transformed.to(self.device), mask_ratio=mask_ratio
        )
        # assert mask.sum() == 0
        if False:
            latent = latent.mean(
                1
            )  # pooling token dimension, [bs, 197, 768]  #[bs, token, c_dim] ->[bs, c_dim]
        else:
            # only use the CLS token for kmeans clustering
            latent = latent[:, 0, :]
        return dict(feat=latent.detach())


class msn_4sg:
    def __init__(
        self,
        dataset_name,
        arch_name="msn_deit_small",
        device="cuda",
        input_size_dataset=32,
    ):
        self.device = device
        self.dataset_name = dataset_name
        self.input_size_official_required = 224
        self.input_size_dataset = input_size_dataset
        if arch_name == "msn_vits16":
            model_name = 'deit_small'
            pretrained_model_path = "~/data/sg_data/msn_ckpt/vits16_800ep.pth.tar"
            self.feat_dim = 384
        elif arch_name == "msn_vitl16":
            model_name = 'deit_large'
            pretrained_model_path = "~/data/sg_data/msn_ckpt/vitl16_600ep.pth.tar"
            self.feat_dim = 1024
        elif arch_name == "msn_vitb16":
            model_name = 'deit_base'
            pretrained_model_path = "~/data/sg_data/msn_ckpt/vitb16_600ep.pth.tar"
            self.feat_dim = 768
        elif arch_name == "msn_vitb4":
            model_name = 'deit_base_p4'
            pretrained_model_path = "~/data/sg_data/msn_ckpt/vitb4_300ep.pth.tar"
            self.feat_dim = 768
        elif arch_name == "msn_vitl7":
            model_name = 'deit_large_p7'
            pretrained_model_path = "~/data/sg_data/msn_ckpt/vitl7_200ep.pth.tar"
            self.feat_dim = 1024
        else:
            raise ValueError(arch_name)

        pretrained_model_path = Path(
            pretrained_model_path).expanduser().resolve()
        logger.warning("loading pretrained model from {}".format(
            pretrained_model_path))
        self.model = init_model(
            device=self.device,
            pretrained=pretrained_model_path,
            model_name=model_name,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(size=256),
                transforms.Resize(size=self.input_size_official_required),
                transforms.CenterCrop(size=self.input_size_official_required),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]
        )

    def transform_image(self, img):
        # img is a tensor object shaped [3,32,32]
        assert isinstance(img, torch.Tensor)
        x_normalized = self.transform(img)
        return x_normalized

    def transform_batch(self, xs, *args, **kwargs):
        assert len(xs.shape) == 4 and xs.shape[1] == 3
        return torch.cat(
            [self.transform_image(_x, *args, **kwargs).unsqueeze(0)
             for _x in xs],
            0,
        )

    @torch.no_grad()
    def batch_encode_feat(self, x_transformed):
        feat = self.model.forward_blocks(
            x_transformed.to(self.device), num_blocks=1, patch_drop=0.0
        ).cpu()
        return dict(feat=feat)


def is_feat_from_simclr(feat_name):
    return feat_name in ["simclr", "simclr_res50"]


def is_feat_from_mae(feat_name):
    return feat_name in ["mae_vitbase", "mae_vitlarge"]


def is_feat_from_timm(feat_name):
    return True


def is_feat_from_dino(feat_name):
    return feat_name in [
        "dino_vits16",
        "dino_vits8",
        "dino_vitb16",
        "dino_vitb8",
        "dino_resnet50",
        "dino_xcit_m24_p8",
    ]


def is_feat_from_msn(feat_name):
    return feat_name.startswith("msn")


def get_ssl_backbone(feat_name, dataset_name, image_size, is_grey=False):
    if is_feat_from_simclr(feat_name=feat_name):
        assert not is_grey
        _backbone = simclr_4sg(dataset_name=dataset_name,
                               image_size=image_size)

    elif is_feat_from_mae(feat_name=feat_name):
        assert not is_grey
        _backbone = mae_4sg(
            arch_name=feat_name,
            dataset_name=dataset_name,
            input_size_dataset=image_size,
        )

    elif is_feat_from_dino(feat_name=feat_name):
        _backbone = dino_4sg(dataset_name=dataset_name,
                             arch_name=feat_name, is_grey=is_grey)

    elif is_feat_from_msn(feat_name=feat_name):
        assert not is_grey
        _backbone = msn_4sg(dataset_name=dataset_name, arch_name=feat_name)

    elif is_feat_from_timm(feat_name=feat_name):
        assert not is_grey
        _backbone = timm_4sg(dataset_name=dataset_name, arch_name=feat_name)

    else:
        raise ValueError(feat_name)

    count_params(_backbone.model, verbose=True)
    return _backbone


if __name__ == "__main__":
    m = mae_4sg()
    x = torch.ones(3, 32, 32)
    x = m._transform_image(x)
    feat = m.batch_encode_feat(x.unsqueeze(0))
    print(feat.shape)
