
# %%%
from cmath import log
from loguru import logger
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import torchvision


class timm_4sg:
    def __init__(
        self,
        dataset_name,
        arch_name="resnet18",
        device="cuda",
        input_size_dataset=32,
    ):
        self.device = device
        self.dataset_name = dataset_name
        self.input_size_official_required = 224
        self.input_size_dataset = input_size_dataset
        assert "timm_" in arch_name, f"arch_name {arch_name} not supported"
        arch_name = arch_name.replace("timm_", "")
        if "random" in arch_name:
            logger.warning(f"random {arch_name} !!!!! ")
            model = timm.create_model(
                arch_name.replace("_random", ""), pretrained=False)
        else:
            model = timm.create_model(arch_name, pretrained=True)
        model.eval()
        try:
            self.feat_dim = model.embed_dim
        except:
            logger.warning(
                f"model {arch_name} has no embed_dim, use model.feature_info[-1]['num_chs'] instead")
            self.feat_dim = model.feature_info[-1]['num_chs']

        logger.warning(
            f"timm_4sg: dataset_name {dataset_name}, arch_name {arch_name}, feat_dim {self.feat_dim}")

        config = resolve_data_config({}, model=model)
        self.transform = create_transform(**config)

        model.to(self.device)
        self.model = model
        #self.transform = transforms.Compose([pth_transforms.Resize(self.input_size_official_required),imagenet_normalization()])

    def transform_image(self, img):
        # img is a tensor object shaped [3,32,32]
        assert isinstance(img, torch.Tensor)
        img = torchvision.transforms.ToPILImage()(img)
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
    def batch_encode_feat(self, x_transformed):
        feat = self.model.forward_features(x_transformed.to(self.device))
        if len(feat.shape) == 4:
            feat = feat.mean(dim=[2, 3])  # global average pooling
        else:
            feat = feat
        return dict(feat=feat.detach())

# %%
