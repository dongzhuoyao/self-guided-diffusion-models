import torchvision.transforms as transforms
from PIL import Image
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from classy_vision.generic.util import load_checkpoint
from vissl.models import build_model
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict


# %%%
from cmath import log
from loguru import logger
import urllib
from PIL import Image
import torch
import torchvision


class vissl_4sg:
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
        assert "vissl_" in arch_name, f"arch_name {arch_name} not supported"

        arch_name = arch_name.replace("vissl_", "")

        # Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
        # All other options override the simclr_8node_resnet.yaml config.
        cfg = [
            # Specify path for the model weights.
            # Turn on model evaluation mode.
            'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True',
            # Freeze trunk.
            'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True',
            # Extract the trunk features, as opposed to the HEAD.
            'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True',
            # Do not flatten features.
            'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False',
            # Extract only the res5avg features.
            #'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]'
        ]

        if arch_name == 'simclr':
            # 71.56 wget https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch -P /home/thu/data/sg_data/vissl_model_ckpt/
            cfg.extend(
                ['config=pretrain/simclr/simclr_8node_resnet.yaml',
                 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/thu/data/sg_data/vissl_model_ckpt/model_final_checkpoint_phase999.torch'])
        elif arch_name == 'deepclusterv2':
            # wget https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar -P /home/thu/data/sg_data/vissl_model_ckpt/
            cfg.extend(
                ['config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet.yaml',
                 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/thu/data/sg_data/vissl_model_ckpt/deepclusterv2_800ep_pretrain.pth.tar'])

        elif arch_name == 'jigsaw':

            # 46.58  wget https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in1k_goyal19.torch -P /home/thu/data/sg_data/vissl_model_ckpt/
            cfg.extend(
                ['config=pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
                 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/thu/data/sg_data/vissl_model_ckpt/converted_vissl_rn50_jigsaw_in1k_goyal19.torch'])

        else:
            raise NotImplementedError

        # Compose the hydra configuration.
        cfg = compose_hydra_configuration(cfg)
        # Convert to AttrDict. This method will also infer certain config options
        # and validate the config is valid.
        _, cfg = convert_to_attrdict(cfg)

        model = build_model(cfg.MODEL, cfg.OPTIMIZER)
        # Load the checkpoint weights.
        weights = load_checkpoint(
            checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

        # Initializei the model with the simclr model weights.
        init_model_from_consolidated_weights(
            config=cfg,
            model=model,
            state_dict=weights,
            #state_dict_key_name="classy_state_dict",
            state_dict_key_name="model_state_dict",
            skip_layers=[],  # Use this if you do not want to load all layers
        )
        print("Weights have loaded")
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.feat_dim = model.embed_dim

        logger.warning(
            f"vissl_4sg: dataset_name {dataset_name}, arch_name {arch_name}, feat_dim {self.feat_dim}")

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size_official_required),
            transforms.CenterCrop(224), transforms.ToTensor(), ])

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


if __name__ == '__main__':
    sggg = vissl_4sg('cifar10', arch_name='vissl_simclr')
    image = Image.open('dog.jpg')

    # Convert images to RGB. This is important
    # as the model was trained on RGB images.
    image = image.convert("RGB")

    image_transformed = sggg.transform_batch(image)
    features = sggg.transform_image(image_transformed)
    print(f"Features extracted have the shape: { features.shape }")

# %%
