# %%%
from cmath import log
import os
from pathlib import Path
import shutil
from diffusion_utils.taokit.vis_utils import upsample_pt
from side_repo.STEGO.src.sg_train_segmentation import LitUnsupervisedSegmenter
from PIL import Image
import requests
from io import BytesIO
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from crf import dense_crf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.io import read_image
import h5py
import numpy as np
from torchvision import transforms as T
from loguru import logger
from einops import rearrange
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from side_repo.STEGO.src.utils import unnorm, remove_axes
import hydra
from omegaconf import DictConfig, OmegaConf
import glob


def img_np_save(img, save_path):
    #logger.warning(f"save to {save_path}")
    img_pil = Image.fromarray(np.uint8(img))
    img_pil.save(save_path)
    if False:
        print(np.unique(np.array(Image.open(save_path))))
        print("*" * 10)


def make_clean_dir(_dir):
    _dir = Path(_dir).expanduser().resolve()
    if os.path.exists(_dir):
        logger.warning(f"{_dir} exists, remove it now.")
        shutil.rmtree(_dir)

    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def papervis_makegrid(tensorlist, vis_file_name, nrow=2, padding=2):
    tensorlist = [rearrange(_t, 'c w h-> 1 c w h') for _t in tensorlist]
    tensorlist = torch.cat(tensorlist, dim=0)
    grid = make_grid(tensorlist, nrow=nrow, padding=padding, pad_value=255.0)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(vis_file_name)
    #logger.warning(vis_file_name)
    # exit(0)


def get_transform():
    return T.Compose([T.ToTensor(), normalize])


class RecursiveDirDataset(Dataset):
    def __init__(self, img_dir, patch_size=16):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.file_names = list()
        for full_file_path in glob.iglob(img_dir + '**/**', recursive=True):
            if os.path.isfile(full_file_path):
                self.file_names.append(full_file_path)
        self.file_names = list(set(self.file_names))
        logger.warning(f"{len(self.file_names)}  files found in {img_dir}")
        self.base_names = [os.path.basename(
            filename) for filename in self.file_names]

        self.transform = get_transform()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        size_im = (
            3,
            int(np.ceil(image.shape[1] / self.patch_size) * self.patch_size),
            int(np.ceil(image.shape[2] / self.patch_size) * self.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : image.shape[1], : image.shape[2]] = image
        padded_wh = np.array([image.shape[1], image.shape[2]])
        image = paded

        return dict(image=image, padded_wh=padded_wh, index=idx)


def get_model(ckpt_path):
    model = LitUnsupervisedSegmenter.load_from_checkpoint(ckpt_path).cuda()
    return model


def stego_batch_pred_vis_save(image_dir,  result_dir, papervis_dir, ckpt_path, is_vis, num_workers, batch_size=1):
    model = get_model(ckpt_path)
    assert batch_size == 1
    ds = DataLoader(RecursiveDirDataset(image_dir),
                    batch_size=batch_size, num_workers=num_workers)
    for data in tqdm(ds, total=len(ds)):
        generated_pred(
            model=model, data=data, result_dir=result_dir, papervis_dir=papervis_dir, ds=ds, is_vis=is_vis
        )


@torch.no_grad()
def generated_pred(model, data, ds, result_dir, papervis_dir, is_vis=True, up_size=256):
    images = data["image"].cuda()
    indexs = data["index"]
    code1 = model(images)
    code2 = model(images.flip(dims=[3]))
    code = (code1 + code2.flip(dims=[3])) / 2
    code = F.interpolate(
        code, images.shape[-2:], mode="bilinear", align_corners=False)
    cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

    for _i, (_img_i, padded_wh, _cluster_prob, _index) in enumerate(zip(images, data["padded_wh"], cluster_probs, indexs)):
        base_name = ds.dataset.base_names[_index]
        single_img = _img_i[:, : padded_wh[0], : padded_wh[1]].cpu()
        cluster_pred = dense_crf(single_img, _cluster_prob).argmax(0)
        #print("np.unique(cluster_pred)", np.unique(cluster_pred))
        img_np_save(
            cluster_pred, os.path.join(
                result_dir, base_name.replace(".jpg", ".png"))
        )
        _unpadded_img = unnorm(single_img)
        if is_vis:
            #################
            if False:
                fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5))
                ax[0].imshow(
                    _unpadded_img
                    .permute(1, 2, 0)
                    .cpu()
                )
                ax[0].set_title("Image")
                ax[1].imshow(model.label_cmap[cluster_pred])
                ax[1].set_title("Cluster Predictions")
                remove_axes(ax)
                save_path = os.path.join(
                    Path(__file__).parent.resolve(), f"pdfs/{base_name}"
                )
                #print(save_path)
                plt.savefig(save_path)
                plt.show()

            else:
                _cluster_pred = torch.from_numpy(
                    model.label_cmap[cluster_pred])
                _cluster_pred = rearrange(_cluster_pred, 'w h c->c w h')
                _unpadded_img = _unpadded_img * 255.0

                _unpadded_img = upsample_pt(
                    _unpadded_img, up_size=up_size, mode='bilinear').to('cuda')

                _cluster_pred = upsample_pt(
                    _cluster_pred, up_size=up_size, mode='nearest').to('cuda')

                papervis_makegrid([_unpadded_img, _cluster_pred], os.path.join(
                    papervis_dir, base_name.replace(".jpg", ".png")))


@hydra.main(config_path="configs", config_name="config_base")
def my_app(cfg: DictConfig) -> None:
    _sample = cfg.sample
    make_clean_dir(_sample.result_dir)
    make_clean_dir(_sample.papervis_dir)
    stego_batch_pred_vis_save(
        image_dir=_sample.image_dir, result_dir=_sample.result_dir,
        papervis_dir=_sample.papervis_dir, ckpt_path=_sample.ckpt_path,
        is_vis=_sample.is_vis, num_workers=_sample.num_workers)


if __name__ == "__main__":
    my_app()

# %%
