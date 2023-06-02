import importlib
import os
import shutil
from pathlib import Path

import torch
import numpy as np

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange
from loguru import logger


def delete_dir(_dir):
    if os.path.exists(_dir):
        logger.warning(f"{_dir} remove it now.")
        shutil.rmtree(_dir)
    else:
        logger.warning(f"{_dir} don not exist, do nothing")


def make_clean_dir(_dir):
    _dir = Path(_dir).expanduser().resolve()
    if os.path.exists(_dir):
        logger.warning(f"{_dir} exists, remove it now.")
        shutil.rmtree(_dir)

    if not os.path.exists(_dir):
        os.makedirs(_dir)
        logger.warning(f"{_dir} created for you now.")
    return _dir


# https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792
def slerp_np(val, low, high):  # float, [C], [C]
    omega = np.arccos(
        np.clip(np.dot(low / np.linalg.norm(low),
                high / np.linalg.norm(high)), -1, 1)
    )
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


# https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
def slerp_batch_torch(val, low, high):
    """
    # float,[B,C],[B,C]
    """
    assert len(low.shape) == 2
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


def tensor_dict_copy(batch_data):
    batch_copyed = dict()
    for k, v in batch_data.items():
        batch_copyed[k] = v.clone()
    return batch_copyed


def clip_x0_minus_one_to_one(pred_x0, clip_denoised, dtp):
    if (
        dtp < 1.0
    ):  # https://github.com/lucidrains/imagen-pytorch/blob/bfe761b52c93f53c1a961c0912bed3b33042382c/imagen_pytorch/imagen_pytorch.py#L1382
        s = torch.quantile(
            rearrange(pred_x0, "b ... -> b (...)").abs(), dtp, dim=-1)
        s.clamp_(min=1.0)  # DTP only take effect if min(s)>1
        s = right_pad_dims_to(pred_x0, s)
        pred_x0 = pred_x0.clamp(-s, s) / s
    else:  # if self.dtp is default 1, don't apply DTP
        if clip_denoised:
            pred_x0.clamp_(-1.0, 1.0)
    return pred_x0


class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(
                    x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def clip_unnormalize_to_zero_to_255(img, clip=True):
    return ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)


def batch_to_same_firstimage(batch):
    batch_new = dict()
    for key_name in batch:
        batch_new[key_name] = torch.zeros_like(batch[key_name])
        for _ in range(len(batch[key_name])):
            # always use first image's attribute to test conditional sample diversity
            batch_new[key_name][_] = batch[key_name][0]
    return batch_new


def batch_to_samecondition(batch, samecondition_num=7):
    batch_new = dict()
    for key_name in batch:
        batch_new[key_name] = torch.zeros_like(batch[key_name])
        for _ in range(len(batch[key_name])):
            batch_new[key_name][_] = batch[key_name][_ // samecondition_num]
    return batch_new


def batch_to_samecondition_v2(batch,  different_key, samecondition_num=7):
    #same_key = 'lostbboxmask'
    #different_key = 'cluster'
    batch_new = dict()
    for key_name in batch:
        if key_name != different_key:
            batch_new[key_name] = torch.zeros_like(batch[key_name])
            for _ in range(len(batch[key_name])):
                batch_new[key_name][_] = batch[key_name][_ //
                                                         samecondition_num]
        else:
            batch_new[key_name] = batch[key_name]

    return batch_new


def samecluster_difflost(batch, samecondition_num=7):
    batch_new = dict()
    for k in batch:
        batch_new[k] = torch.zeros_like(batch[k])
        for _ in range(len(batch[k])):
            batch_new[k][_] = batch[k][_ // samecondition_num]
    return batch_new


def samestego_diffcluster(batch, samecondition_num=7):
    batch_new = dict()
    for k in batch:
        batch_new[k] = torch.zeros_like(batch[k])
        for _ in range(len(batch[k])):
            batch_new[k][_] = batch[k][_ // samecondition_num]
    return batch_new


def samecluster_diffstego(batch, samecondition_num=7):
    batch_new = dict()
    for k in batch:
        batch_new[k] = torch.zeros_like(batch[k])
        for _ in range(len(batch[k])):
            batch_new[k][_] = batch[k][_ // samecondition_num]
    return batch_new


def batch_to_conditioninterp(batch, interp=9):
    batch_size = len(batch["cond"])  # [bs, C] for cond
    if batch_size < interp:
        logger.warning(f"only can log interp {batch_size}")
        interp = batch_size

    feat1 = rearrange(batch["cond"][0], "c -> 1 c")
    feat2 = rearrange(batch["cond"][1], "c -> 1 c")
    lin_w = torch.linspace(0, 1, interp).reshape(-1,
                                                 1).to(batch["cond"].device)
    feat_interped = feat1 * lin_w + feat2 * (1 - lin_w)  # [N, c]
    # n1, n2 = 50, 50
    # xv, yv = torch.meshgrid([torch.linspace(0, 1, n1), torch.linspace(0, 1, n2)])
    batch_new = dict(cond_scale=batch["cond_scale"], cond=feat_interped)
    return batch_new


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start: start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.warning(
            f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params."
        )
    return total_params


def instantiate_from_config(config):
    assert "target" in config
    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    except:
        print(config["target"])
        raise


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size, device=x.device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_onlyx(x, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=(x.shape[0], 1))
    else:
        lam = np.ones((x.shape[0],))
    lam = torch.from_numpy(lam).to(x.device)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, lam


def mixup_data_half(x, y):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = 0.5
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
