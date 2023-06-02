import torch


def upsample_pt(pt, up_size=256, mode="bilinear"):
    _dtype = pt.dtype
    pt = pt.float()
    ret = torch.nn.functional.interpolate(
        pt.unsqueeze(0), up_size, mode=mode,
    )[0]
    return ret.to(_dtype).cpu()
