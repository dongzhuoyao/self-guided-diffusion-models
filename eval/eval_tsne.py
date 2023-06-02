import math
import os

import numpy as np
import torch
import wandb
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from eval.test_exps.common_stuff import get_sample_fn, img_pil_save

from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision
from einops import rearrange, repeat
from diffusion_utils.util import clip_unnormalize_to_zero_to_255, make_clean_dir
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import seaborn as sns
# Random state.
RS = 20150101


def scatter(xs, ys, target_names, marker, palette, ax):

    target_ids = range(len(target_names))
    for i, c, label in zip(target_ids, palette, target_names):
        plt.scatter(xs[ys == i, 0], xs[ys == i, 1],
                    c=c, marker=marker, label=label)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    if False:
        txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(xs[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    else:
        txts = []

    return ax, txts


def tsne_vis_both(xs_list, ys_list, target_names, save_name='outputs/output_vis/tsne-generated.png',  save=True):

    # We import seaborn to make nice plots.

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", len(target_names)))
    # We create a scatter plot.
    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')

    first_len = len(xs_list[0])
    xs = np.concatenate(xs_list, 0)
    ys = np.concatenate(ys_list, 0)
    feat_proj = TSNE(random_state=RS).fit_transform(xs)

    scatter(xs=feat_proj[:first_len], ys=ys[:first_len],
            target_names=target_names, marker='o', palette=palette, ax=ax)
    scatter(xs=feat_proj[first_len:], ys=ys[first_len:],
            target_names=target_names, marker='v', palette=palette, ax=ax)

    plt.legend()

    if save:
        plt.savefig(save_name, dpi=120)
        print(save_name)
    else:
        plt.show()
    plt.close()


class DirDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.file_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.file_names)

    def get_cluster_id(self, index):
        file_name = self.file_names[index]
        file_name = file_name.split('cluster')[-1]
        cluster_id = int(file_name.split(".")[0])
        return cluster_id

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        image = read_image(img_path)
        return dict(image=image, index=idx)


def get_feat_imgs_clusterids(feat_extract_fn, image_dir, batch_size):
    ds = DataLoader(DirDataset(image_dir), batch_size=batch_size)
    feats, imgs, cluster_ids = [], [], []

    for batch_id, data_dict in tqdm(enumerate(ds), desc='extracting info for t-sne vis'):
        with torch.no_grad():
            _imgs, indexs = data_dict['image'], data_dict['index']
            _feats = feat_extract_fn(_imgs)
            feats.append(_feats)
            imgs.append(_imgs.cpu())
            for _index in indexs:
                cluster_id = ds.dataset.get_cluster_id(_index)
                cluster_ids.append(cluster_id)

    feats, imgs = torch.cat(feats, 0), torch.cat(imgs, 0)  # [B, C]
    feats = torch.nn.functional.normalize(
        feats, dim=1, p=2.0)  # Normalize feat
    return feats.cpu().numpy(), imgs.cpu().numpy(), cluster_ids


def get_feat_extract_fn(fid_kwargs, name='simclr'):
    from self_sl.ssl_backbone import simclr_4sg
    if name == 'simclr':
        feat_backbone = simclr_4sg(
            dataset_name=fid_kwargs['dataset_name'], image_size=fid_kwargs['image_size'])

        def feat_extract_fn(_imgs):
            batch_transformed = feat_backbone.transform_batch(_imgs)
            _feats = feat_backbone.batch_encode_feat(
                batch_transformed)['feat'].cpu()
            return _feats
        return feat_extract_fn
    elif name == 'image':
        feat_backbone = simclr_4sg(
            dataset_name=fid_kwargs['dataset_name'], image_size=fid_kwargs['image_size'])

        def feat_extract_fn(_imgs):
            raise
            return _imgs
        return feat_extract_fn
    else:
        raise


def kluster_tsne_vis(pl_module, condition_kwargs, sampling_kwargs, fid_kwargs, debug=False, **kwargs):
    """
    t-sne feature
    random select 10 clusters from in32-cluster10k, extract features, do t-sne
    run classifier-guidance on those 10 clusters genearate several images, and to t-sne again. draw those two part of data in t-sne together.
    """
    # KNN
    feat_name = 'simclr'
    sample_dir = 'outputs/output_vis/kluster_tsne_vis/sample'
    gt_dir_4fid = 'outputs/output_vis/kluster_tsne_vis/gt'
    make_clean_dir(sample_dir)
    make_clean_dir(gt_dir_4fid)

    logger.warning(
        f'begin running kluster_tsne_vis(debug{debug}) for sample dir: {sample_dir}, len={len(os.listdir(sample_dir))} gt_dir: {gt_dir_4fid}, len={len(os.listdir(gt_dir_4fid))}')

    kluster_num = 2 if debug else 20
    cluster_total = 10000
    kluster_ids = np.random.randint(0, cluster_total, (kluster_num))
    samples_per_kluster = 60

    sample_fn = get_sample_fn(pl_module=pl_module, sampling_cfg=sampling_kwargs,
                              condition_cfg=condition_kwargs, prefix='kluster_tsne_vis')

    global_id_sample = 0
    for data_dict in tqdm(pl_module.trainer.datamodule.train_dataloader(), desc='generate real data'):
        data_dict['image'] = clip_unnormalize_to_zero_to_255(
            data_dict['image'])
        for _cluster_id, _img in zip(data_dict['cluster'], data_dict['image']):
            _cluster_id = _cluster_id.argmax().long().item()
            if _cluster_id in kluster_ids:
                img_pil_save(_img, os.path.join(
                    gt_dir_4fid, f"real_img{global_id_sample}_cluster{_cluster_id}.png"))
            global_id_sample += 1

    #######################################
    global_id_sample = 0
    for kluster_id in tqdm(kluster_ids, desc='generate fake data'):
        _cluster = torch.nn.functional.one_hot(
            torch.tensor(kluster_id), num_classes=cluster_total)
        _cluster = repeat(_cluster, 'c -> b c', b=samples_per_kluster)
        sample_data_dict = dict(cluster=_cluster)
        sample_data_dict = pl_module.prepare_batch(
            sample_data_dict, subset=None)
        gen_samples, pred_x0 = sample_fn(
            sample_data_dict, subset=None, batch_id=0, batch_size=samples_per_kluster, prefix='kluster_tsne_vis')

        for img in gen_samples:
            img_pil_save(img, os.path.join(
                sample_dir, f"fake_img{global_id_sample}_cluster{kluster_id}.png"))
            global_id_sample += 1

    batch_size = 16

    logger.warning(
        f'begin running kluster_tsne_vis(debug{debug}) for sample dir: {sample_dir}, len={len(os.listdir(sample_dir))} gt_dir: {gt_dir_4fid}, len={len(os.listdir(gt_dir_4fid))}')

    feat_extract_fn = get_feat_extract_fn(
        fid_kwargs=fid_kwargs, name=feat_name)
    sample_feats, sample_imgs, sample_clusterids = get_feat_imgs_clusterids(
        feat_extract_fn=feat_extract_fn, image_dir=sample_dir, batch_size=batch_size)
    gt_feats, gt_imgs, gt_clusterids = get_feat_imgs_clusterids(
        feat_extract_fn=feat_extract_fn, image_dir=gt_dir_4fid, batch_size=batch_size)

    clusterid_unique = list(np.unique(np.array(sample_clusterids)))
    target_names = [f'cluster{k}' for k in range(len(clusterid_unique))]

    sample_ys = [clusterid_unique.index(y) for y in sample_clusterids]
    gt_ys = [clusterid_unique.index(y) for y in gt_clusterids]
    tsne_vis_both(xs_list=[sample_feats, gt_feats], ys_list=[
                  sample_ys, gt_ys], target_names=target_names)
    exit(0)


if __name__ == '__main__':

    sample_feats, sample_clusterids = np.random.randn(
        300, 512), np.random.randint(0, 10, (300,))
    gt_feats, gt_clusterids = np.random.randn(
        300, 512), np.random.randint(0, 10, (300,))

    clusterid_unique = list(np.unique(np.array(sample_clusterids)))
    target_names = [f'cluster{k}' for k in clusterid_unique]

    sample_ys = [clusterid_unique.index(y) for y in sample_clusterids]
    gt_ys = [clusterid_unique.index(y) for y in gt_clusterids]
    tsne_vis_both(xs_list=[sample_feats, gt_feats], ys_list=[
                  sample_ys, gt_ys], target_names=target_names)
