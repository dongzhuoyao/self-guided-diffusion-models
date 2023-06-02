import math
import os

import numpy as np
import torch
import wandb
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from sklearn.neighbors import NearestNeighbors
from torchvision.utils import make_grid
from torchvision.io import read_image
import torchvision
from einops import rearrange, repeat
from diffusion_utils.util import make_clean_dir


class DirDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.file_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        image = read_image(img_path)
        return image


def get_embedding_extract_fn(fid_kwargs, name="simclr"):
    if name == "simclr":
        from self_sl.ssl_backbone import simclr_4sg
        feat_backbone = simclr_4sg(
            dataset_name=fid_kwargs["dataset_name"], image_size=fid_kwargs["image_size"]
        )

        def feat_extract_fn(_imgs):
            batch_transformed = feat_backbone.transform_batch(_imgs)
            _feats = feat_backbone.batch_encode_feat(batch_transformed)["feat"].cpu()
            return _feats

        return feat_extract_fn
    elif name == "imagespace":

        def feat_extract_fn(_imgs):
            _imgs = rearrange(_imgs, "b c w h ->b (c w h)").to(torch.float32).cpu()
            return _imgs

        return feat_extract_fn
    else:
        raise


def get_embedding_and_imgs(embedding_extract_fn, image_dir, batch_size):
    ds = DataLoader(DirDataset(image_dir), batch_size=batch_size)
    feats, imgs = [], []
    for _, _imgs in tqdm(enumerate(ds), desc="extracting info for KNN vis"):
        with torch.no_grad():
            _feats = embedding_extract_fn(_imgs)
            feats.append(_feats)
            imgs.append(_imgs.cpu())
    feats, imgs = torch.cat(feats, 0), torch.cat(imgs, 0)  # [B, C]
    feats = torch.nn.functional.normalize(feats, dim=1, p=2.0)  # Normalize feat
    return feats, imgs


def get_wandbimg_by_knn(q_feats, q_imgs, imgs_gallery, nn_obj_gallery):
    query_wandb_result = []
    query_result = nn_obj_gallery.kneighbors(q_feats, return_distance=False)
    for _i in range(len(query_result)):
        result_idx = query_result[_i]
        # indexing result and removing itself
        result_img_row = imgs_gallery[result_idx, :]
        cur_row = np.concatenate([q_imgs[_i : _i + 1], result_img_row], 0)  # [B,3,W,H]
        query_row_concated = make_grid(
            torch.tensor(cur_row, dtype=torch.uint8),
            nrow=len(cur_row),
            scale_each=True,
            pad_value=255,
        )
        query_wandb_result.append(wandb.Image(query_row_concated / 255.0))
    return query_wandb_result


def knn_papervis(
    q_feats, q_imgs, gallery_imgs, scipy_gallery, sample_num, output_dir, width, grid_size=10
):
    def draw_bbox(image, color):
        c, w, h = image.shape
        # (xmin, ymin, xmax, ymax)
        boxes = rearrange(torch.tensor([0, 0, w, h]), "c->1 c")
        image = torchvision.utils.draw_bounding_boxes(
            image, boxes=boxes, width=width, colors=[color]
        )
        return image

    make_clean_dir(output_dir)
    query_result = scipy_gallery.kneighbors(q_feats, return_distance=False)

    for _i, result_idx in enumerate(query_result):
        current_q = q_imgs[_i : _i + 1]
        tensor_list = []
        for result_id in result_idx:
            color = "red" if result_id < sample_num else "blue"
            # indexing result and removing itself
            result_img_single = torch.tensor(gallery_imgs[result_id, :])
            result_img_single = draw_bbox(result_img_single, color=color) / 255.0
            tensor_list.append(result_img_single)

        assert int(math.sqrt(len(tensor_list))) >= grid_size
        tensor_list = tensor_list[:grid_size ** 2]
        query_row_concated = make_grid(
            tensor_list,
            nrow=grid_size,
            scale_each=True,
            pad_value=255,
        )

        img_path = os.path.join(output_dir, f"q_{_i}_result.png")
        torchvision.utils.save_image(query_row_concated, img_path)

        query_img_path = os.path.join(output_dir, f"q_{_i}.png")
        torchvision.utils.save_image(current_q.float() / 255.0, query_img_path)
        print(f"saving {img_path}")


def get_knn_eval_dict(
    sample_dir,
    gt_dir_4fid,
    fid_kwargs,
    knn_k=32,
    q_num=10,
    batch_size=16,
    width=1,
    debug=False,
    papervis=False,
):
    # KNN
    logger.warning(
        f"begin running KNN(debug{debug}) for sample dir: {sample_dir}, len={len(os.listdir(sample_dir))} gt_dir: {gt_dir_4fid}, len={len(os.listdir(gt_dir_4fid))}"
    )
    if debug:
        knn_k = 4
    nn_obj_sample = NearestNeighbors(n_neighbors=knn_k)
    nn_obj_gt = NearestNeighbors(n_neighbors=knn_k)
    nn_obj_all = NearestNeighbors(n_neighbors=knn_k)
    knn_dict = dict()

    similarity_metric_list = ["simclr"]
    if papervis:
        similarity_metric_list = ["simclr", "imagespace"]
    for similarity_metric_name in similarity_metric_list:
        embedding_extract_fn = get_embedding_extract_fn(
            fid_kwargs=fid_kwargs, name=similarity_metric_name
        )
        assert batch_size > 1

        sample_feats, sample_imgs = get_embedding_and_imgs(
            embedding_extract_fn=embedding_extract_fn,
            image_dir=sample_dir,
            batch_size=batch_size,
        )
        gt_feats, gt_imgs = get_embedding_and_imgs(
            embedding_extract_fn=embedding_extract_fn,
            image_dir=gt_dir_4fid,
            batch_size=batch_size,
        )
        # set desired number of neighbors
        nn_obj_sample.fit(sample_feats.cpu())
        nn_obj_gt.fit(gt_feats.cpu())
        if True:
            feats_all = torch.cat([sample_feats, gt_feats], 0)
            imgs_all = torch.cat([sample_imgs, gt_imgs], 0)
            nn_obj_all.fit(feats_all.cpu())

        query_wandb_result = get_wandbimg_by_knn(
            q_feats=sample_feats[:q_num],
            q_imgs=sample_imgs[:q_num],
            imgs_gallery=gt_imgs,
            nn_obj_gallery=nn_obj_gt,
        )
        knn_dict[
            f"knn_{similarity_metric_name}_query_gt_by_sample"
        ] = query_wandb_result
        query_wandb_result = get_wandbimg_by_knn(
            q_feats=gt_feats[:q_num],
            q_imgs=gt_imgs[:q_num],
            imgs_gallery=sample_imgs,
            nn_obj_gallery=nn_obj_sample,
        )
        knn_dict[
            f"knn_{similarity_metric_name}_query_sample_by_gt"
        ] = query_wandb_result

        if papervis:
            knn_papervis(
                q_feats=gt_feats[:q_num],
                q_imgs=gt_imgs[:q_num],
                gallery_imgs=imgs_all,
                scipy_gallery=nn_obj_all,
                width=width,
                sample_num=len(sample_feats),
                output_dir=f"outputs/output_vis/knn_{similarity_metric_name}_vis_query_gt_by_sample",
            )

            knn_papervis(
                q_feats=sample_feats[:q_num],
                q_imgs=sample_imgs[:q_num],
                gallery_imgs=imgs_all,
                scipy_gallery=nn_obj_all,
                width=width,
                sample_num=len(sample_feats),
                output_dir=f"outputs/output_vis/knn_{similarity_metric_name}_vis_query_sample_by_gt",
            )
    if papervis:
        print('knn_papervis done, kill process exit(0)')
        exit(0)


    logger.warning(f"end running KNN(debug{debug})")
    return knn_dict

    # KNN


if __name__ == "__main__":
    sample_dir = "/home/thu/data/sg_fid_eval/cifar10_val"
    ds = DataLoader(DirDataset(sample_dir), batch_size=10)
    _iter = iter(ds)
    _batch = next(_iter)
    print(_batch.shape)

    fid_kwargs = dict()
    fid_kwargs["dataset_name"] = "cifar10"
    fid_kwargs["image_size"] = 32

    get_knn_eval_dict(
        sample_dir=sample_dir,
        gt_dir_4fid=sample_dir,
        fid_kwargs=fid_kwargs,
        batch_size=10,
    )
