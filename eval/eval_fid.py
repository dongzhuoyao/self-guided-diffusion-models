from ast import Not
import math
import shutil

import torch
import wandb

import numpy as np
import os
from cleanfid import fid as clean_fid
from tqdm import tqdm
from einops import rearrange
from loguru import logger
import torch_fidelity
from dataset.ds_utils.dataset_common_utils import ds_has_label_info, need_to_upsample256
from diffusion_utils.util import (
    batch_to_same_firstimage,
    batch_to_samecondition,
    batch_to_samecondition_v2,
    clip_unnormalize_to_zero_to_255,
    
    make_clean_dir,
)
from pytorch_fid import fid_score

from distinctipy import distinctipy
from eval.compute_pdrc_from_icgan import compute_prdc
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid

from eval.papervis_utils import (
    draw_grid_condscale_stego,
    draw_grid_img,
    draw_grid_imgsize32_interp,
    draw_grid_random_lost_with_box,
    draw_grid_random_stego_with_mask,
    draw_grid_scoremix_vis,
    draw_grid_stego,
    draw_grid_lost_bbox,
    draw_grid_stego_chainvis,
    draw_grid_lost_chainvis
)
from eval.test_exps.common_stuff import img_pil_save, should_exp


def cleanfid_compute_fid_return_feat(
    fdir1,
    fdir2,
    mode="clean",
    num_workers=0,
    batch_size=8,
    device=torch.device("cuda:0"),
    verbose=True,
    custom_image_tranform=None,
):

    feat_model = clean_fid.build_feature_extractor(mode, device)

    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = clean_fid.get_folder_features(
        fdir1,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname1} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
    )
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = clean_fid.get_folder_features(
        fdir2,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname2} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
    )
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = clean_fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid, np_feats1, np_feats2


def cycle(
    dl,
):  # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/8c3609a6e3c216264e110c2019e61c83dafad9f5/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L34
    while True:
        for data in dl:
            yield data


def get_torch_fidelity_dict(gt_dir, sample_dir, debug):
    tf_new_dict = dict()
    if not debug:
        tf_metrics_dict = torch_fidelity.calculate_metrics(
            input1=gt_dir,
            input2=sample_dir,
            cuda=True,
            isc=False,
            fid=True,
            ppl=False,
            # kid=False if debug else True, when on small dataset like FFHQ, the kid is not convenient for kid
            kid=False,
            verbose=False,
        )
        tf_new_dict["fid_tf"] = tf_metrics_dict["frechet_inception_distance"]
        # if not debug:
        #    tf_new_dict['kid_tf'] = tf_metrics_dict['kernel_inception_distance_mean']
        #    tf_new_dict['kid_std_tf'] = tf_metrics_dict['kernel_inception_distance_std']

        for isc_splits in [1, 10]:
            tf_metrics_dict = torch_fidelity.calculate_metrics(
                input1=sample_dir,
                cuda=True,
                isc=True,
                isc_splits=isc_splits,
                verbose=False,
            )
            tf_new_dict[f"is_tf_s{isc_splits}"] = tf_metrics_dict[
                "inception_score_mean"
            ]
            tf_new_dict[f"is_std_tf_s{isc_splits}"] = tf_metrics_dict[
                "inception_score_std"
            ]
    return tf_new_dict


def get_fid_dict(sample_dir, gt_dir, dataset_name, debug, nearest_k=5):
    cleanfid_dict = dict()
    logger.warning("begin calculating FID between two image folders...")
    if dataset_name == "cifar10":
        fid_clean_w_c10train = clean_fid.compute_fid(
            sample_dir, dataset_name="cifar10", dataset_res=32, dataset_split="train"
        )
        fid_clean_w_c10val = clean_fid.compute_fid(
            sample_dir, dataset_name="cifar10", dataset_res=32, dataset_split="test"
        )
        cleanfid_dict.update(
            fid_clean_w_c10train=fid_clean_w_c10train,
            fid_clean_w_c10val=fid_clean_w_c10val,
        )
    else:
        logger.warning(
            "clean_fid with precomputing is only supporting cifar10 dataset now."
        )

    sfid = fid_score.main(_path1=gt_dir,_path2=sample_dir)
    cleanfid_dict['sfid'] = sfid

    cleanfid_dict.update(
        get_torch_fidelity_dict(
            gt_dir=gt_dir, sample_dir=sample_dir, debug=debug)
    )
    clean_fid_raw, feat_sample, feat_real = cleanfid_compute_fid_return_feat(
        fdir1=sample_dir, fdir2=gt_dir, num_workers=0, batch_size=8, device=torch.device("cuda:0"),
    )
    cleanfid_dict.update(clean_fid_raw=clean_fid_raw)

    #########################
    num_pr_images = min(
        len(feat_real), len(feat_sample), 5000
    )  # min to be more robust,most case should be 5000
    logger.warning(f"Subsampling {num_pr_images} samples for prdc metrics!")
    idxs_selected_real = np.random.choice(
        range(len(feat_real)), num_pr_images, replace=False
    )
    idxs_selected_sample = np.random.choice(
        range(len(feat_sample)), num_pr_images, replace=False
    )
    prdc_metrics = compute_prdc(
        real_features=feat_real[idxs_selected_real],
        fake_features=feat_sample[idxs_selected_sample],
        nearest_k=nearest_k,
    )
    cleanfid_dict.update(prdc_metrics)

    logger.warning("#" * 66)
    logger.warning(f"{cleanfid_dict}")
    logger.warning("#" * 66)
    logger.warning("finish calculating FID between two image folders...")
    return cleanfid_dict, clean_fid_raw


def get_samecondition_num(dataset_name):
    if dataset_name == "in64":
        samecondition_num = 9
    elif dataset_name == 'in32':
        samecondition_num = 18
    else:
        samecondition_num = 11
    return samecondition_num


def get_makegrid_padding(dataset_name):
    if dataset_name == 'in32':
        _padding = 1
    elif dataset_name in ['cocostuff64', 'coco64', 'voc64']:
        _padding = 5
    elif dataset_name in ['in64']:
        _padding = 2
    else:
        raise NotImplementedError
    return _padding


def eval_fid_callback_before(ds_name, condition_method, vis, _exp, data_dict):
    from .test_exps.common_stuff import should_vis

    samecondition_num = get_samecondition_num(ds_name)

    if should_vis(vis, "random"):
        logger.warning("vis random samples")

    if should_vis(vis, "random_stego_with_mask"):
        logger.warning("vis random_stego_with_mask")

    if should_vis(vis, "random_lost_with_box"):
        logger.warning("vis random_lost_with_box")

    if should_vis(vis, "samecondition"):

        data_dict = batch_to_samecondition(
            data_dict, samecondition_num=samecondition_num
        )
        if ds_name in ['in32, in64']:
            logger.info(
                "samecondition, clustering_id", data_dict["cluster"][0].argmax(
                    0)
            )

    if should_vis(vis, "same_cluster_same_lost"):

        data_dict = batch_to_samecondition(
            data_dict, samecondition_num=samecondition_num
        )
        logger.info("same_cluster_same_lost .....")

    if should_vis(vis, "same_cluster_diff_lost"):
        different_key = 'lostbboxmask'
        data_dict = batch_to_samecondition_v2(
            data_dict,  different_key=different_key, samecondition_num=samecondition_num
        )
        logger.info("same_cluster_diff_lost .....")

    if should_vis(vis, "diff_cluster_same_lost"):

        different_key = 'cluster'
        data_dict = batch_to_samecondition_v2(
            data_dict,  different_key=different_key, samecondition_num=samecondition_num
        )
        logger.info("diff_cluster_same_lost .....")

    if should_vis(vis, "same_stego_diff_cluster"):

        # different_key = 'stegomask'
        different_key = 'cluster'  # in fist exp, it's used, we hope in later exp, it's not used
        data_dict = batch_to_samecondition_v2(
            data_dict,  different_key=different_key, samecondition_num=samecondition_num
        )
        logger.info("same_stego_diff_cluster .....")

    if should_vis(vis, "diff_z_same_stego"):
        samecondition_num = 8
        # different_key = 'stegomask'
        different_key = 'cluster'  # in fist exp, it's used, we hope in later exp, it's not used
        data_dict = batch_to_samecondition_v2(
            data_dict,  different_key=different_key, samecondition_num=samecondition_num
        )
        logger.info("diff_z_same_stego .....")

    return data_dict


def eval_fid_callback_after(ds_name, condition_method, vis, papervis_dir, gen_samples, pred_x0, batch_id, data_dict):
    from .test_exps.common_stuff import should_vis

    _padding = get_makegrid_padding(ds_name)
    samecondition_num = get_samecondition_num(ds_name)
    _prefix = f"{ds_name}_{condition_method}"

    if should_vis(vis, "random"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_random_uncurated_{batch_id}.png")
        _nrow = 16 if "32" in ds_name else 9
        draw_grid_img(tensorlist=gen_samples,  dataset_name=ds_name, nrow=_nrow, padding=_padding,
                      ncol=_nrow, save_path=papervis_save_path)

    if should_vis(vis, "random_stego_with_mask"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_random_stego_with_mask_{batch_id}.png")
        draw_grid_random_stego_with_mask(
            tensorlist=gen_samples,
            masks=data_dict['stegomask'],
            original_images=data_dict['image'],
            save_path=papervis_save_path,
            nrow=4,
            ncol=8,
            mask_alpha=1, padding=_padding
        )

    if should_vis(vis, "random_lost_with_box"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_random_lost_with_box_{batch_id}.png")
        draw_grid_random_lost_with_box(
            tensorlist=gen_samples,
            lostmask=data_dict['lostbboxmask'],
            original_images=data_dict['image'],
            save_path=papervis_save_path,
            nrow=8,
            ncol=8,
            padding=_padding
        )

    if should_vis(vis, "stego_chainvis"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_stego_chainvis_{batch_id}.png")
        draw_grid_stego_chainvis(
            x_inter=pred_x0,  # pred_x0: [timestep, batch, c, w, h]
            masks=data_dict['stegomask'],
            original_images=data_dict['image'],
            save_path=papervis_save_path,
            padding=_padding
        )
        logger.info("stego_chainvis .....")

    if should_vis(vis, "samecondition"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_samecondition_{batch_id}.png")
        draw_grid_img(
            dataset_name=ds_name,
            tensorlist=gen_samples,
            nrow=samecondition_num,
            ncol=len(gen_samples)//samecondition_num,
            save_path=papervis_save_path, padding=_padding
        )

    if should_vis(vis, "lost_chainvis"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_lost_chainvis_{batch_id}.png")
        draw_grid_lost_chainvis(
            x_inter=pred_x0,  # pred_x0: [timestep, batch, c, w, h]
            lostmask=data_dict['lostbboxmask'],
            original_images=data_dict['image'],
            save_path=papervis_save_path,
            padding=_padding
        )
        logger.info("lost_chainvis .....")

    if should_vis(vis, "same_cluster_same_lost"):
        for _id, start_id in enumerate(range(0, len(gen_samples)-samecondition_num, samecondition_num)):
            papervis_save_path = os.path.join(
                papervis_dir, f"{_prefix}_same_cluster_same_lost_{batch_id}_{_id}.png")
            draw_grid_lost_bbox(
                tensorlist=gen_samples[start_id:start_id+samecondition_num],
                lostmask=data_dict['lostbboxmask'][start_id:start_id +
                                                   samecondition_num],
                original_images=data_dict['image'][start_id:start_id +
                                                   samecondition_num],
                save_path=papervis_save_path,
                nrow=samecondition_num,
                ncol=1,
                mask_alpha=0.4, use_mask=False, padding=_padding
            )

    if should_vis(vis, "same_cluster_diff_lost"):
        for _id, start_id in enumerate(range(0, len(gen_samples)-samecondition_num, samecondition_num)):
            papervis_save_path = os.path.join(
                papervis_dir, f"{_prefix}_same_cluster_diff_lost_{batch_id}_{_id}.png")
            draw_grid_lost_bbox(
                tensorlist=gen_samples[start_id:start_id+samecondition_num],
                lostmask=data_dict['lostbboxmask'][start_id:start_id +
                                                   samecondition_num],
                original_images=data_dict['image'][start_id:start_id +
                                                   samecondition_num],
                save_path=papervis_save_path,
                nrow=samecondition_num,
                ncol=1,
                mask_alpha=0.4, use_mask=False, padding=_padding
            )

    if should_vis(vis, "diff_cluster_same_lost"):
        for _id, start_id in enumerate(range(0, len(gen_samples)-samecondition_num, samecondition_num)):
            papervis_save_path = os.path.join(
                papervis_dir, f"{_prefix}_diff_cluster_same_lost_{batch_id}_{_id}.png")
            draw_grid_lost_bbox(
                tensorlist=gen_samples[start_id:start_id+samecondition_num],
                lostmask=data_dict['lostbboxmask'][start_id:start_id +
                                                   samecondition_num],
                original_images=data_dict['image'][start_id:start_id +
                                                   samecondition_num],
                save_path=papervis_save_path,
                nrow=samecondition_num,
                ncol=1,
                mask_alpha=0.4, use_mask=False, bbox_width=3, padding=_padding
            )

    if should_vis(vis, "same_stego_diff_cluster"):
        for _id, start_id in enumerate(range(0, len(gen_samples)-samecondition_num, samecondition_num)):
            papervis_save_path = os.path.join(
                papervis_dir, f"{_prefix}_same_stego_diff_cluster_{batch_id}_{_id}.png")
            draw_grid_stego(
                tensorlist=gen_samples[start_id:start_id+samecondition_num],
                masks=data_dict['stegomask'][start_id:start_id +
                                             samecondition_num],
                original_images=data_dict['image'][start_id:start_id +
                                                   samecondition_num],
                save_path=papervis_save_path,
                nrow=samecondition_num,
                ncol=1,
                mask_alpha=1, padding=_padding
            )

    if should_vis(vis, "diff_z_same_stego"):
        samecondition_num = 8
        for _id, start_id in enumerate(range(0, len(gen_samples)-samecondition_num, samecondition_num)):
            papervis_save_path = os.path.join(
                papervis_dir, f"{_prefix}_diff_z_same_stego_{batch_id}_{_id}.png")
            draw_grid_stego(
                tensorlist=gen_samples[start_id:start_id+samecondition_num],
                masks=data_dict['stegomask'][start_id:start_id +
                                             samecondition_num],
                original_images=data_dict['image'][start_id:start_id +
                                                   samecondition_num],
                save_path=papervis_save_path,
                nrow=samecondition_num,
                ncol=1,
                mask_alpha=1, padding=_padding
            )

    if should_vis(vis, "interp"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_interp_{batch_id}.png")
        draw_grid_imgsize32_interp(
            tensorlist=gen_samples,
            save_path=papervis_save_path,
            nrow=vis.interp_c.n,
            ncol=vis.interp_c.samples, padding=_padding
        )

    if should_vis(vis, "condscale"):
        papervis_save_path = os.path.join(
            papervis_dir, f"{_prefix}_condscale_{batch_id}.png")
        assert len(data_dict['stegomask']) >= vis.condscale_c.samples
        assert len(data_dict['image']) >= vis.condscale_c.samples

        draw_grid_condscale_stego(masks=data_dict['stegomask'][:vis.condscale_c.samples],
                                  original_images=data_dict['image'][:vis.condscale_c.samples],
                                  tensorlist=gen_samples,
                                  save_path=papervis_save_path,
                                  samples=vis.condscale_c.samples, padding=_padding
                                  )

    if should_vis(vis, "scoremix_vis"):
        raise NotImplementedError
        papervis_save_path = os.path.join(
            papervis_dir, f"{ds_name}_scoremix_vis_{batch_id}.png")
        # pred_x0: [timestep, batch, c, w, h]
        samples = vis.scoremix_vis_c.samples
        interp = vis.scoremix_vis_c.interp
        draw_grid_scoremix_vis(
            tensorlist=gen_samples, nrow=interp, save_path=papervis_save_path
        )


@torch.no_grad()
def eval_fid(
    sample_fn,
    prepare_batch_fn,
    fid_kwargs,
    prefix="",
    debug=False,
    sample_bs=None,
    harddrive_vis_num=100,
):
    logger.warning("begin eval_fid")
    dl_sample = fid_kwargs["dl_sample"]
    fid_num, dataset_name = fid_kwargs["fid_num"], fid_kwargs["dataset_name"]
    sample_dir, save_dir = fid_kwargs["sample_dir"], fid_kwargs["save_dir"]
    gt_dir_4fid, fid_debug_dir = fid_kwargs["gt_dir_4fid"], fid_kwargs["fid_debug_dir"]
    condition_method = fid_kwargs['condition_kwargs']['condition_method']
    vis = fid_kwargs["vis"]
    _exp = fid_kwargs["exp"]
    assert os.path.exists(gt_dir_4fid), f"{gt_dir_4fid} not exists"
    if debug:
        gt_dir_4fid = fid_debug_dir
        logger.warning(f"set debug gt_dir {gt_dir_4fid}")

    if len(prefix) > 0:
        sample_dir = prefix.replace("/", "_") + "_" + sample_dir
    sample_dir = os.path.join(save_dir, sample_dir)

    make_clean_dir(sample_dir)
    papervis_dir = sample_dir + "_papervis"
    make_clean_dir(papervis_dir)
    if should_exp(_exp, 'ablate_scale'):
        cond_gt_img_dir = sample_dir + "_" + 'cond_gt_img'
        make_clean_dir(cond_gt_img_dir)
        cond_gt_mask_dir = sample_dir + "_" + 'cond_gt_mask'
        make_clean_dir(cond_gt_mask_dir)
    ####################################

    global_id_sample = 0
    # sampling
    if sample_bs is None:
        sample_bs = dl_sample.batch_size
    sample_iter = cycle(dl_sample)
    for batch_id in tqdm(
        range(math.ceil(fid_num / sample_bs)),
        desc=f"{prefix}: sampling images for fid",
    ):
        sample_data_dict = next(sample_iter)

        sample_data_dict = prepare_batch_fn(
            sample_data_dict, subset=None)
        ####################################
        sample_data_dict = eval_fid_callback_before(ds_name=dataset_name, condition_method=condition_method,
                                                    vis=vis, _exp=_exp, data_dict=sample_data_dict
                                                    )
        ####################################

        gen_samples, pred_x0 = sample_fn(
            sample_data_dict,
            subset=None,
            batch_id=batch_id,
            batch_size=sample_bs,
            prefix=prefix,
        )

        eval_fid_callback_after(ds_name=dataset_name, condition_method=condition_method,
                                vis=vis,
                                papervis_dir=papervis_dir,
                                gen_samples=gen_samples,
                                pred_x0=pred_x0,
                                batch_id=batch_id,
                                data_dict=sample_data_dict,
                                )
        ####################################

        if should_exp(_exp, 'ablate_scale') and not ds_has_label_info(dataset_name) and condition_method.startswith('stego'):
            _images = clip_unnormalize_to_zero_to_255(
                sample_data_dict['image'])
            _stegomasks = torch.argmax(
                sample_data_dict['stegomask'], 1, keepdim=False)  # [B, H, W]
            for _id, (_sample, _image, _stegomask) in enumerate(zip(gen_samples, _images, _stegomasks)):
                img_pil_save(
                    _sample, os.path.join(
                        sample_dir, f"img{global_id_sample}.png"))
                img_pil_save(
                    _image, os.path.join(
                        cond_gt_img_dir, f"img{global_id_sample}.png"),)
                img_pil_save(
                    _stegomask, os.path.join(
                        cond_gt_mask_dir, f"img{global_id_sample}.png"), pil_mode="L"
                )
                global_id_sample += 1

        else:
            for _id, _sample in enumerate(gen_samples):
                img_pil_save(
                    _sample, os.path.join(
                        sample_dir, f"img{global_id_sample}.png")
                )
                global_id_sample += 1

    result_dict = dict(
        imgs_sample=len(os.listdir(sample_dir)),
        sample_min=gen_samples.float().min().item(),
        sample_max=gen_samples.float().max().item(),
        sample_mean=gen_samples.float().mean().item(),
    )

    fid_dict, fid_for_ckpt = get_fid_dict(
        sample_dir=sample_dir,
        gt_dir=gt_dir_4fid,
        dataset_name=dataset_name,
        debug=debug,
    )
    result_dict.update(fid_dict)

    fid_sample_harddrive = [
        wandb.Image(os.path.join(sample_dir, file_name))
        for file_name in os.listdir(sample_dir)[:harddrive_vis_num]
    ]
    fid_gt_harddrive = [
        wandb.Image(os.path.join(gt_dir_4fid, file_name))
        for file_name in os.listdir(gt_dir_4fid)[:harddrive_vis_num]
    ]
    result_dict.update(
        dict(
            fid_sample_harddrive=fid_sample_harddrive, fid_gt_harddrive=fid_gt_harddrive
        )
    )

    # delete_dir(gt_dir)
    # delete_dir(sample_dir)
    logger.warning("end eval_fid")

    return result_dict, fid_for_ckpt, sample_dir, gt_dir_4fid


if __name__ == "__main__":
    # https://github.com/GaParmar/clean-fid
    # https://github.com/toshas/torch-fidelity

    class dummyDL(object):
        def __init__(
            self,
        ):
            pass

        def __len__(self):
            return int(1e4)

        def __getitem__(self, idx):
            return dict(video=torch.zeros((3, 32, 32)))

    def sample_func(bs):
        return torch.zeros((bs, 3, 256, 256))

    def decode_func(bs):
        return torch.zeros((bs, 3, 256, 256)).float()

    dl = torch.utils.data.DataLoader(
        dummyDL(),
        batch_size=16,
        num_workers=0,
    )

    fid, _, sample_dir, gt_dir = eval_fid(sample_fn=sample_func, fid_num=50)

    print(fid)
