import numpy as np
import torch
import wandb
from einops import rearrange


#sns.set_theme(style="whitegrid")
#import seaborn as sns
import matplotlib.pyplot as plt


def wandb_scatter_fig(x_list, y_list, dict_key):
    wandb_dict = dict()
    data = [[x, y] for (x, y) in zip(x_list, y_list)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb_dict.update({dict_key: wandb.plot.scatter(table, "x", "y", title=dict_key)})
    return wandb_dict

def vis_timestep_loss(_stat_loss, _stat_t, num_timesteps):

    t_ordered = torch.arange(0, num_timesteps)
    _stat_t_overall = torch.zeros_like(t_ordered)
    _stat_t_overall_counter = torch.zeros_like(t_ordered)
    for _t in range(num_timesteps):
        _all = torch.sum(torch.where(_stat_t == _t, _stat_loss, torch.tensor(0, dtype=torch.float32)))
        _stat_t_overall[_t] = _all + _stat_t_overall[_t]
        _stat_t_overall_counter[_t] = _stat_t_overall_counter[_t] + torch.sum(_stat_t == _t)
    _stat_t_overall_mean = _stat_t_overall / (_stat_t_overall_counter + 1e-10)
    wandb_dict = dict()
    loss_wandb = list(_stat_t_overall.cpu().numpy())
    data = [[x, y] for (x, y) in zip(range(len(loss_wandb)), list(loss_wandb))]
    table = wandb.Table(data=data, columns=["timestep", "Timestep_Loss"])
    wandb_dict.update({"Timestep_Loss_vs_time": wandb.plot.line(table, "timestep", "Timestep_Loss",
                                                                title="Timestep_Loss vs time")})

    lossmean_wandb = list(_stat_t_overall_mean.cpu().numpy())
    data = [[x, y] for (x, y) in zip(range(len(lossmean_wandb)), list(lossmean_wandb))]
    table = wandb.Table(data=data, columns=["timestep", "Timestep_Loss_mean"])
    wandb_dict.update({"Timestep_Loss_mean_vs_time": wandb.plot.line(table, "timestep", "Timestep_Loss_mean",
                                                                     title="Timestep_Loss_mean vs time")})
    return wandb_dict


def vis_schedule_ddpm(_betas, _alphas_cumprod, _snr_derivative):
    _alphas = 1 - _betas
    _mean = np.sqrt(_alphas)
    _std = np.sqrt(1 - _alphas)

    _mean_q_xt_given_x0 = np.sqrt(_alphas_cumprod)
    _std_q_xt_given_x0 = np.sqrt(1-_alphas_cumprod)
    _log_snr = np.log(_alphas_cumprod/(1-_alphas_cumprod+1e-10))
    _timestep = _log_snr.shape[0]

    wandb_dict = dict()
    if _snr_derivative is not None:
        data = [[x, y] for (x, y) in zip(range(_timestep), list(_snr_derivative))]
        table = wandb.Table(data=data, columns=["timestep", "SNR_derivative"])
        wandb_dict.update({"SNR_derivative_vs_time": wandb.plot.line(table, "timestep", "SNR_derivative", title="SNR_derivative vs time")})
    ##
    data = [[x, y] for (x, y) in zip(range(_timestep), list(_log_snr))]
    table = wandb.Table(data=data, columns=["timestep", "logSNR"])
    wandb_dict.update({"logSNR_vs_time": wandb.plot.line(table, "timestep", "logSNR", title="logSNR vs time")})
    ##
    data = [[x, y] for (x, y) in zip(range(_timestep), list(_mean))]
    table = wandb.Table(data=data, columns=["timestep", "mean"])
    wandb_dict.update({"mean_vs_time": wandb.plot.line(table, "timestep", "mean", title="mean vs time")})
    ##
    data = [[x, y] for (x, y) in zip(range(_timestep), list(_std))]
    table = wandb.Table(data=data, columns=["timestep", "std"])
    wandb_dict.update({"std_vs_time": wandb.plot.line(table, "timestep", "std", title="std vs time")})
    ##
    data = [[x, y] for (x, y) in zip(range(_timestep), list(_mean_q_xt_given_x0))]
    table = wandb.Table(data=data, columns=["timestep", "mean_q_xt_given_x0"])
    wandb_dict.update({"mean_q_xt_given_x0_vs_time": wandb.plot.line(table, "timestep", "mean_q_xt_given_x0", title="mean_q_xt_given_x0 vs time")})
    ##
    data = [[x, y] for (x, y) in zip(range(_timestep), list(_std_q_xt_given_x0))]
    table = wandb.Table(data=data, columns=["timestep", "std_q_xt_given_x0"])
    wandb_dict.update({"std_q_xt_given_x0_vs_time": wandb.plot.line(table, "timestep", "std_q_xt_given_x0", title="std_q_xt_given_x0 vs time")})
    return wandb_dict


def mask_vis_bybatch_router(ml_feat, dataset_name='undefined'):
        if dataset_name=='voc':
            if len(ml_feat.shape) == 3:#[B,W,H]
                return vocmask_vis(ml_feat=ml_feat)
            elif len(ml_feat.shape) == 4:
                assert ml_feat.shape[1] == 21
                return vocmask_vis(ml_feat=ml_feat.argmax(1))
            else:raise ValueError( len(ml_feat.shape))
        elif dataset_name=='celebamask':
            if len(ml_feat.shape)==3:
                return cm_vis(ml_feat=ml_feat)
            elif len(ml_feat.shape)==4:
                assert ml_feat.shape[1]==19
                return cm_vis(ml_feat=ml_feat.argmax(1))
            else:raise ValueError(len(ml_feat.shape))
        else:raise ValueError(dataset_name)

def id2name(dataset_name='undefined'):
    if dataset_name == 'voc':
            labels =  [#https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
    elif dataset_name == 'celebamask':
        labels = ['class0_placeholder',
                               'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                               'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    else:
        raise
    labels_id2_name = dict()
    for i, l in enumerate(labels):
        labels_id2_name[i] = l
    return labels_id2_name


def dataset_class_num(dataset_name='undefined'):
    _id2name = id2name(dataset_name)
    return len(_id2name.keys())

def vocmask_vis(ml_feat):
        # B,C,W,H
        b, w, h = ml_feat.shape
        feat_result = ml_feat
        feat_result_list = []
        for _b in range(b):
            tmp = decode_segmap(feat_result[_b].cpu().numpy(), dataset='pascal')
            feat_result_list.append(tmp[np.newaxis, :])
        feat_result = np.concatenate(feat_result_list, 0)
        feat_result = torch.from_numpy(feat_result).to(torch.float32)
        return rearrange(feat_result, 'b w h c-> b c w h')



def cm_vis(ml_feat):
        # B,C,W,H
        b, w, h = ml_feat.shape
        feat_result_list = []
        for _b in range(b):
            tmp = cm_parsing_maps(parsing_anno=ml_feat[_b].unsqueeze(-1).cpu().numpy(), img_origin=None)
            feat_result_list.append(tmp[np.newaxis, :])
        feat_result = np.concatenate(feat_result_list, 0)
        feat_result = torch.from_numpy(feat_result).to(torch.float32)
        return rearrange(feat_result, 'b w h c-> b c w h')