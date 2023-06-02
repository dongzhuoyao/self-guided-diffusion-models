
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger

"""
python sg_sample_segmask.py sample=voc_condscale_s0_gt && 
python sg_sample_segmask.py sample=voc_condscale_s0 && 
python sg_sample_segmask.py sample=voc_condscale_s3 && 
python sg_sample_segmask.py sample=voc_condscale_s3_gt 



"""

_root = '/home/thu/lab/self-guidance/outputs/voc_condscale_stegoclusterlayout/26-09-2022/00-38-52'



from torchmetrics import ConfusionMatrix


s_list = [0,1,2,3,4,0.5,1.5]

for s in s_list:
    
    _cm =ConfusionMatrix(num_classes=21)
    logger.warning('image_Num: ', len(os.listdir(cluster_gt_dir)))
    cluster_gt_dir = os.path.join(_root, f'eval_test_v2_ddim250_s{s}_sample_rank0_cond_gt_img_stego')
    s0_pred_dit = os.path.join(_root, f'eval_test_v2_ddim250_s{s}_sample_rank0_stego')
    for basename in os.listdir(cluster_gt_dir):
        basename = basename.split('.')[0]
        gt = np.array(Image.open(os.path.join(cluster_gt_dir, basename+'.png')))
        pred = np.array(Image.open(os.path.join(s0_pred_dit, basename+'.png')))
        #print(basename)
        #print(np.unique(gt))
        #print(np.unique(pred))
        _cm.update(torch.from_numpy(pred).reshape(-1), torch.from_numpy(gt).reshape(-1))


    current = _cm.compute()
    current = current.numpy()
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    logger.warning(f"guidance strength w= {s}, mIoU={np.mean(IoU)}")
