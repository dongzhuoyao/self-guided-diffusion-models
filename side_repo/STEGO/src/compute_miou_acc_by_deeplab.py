
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm


_root = '/home/thu/lab/self-guidance/outputs/voc_condscale_stegoclusterlayout/26-09-2022/00-38-52'

from torchmetrics import ConfusionMatrix

s_list = [0,1,2,3,4,0.5,1.5]




def get_backbone():
    from torchvision.io.image import read_image
    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    from torchvision.transforms.functional import to_pil_image

    # Step 1: Initialize model with the best available weights
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()


    def extract_segmask_fn(input_path):
        img = read_image(input_path)
        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and visualize the prediction
        prediction = model(batch)["out"]
        normalized_masks = prediction.softmax(dim=1)
        class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        #mask = normalized_masks[0, class_to_idx["dog"]]
        #to_pil_image(mask).show()
        return normalized_masks

    return extract_segmask_fn


extract_segmask_fn = get_backbone()

for s in s_list:
    
    _cm =ConfusionMatrix(num_classes=21)
    img_num = len(os.listdir(cluster_gt_dir))
    logger.warning('image_Num: ', img_num)
    cluster_gt_dir = os.path.join(_root, f'eval_test_v2_ddim250_s{s}_sample_rank0_cond_gt_img')
    s0_pred_dit = os.path.join(_root, f'eval_test_v2_ddim250_s{s}_sample_rank0')
    for basename in tqdm(os.listdir(cluster_gt_dir)):
        basename = basename.split('.')[0]
        gt_path = os.path.join(cluster_gt_dir, basename+'.png')
        pred_path  = os.path.join(s0_pred_dit, basename+'.png')
        gt_mask = extract_segmask_fn(gt_path)
        pred_mask = extract_segmask_fn(pred_path)

        _cm.update(torch.from_numpy(gt_mask).reshape(-1), torch.from_numpy(pred_mask).reshape(-1))


    current = _cm.compute()
    current = current.numpy()
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    logger.warning(f"guidance strength w= {s}, mIoU={np.mean(IoU)}")
