# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import pickle
from einops import rearrange, repeat, reduce
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image
from kmeans_faiss import run_kmeans

from networks import get_model
from datasets import  ImageDIRDataset_V2, ImageDataset, Dataset, bbox_iou
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion
from object_discovery import lost, detect_box, dino_seg
from torchvision.io import read_image
import skimage.io
import h5py

import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from loguru import logger
import torch 
from einops import rearrange



if __name__ == "__main__":
    """
    make lost grid:
    python main_lost.py --papervis 1 --dataset COCO20k
    """
    parser = argparse.ArgumentParser("Unsupervised object discovery with LOST.")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "resnet50",
            "vgg16_imagenet",
            "resnet50_imagenet",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        #default="VOC12AUG",
        default="COCO20k",
        type=str,
        choices=[None, "VOC07", "VOC12","VOC12AUG", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="trainval",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        #default='/home/thu/dataset/VOCdevkit/VOC2012/JPEGImages',
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="dataset/data_files/lost", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["fms", "seed_expansion", "pred", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # For ResNet dilation
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")

    # LOST parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)
    parser.add_argument("--papervis", type=int, default=0)

    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    if args.papervis:
        args.visualize='pred'

    if args.dataset=='COCO20k':
        args.set='train'
    elif args.dataset=='VOC12AUG':
        args.set = 'trainval'

    if args.image_path is not None:
        #args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        #dataset = ImageDataset(args.image_path)
        dataset = ImageDIRDataset_V2(args.image_path)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, args.resnet_dilate, device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with LOST
        exp_name = f"LOST-{args.arch}"
        if "resnet" in args.arch:
            exp_name += f"dilate{args.resnet_dilate}"
        elif "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"
    
    exp_name = exp_name+f'_{dataset.name}'

    print(f"Running LOST on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if True or args.visualize:
        vis_folder = f"outputs/output_vis/lost/{dataset.name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    preds_featpooled_dict = {}
    preds_imgpatch_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        channel_num = img.shape[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            channel_num,
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu
        img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    # Modality selection
                    if args.which_features == "k":
                        feats = k[:, 1:, :]
                    elif args.which_features == "q":
                        feats = q[:, 1:, :]
                    elif args.which_features == "v":
                        feats = v[:, 1:, :]

            elif "resnet" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            elif "vgg16" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            else:
                raise ValueError("Unknown model.")

        # ------------ Apply LOST -------------------------------------------
        if not args.dinoseg:
            pred, pred_feats, A, scores, seed = lost(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                k_patches=args.k_patches,
            )

        
            
            feats = rearrange(feats, '1 (w h) c-> w h c',w=w_featmap)
            
            #  

            # ------------ Visualizations -------------------------------------------
            if args.visualize == "fms":
                visualize_fms(A.clone().cpu().numpy(), seed, scores, [w_featmap, h_featmap], scales, vis_folder, im_name)

            elif args.visualize == "seed_expansion":
                image = dataset.load_image(im_name)

                # Before expansion
                pred_seed, _ = detect_box(
                    A[seed, :],
                    seed,
                    [w_featmap, h_featmap],
                    scales=scales,
                    initial_im_size=init_image_size[1:],
                )
                visualize_seed_expansion(image, pred, seed, pred_seed, scales, [w_featmap, h_featmap], vis_folder, im_name)

            elif args.visualize == "pred":
                image = dataset.load_image(im_name)
                visualize_predictions(image, pred, seed, scales, [w_featmap, h_featmap], vis_folder, im_name)

        # Save the prediction
        preds_dict[im_name] = pred
        preds_featpooled_dict[im_name] = feats.mean([0,1])
        if True:
             image = dataset.load_image(im_name)
             x0,x1 = (int(pred[0]), int(pred[2]))
             y0,y1 = (int(pred[1]), int(pred[3]))
             preds_imgpatch_dict[im_name] = image[y0:y1,x0:x1]
             if False:
                pltname = f"{vis_folder}/LOST_patchedv2_{im_name}.png"
                Image.fromarray(preds_imgpatch_dict[im_name]).save(pltname)

        
        if  args.debug and im_id>500:break
        if args.papervis and im_id>50:
            break

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")
        
    

    if args.papervis:
        _resize = 256
        nrow = 4
        ncol = 4
        padding = 2
        img_list = []
        for img_name in os.listdir(vis_folder):
            img = Image.open(os.path.join(vis_folder, img_name)).convert('RGB')
            img = img.resize((_resize,_resize), Image.BILINEAR)
            img = np.array(img).astype(np.uint8)
            img_list.append(img[np.newaxis,:,:,:])

        img_list = img_list[:nrow*ncol]
        assert len(img_list)==nrow*ncol
        tensorlist = torch.from_numpy(np.concatenate(img_list,0))
        tensorlist = rearrange(tensorlist, 'b w h c-> b c w h')
        grid = make_grid(tensorlist, nrow=nrow, padding=padding, pad_value=255.0)
        img = torchvision.transforms.ToPILImage()(grid)
        save_path = os.path.join(vis_folder,'full_size.png')
        img.save(save_path)
        print(save_path)
        exit(0)
            


    #TODO, do k-clustering on feats, and set cluster-id for every bbox of per image.
    kmeans_k = 100
    in_names, feat_np = [], []
    for k,v in preds_featpooled_dict.items():
        in_names.append(k)
        feat_np.append(rearrange(v,'c->1 c'))
    feat_np = torch.cat(feat_np, 0).cpu().numpy()
    trainval_assigned, centroids = run_kmeans(feat_np, feat_np, cluster_k=kmeans_k, niter=20, minp=200, verbose=False)
    
    #add kmeans info
    new_dict = dict()
    for k,v in preds_dict.items():
        new_dict[k]=dict(cluster_id=trainval_assigned[in_names.index(k)],bbox=v)
    preds_dict = new_dict

    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"lost_k{kmeans_k}.h5")
        f = h5py.File(filename, mode='w')
        f.close()
        f = h5py.File(filename, mode='a')
        for k,v in preds_dict.items():
            f.create_dataset(f'{k}_clusterid', data=v['cluster_id'], dtype=np.int64)
            f.create_dataset(f'{k}_bbox',  data=v['bbox'], dtype=np.int64)

        dset = f.create_dataset('all_attributes', (1,))
        dset.attrs['cluster_k'] = kmeans_k

        f.close()
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)
