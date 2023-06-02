import argparse
from datetime import datetime
from pathlib import Path
import shutil
from loguru import logger
from sklearn.metrics import normalized_mutual_info_score


import torch
from clustering.utils.cluster_emsemble import clustering_ensemble
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
import io
import os
import torch.nn.functional as F
import random
import h5py
from tqdm import tqdm
from clustering.cal_cluster_metric import cal_cluster_metric
from clustering.faiss_kmeans import run_kmeans, run_nns
from clustering.utils.cluster_standard import clustering
from clustering.utils.cluster_pca import clustering_pca
from clustering.utils.cluster_patch import clustering_withpatches
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


if __name__ == "__main__":
    # python cluster_ds.py --feat_from vitbase --ds in32 --bs 256 --k 8192
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feat_h5",
        type=str,
        default="your_feat.h5",
        help="the output image",
    )

    p.add_argument("--k", type=int, default=1000, help="the output image")
    p.add_argument("--nns", type=int, default=-1, help="the output image")
    p.add_argument("--niter", type=int, default=30, help="the output image")
    p.add_argument("--minp", type=int, default=200, help="the output image")
    p.add_argument("--p", type=float, default=1.0, help="the output image")
    p.add_argument("--pca", type=int, default=0, help="the output image")
    p.add_argument("--pca_shuffle", type=int,
                   default=0, help="the output image")

    p.add_argument(
        "--ensemble_num", type=int, default=0, help="the output image"
    )
    p.add_argument("--debug", type=int, default=1, help="the output image")

    p.add_argument("--cluster_h5_root", type=str, default=None)
    args = p.parse_args()

    if "patches" in args.feat_h5:
        clustering_withpatches(
            src_feat_h5py_path=args.feat_h5,
            cluster_k=args.k,
            niter=args.niter,
            minp=args.minp,
            cluster_h5_root=args.cluster_h5_root,
            debug=args.debug,
        )
    elif args.ensemble_num > 0:
        clustering_ensemble(feat_h5_path=args.feat_h5,
                            nns=args.nns,
                            pca_shuffle=args.pca_shuffle,
                            cluster_k=args.k,
                            niter=args.niter,
                            minp=args.minp,
                            cluster_h5_root=args.cluster_h5_root,
                            debug=args.debug, ensemble_num=args.ensemble_num)

    elif args.pca:
        clustering_pca(feat_h5_path=args.feat_h5,
                       nns=args.nns,
                       pca_shuffle=args.pca_shuffle,
                       cluster_k=args.k,
                       niter=args.niter,
                       minp=args.minp,
                       cluster_h5_root=args.cluster_h5_root,
                       debug=args.debug)
    else:
        clustering(
            feat_h5_path=args.feat_h5,
            nns=args.nns,
            cluster_k=args.k,
            niter=args.niter,
            minp=args.minp,
            cluster_h5_root=args.cluster_h5_root,
            debug=args.debug,
        )
