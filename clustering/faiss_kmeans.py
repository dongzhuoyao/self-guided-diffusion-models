# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import sklearn
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

# https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def run_kmeans(feat_train, feat_trainval, cluster_k, niter=20, minp=200, verbose=False):
    n_data, feat_dim = feat_train.shape

    feat_train /= np.linalg.norm(
        feat_train, axis=1, keepdims=True
    )  # notice this step! you cannot operate this centroids when training any more.

    # k-means
    print(f"Training k-means with {cluster_k} centers, feat_dim={feat_dim}...")

    kmeans = faiss.Kmeans(
        d=feat_dim,
        k=cluster_k,
        niter=niter,
        verbose=True,
        gpu=True,
        min_points_per_centroid=minp,
        spherical=False,
    )
    kmeans.train(feat_train.astype(np.float32))

    _, I = kmeans.index.search(feat_trainval, 1)

    print(f"k-means loss evolution: {kmeans.obj}")
    return I.squeeze(-1), kmeans.centroids


def run_nns(feats, features_trainval, k_nn=20, faiss_lib=True, gpu=True):
    print("run nns.......")
    num_imgs, feat_sz = features_trainval.shape
    # feats = feats.astype("float32")
    # K_nn computation takes into account the input sample as the first NN,
    # so we add an extra NN to later remove the input sample.

    sample_nns = -1 * np.ones(
        (num_imgs, k_nn), dtype=np.int64
    )  # [[] for _ in range(num_imgs)]
    sample_nn_radius_all = -1 * np.ones((num_imgs, k_nn), dtype=np.float32)

    k_nn += 1

    if faiss_lib:
        cpu_index = faiss.IndexFlatL2(feat_sz)
        if gpu:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index
            index = gpu_index
        else:
            index = cpu_index
        index.add(feats)
        kth_values, kth_values_arg = index.search(features_trainval, k_nn)
        # kth_values = np.sqrt(kth_values)
        # knn_radii = np.sqrt(kth_values[:, -1])
        knn_radii_all = np.sqrt(kth_values)
    else:
        raise

    for i_sample in range(num_imgs):
        knns = kth_values_arg[i_sample]
        # Discarding the input sample, also seen as the 0-NN.
        knns = knns[1:]
        sample_nns[i_sample, :] = knns
        sample_nn_radius_all[i_sample, :] = knn_radii_all[i_sample][1:]
    assert sample_nns.min() >= 0 and sample_nn_radius_all.min() >= 0
    print("finish computing NNs.")
    return sample_nns, sample_nn_radius_all
