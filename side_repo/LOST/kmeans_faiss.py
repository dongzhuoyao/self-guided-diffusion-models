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

#https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def run_kmeans(features_train, features_trainval, cluster_k, niter=20, minp=200, verbose=False):
    n_data, feat_dim = features_train.shape
    # Normalize features
    features_train /= np.linalg.norm(features_train, axis=1, keepdims=True)#notice this step! you cannot operate this centroids when training any more.
 
    # k-means
    print("Training k-means with %i centers..." % (cluster_k))
    kmeans = faiss.Kmeans(
        d=feat_dim,
        k=cluster_k,
        niter=niter,
        verbose=True,
        gpu=True,
        min_points_per_centroid=minp,
        spherical=False,
    )
    kmeans.train(features_train.astype(np.float32))

    _, I = kmeans.index.search(features_trainval, 1)
    # Find closest instances to each k-means cluster
    print("Finding closest instances to centers...")
    if False:
        index = faiss.IndexFlatL2(feat_dim)
        index.add(features_trainval.astype(np.float32))
        D, closest_sample = index.search(kmeans.centroids, 1)
    print('k-means loss evolution: {0}'.format(kmeans.obj))
    return I.squeeze(-1), kmeans.centroids
    #return closest_sample.squeeze(-1), kmeans.centroids


def run_nns(feats, features_trainval, k_nn=20, faiss_lib=True,  gpu=True):
    """
    It obtains the neighborhoods for all instances using the k-NN algorithm.

    Parameters
    ----------
    k_nn: int, optional
        Number of neighbors (k).
    faiss_lib: bool, optional
        If True, use the faiss library implementation of k-NN. If not, use the slower
        implementation of sklearn.
    feat_sz: int, optional
        Feature dimensionality.
    gpu: bool, optional
        If True, leverage GPU resources to speed up computation with the faiss library.

    """
    print('run nns.......')
    num_imgs, feat_sz =features_trainval.shape
    #feats = feats.astype("float32")
    # K_nn computation takes into account the input sample as the first NN,
    # so we add an extra NN to later remove the input sample.
   
    sample_nns = -1 * np.ones((num_imgs, k_nn), dtype=np.int64)#[[] for _ in range(num_imgs)]
    sample_nn_radius_all = -1 * np.ones((num_imgs,k_nn), dtype=np.float32)

    k_nn += 1

    if faiss_lib:
        cpu_index = faiss.IndexFlatL2(feat_sz)
        if gpu:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index
            index = gpu_index
        else:
            index = cpu_index
        index.add(feats)
        kth_values, kth_values_arg = index.search(
            features_trainval, k_nn
        )
        #kth_values = np.sqrt(kth_values)
        #knn_radii = np.sqrt(kth_values[:, -1])
        knn_radii_all = np.sqrt(kth_values)

    else:
        raise 
        dists = sklearn.metrics.pairwise_distances(
            feats, feats, metric="euclidean", n_jobs=-1
        )
        print("Computed distances.")
        knn_radii, kth_values_arg = self._get_kth_value_accurate(dists, k_nn)
    for i_sample in range(num_imgs):
        knns = kth_values_arg[i_sample]
        # Discarding the input sample, also seen as the 0-NN.
        knns = knns[1:]
        sample_nns[i_sample,:] = knns
        sample_nn_radius_all[i_sample,:] = knn_radii_all[i_sample][1:]
    assert sample_nns.min()>=0 and sample_nn_radius_all.min()>=0
    print("Computed NNs.")
    return sample_nns,sample_nn_radius_all