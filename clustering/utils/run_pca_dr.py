# %%%
import faiss
import numpy as np
from sklearn.decomposition import PCA


def run_pca_faiss(mt, dr_dim=128, downsample_num=100_000):
    mt = mt.astype('float32')
    _pca = faiss.PCAMatrix(mt.shape[1], dr_dim)
    _pca.train(mt)
    assert _pca.is_trained
    tr = _pca.apply(mt)
    return tr


def run_pca_sklearn(mt, feat_trainval,  downsample_num=100_000, svd_solver='full', total_view=4,  _type='separate', pca_shuffle=False):
    mt = mt.astype('float32')
    # using full dimension as component
    pca = PCA(n_components=0.9, svd_solver=svd_solver)
    pca.fit(mt[:downsample_num])
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())
    tr = pca.transform(feat_trainval)
    feat_dim = tr.shape[-1]
    group_size = feat_dim//total_view

    if _type == 'accumulate':
        _list = []
        for i in range(0, total_view):
            _list.append(tr[:, :(i+1)*group_size])
        return _list
    elif _type == 'separate':
        _list = []
        for i in range(0, total_view):
            if not pca_shuffle:
                _list.append(tr[:, i*group_size:(i+1)*group_size])
            else:
                _list.append(tr[:, i::total_view])
        return _list
    else:
        raise ValueError('type should be accumulate or separate')


if __name__ == "__main__":
    mt = np.random.rand(1000, 400)
    r = run_pca_sklearn(mt)
    print(r.shape)

# %%
