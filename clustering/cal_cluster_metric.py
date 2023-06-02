

import numpy as np 
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score


def cal_cluster_metric(gt_np, pred_np, need_nmi=True, need_ami=True, need_ari=True):
    labels_pred_num= len(list(np.unique(gt_np)))
    result_dict = dict(labels_pred_num=labels_pred_num)
    if need_nmi:
        result_dict['nmi'] = normalized_mutual_info_score(gt_np, pred_np)
    if need_ami:
        result_dict['ami'] = adjusted_mutual_info_score(labels_true=gt_np, labels_pred=pred_np)
    if need_ari:
        result_dict['ari'] = adjusted_rand_score(labels_pred=pred_np, labels_true=gt_np)
    return result_dict 