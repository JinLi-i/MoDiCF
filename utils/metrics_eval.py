# encoding: utf-8
# @email: enoche.chow@gmail.com
"""
############################
"""

from logging import getLogger

import numpy as np

def fair_incomplete_(pos_index):
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)


def fair_proportion_(pos_index, p_dataset, eps=1e-2):
    cumulative_sums = np.cumsum(pos_index, axis=1)

    ranks = np.arange(1, pos_index.shape[1] + 1)

    p_recom = cumulative_sums / ranks

    fair_proportion = 1 - np.abs(p_recom - p_dataset) / max(p_dataset, eps)

    return fair_proportion.mean(axis=0)


def f_fair_(fair, precision):
    assert len(fair) == len(precision), "Length of fair and precision should be the same"
    fpp = fair + precision
    f_fair = np.where(fpp != 0, 2 * fair * precision / fpp, 0)
    return f_fair

def recall_(pos_index, pos_len):
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def recall2_(pos_index, pos_len):
    rec_cum = np.cumsum(pos_index, axis=1)
    rec_ret = rec_cum.sum(axis=0) / pos_len.sum()
    return rec_ret


def ndcg_(pos_index, pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)


def hit_rate_(pos_index, pos_len):
    hits = (pos_index.cumsum(axis=1) > 0).astype(float)

    hit_rate = hits.mean(axis=0)

    return hit_rate

def map_(pos_index, pos_len):
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index.astype(float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=float)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1]+1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


def precision_(pos_index, pos_len):
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)


metrics_dict = {
    'ndcg': ndcg_,
    'recall': recall_,
    'recall2': recall2_,
    'precision': precision_,
    'map': map_,
    'hit_ratio': hit_rate_,
    'fair_incomplete': fair_incomplete_,
    'fair_p': fair_proportion_,
    'f_fair': f_fair_,
}
