import numpy as np
from sklearn.utils import check_consistent_length


def kl_divergence(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    if np.any(p_pred == 0):
        raise NotImplementedError
    kl = [p_p * np.log(p_t/p_p) for p_p, p_t in zip(p_pred, p_true)]
    return np.sum(kl)


def absolute_error(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    return np.mean(np.abs(p_pred - p_true))


def normalized_absolute_error(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    return np.sum(np.abs(p_pred - p_true)) / (2 * (1 - np.min(p_true)))


def relative_absolute_error(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    if np.any(p_true == 0):
        raise NotImplementedError
    return np.mean(np.abs(p_pred - p_true) / p_true)


def normalized_relative_absolute_error(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    l = p_true.shape[0]
    return relative_absolute_error(p_true, p_pred) / (l - 1 + (1 - np.min(p_true)) / np.min(p_true))


def bray_curtis(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    return np.sum(np.abs(p_true - p_pred)) / np.sum(p_true + p_pred)

