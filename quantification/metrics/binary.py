import numpy as np
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length


def bias(p_true, p_pred):
    return p_pred - p_true


def absolute_error(p_true, p_pred):
    return np.abs(p_pred - p_true)


def square_error(p_true, p_pred):
    return np.power(p_pred - p_true, 2)


def mean_square_error(p_true, p_pred):
    check_consistent_length(p_true, p_pred)
    ae = [square_error(p_t, p_p) for p_t, p_p in zip(p_true, p_pred)]
    return np.mean(ae)


def relative_absolute_error(p_true, p_pred, epsilon=None):
    p_true = check_array(p_true)[0]
    p_pred = check_array(p_pred)[0]

    if (p_true == 0).sum():
        if not epsilon:
            raise ValueError("p_true is zero, so epsilon is required to compute de RSE. Common value is 1/(2|T|)")
        p_true[p_true == 0] = epsilon
        return (np.abs(p_pred - p_true) + epsilon) / (p_true + epsilon)

    return np.abs(p_pred - p_true) / p_true


def symmetric_absolute_error(p_true, p_pred):
    return np.abs(p_pred - p_true) / (p_pred + p_true)


def normalized_absolute_score(p_true, p_pred):
    return 1 - (np.abs(p_pred - p_true) / np.max([p_true, 1 - p_true]))


def normalized_square_score(p_true, p_pred):
    1 - np.power(np.abs(p_pred - p_true) / np.max([p_true, 1 - p_true]), 2)

