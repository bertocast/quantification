import numpy as np


def create_partitions(X, y, n_splits):
    cv_iter = split(X, n_splits)
    for val, train in cv_iter:
        yield X[train[0]], y[train[0]], np.concatenate(X[val]), np.concatenate(y[val])


def split(X, n_splits):
    indices = np.arange(X.shape[0])
    for test_index in _iter_test_masks(X, n_splits):
        train_index = indices[np.logical_not(test_index)]
        test_index = indices[test_index]
        yield train_index, test_index


def _iter_test_masks(X, n_splits):
    for test_index in _iter_test_indices(X, n_splits):
        test_mask = np.zeros(X.shape[0], dtype=np.bool)
        test_mask[test_index] = True
        yield test_mask


def _iter_test_indices(X, n_splits):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        yield indices[start:stop]
        current = stop
