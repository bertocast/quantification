import numpy as np
from sklearn.utils import check_X_y


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


def create_bags_with_multiple_prevalence(X, y, n=1001, random_state=None):

    if random_state:
        np.random.seed(random_state)

    X, y = check_X_y(X, y)
    classes = np.unique(y)
    n_classes = len(classes)
    m = len(X)

    for i in range(n):
        ps = np.random.uniform(0.05, 0.95, n_classes)
        ps /= ps.sum()
        idxs = []

        for n, p in zip(classes, ps.tolist()):
            idx = np.random.choice(np.where(y == n)[0], int(p * m), replace=True)
            idxs.append(idx)

        idxs = np.concatenate(idxs)
        yield X[idxs], y[idxs], ps
