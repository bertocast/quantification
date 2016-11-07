import logging

import numpy as np
import dispy

def cross_validation_score(estimator, X, y, cv=3, score=None):
    cv_iter = list(split(X, cv))

    dependencies = [_score]
    cluster = dispy.JobCluster(_fit_and_score, depends=dependencies, loglevel=logging.ERROR)
    jobs = []
    for train, test in cv_iter:
        job = cluster.submit(estimator=estimator, X=X, y=y, train=train, test=test, score=score)
        jobs.append(job)
    cluster.wait()

    scores = []
    for job in jobs:
        job()
        scores.append(job.result)

    return np.array(scores)


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


def _fit_and_score(estimator, X, y, train, test, score):
    X_train = X[train,]
    y_train = y[train]
    X_test = X[test,]
    y_test = y[test]
    estimator.fit(X_train, y_train)
    return _score(y_test, estimator.predict(X_test), score)


def _score(y_true, y_pred, score):
    if score == "accuracy":
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    if score == "confusion_matrix":
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)