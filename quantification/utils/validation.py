import numpy as np

from quantification.utils.parallelism import ClusterParallel


def cross_validation_score(estimator, X, y, cv=3, score=None, local=False, **kwargs):
    cv_iter = split(X, cv)
    # TODO: Split data before give it to the cluster. Computational issues.
    kw_args = {'X':X, 'y':y, 'estimator':estimator,'score':score}
    kw_args.update(**kwargs)
    parallel = ClusterParallel(_fit_and_score, cv_iter, kw_args, dependencies=[_score], local=local)
    return parallel.retrieve()


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


def _fit_and_score(train, test, X, y, estimator, score, **kwargs):
    X_train = X[train,]
    y_train = y[train]
    X_test = X[test,]
    y_test = y[test]
    estimator.fit(X_train, y_train)
    return _score(y_test, estimator.predict(X_test), score, **kwargs)


def _score(y_true, y_pred, score, **kwargs):
    if score == "accuracy":
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    if score == "confusion_matrix":
        from sklearn.metrics import confusion_matrix
        if not kwargs['labels']:
            raise ValueError('labels parameter needed to compute the confusion matrix')
        return confusion_matrix(y_true, y_pred, kwargs['labels'])
