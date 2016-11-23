import dispy
import functools
import numpy as np
from sklearn.metrics import confusion_matrix

from quantification.utils.parallelism import ClusterParallel


def setup(X_file, y_file):
    global X, y
    with open(X_file, 'rb') as f_x:
        X = np.load(f_x)

    with open(y_file, 'rb') as f_y:
        y = np.load(f_y)
    return 0


def cleanup():
    global X, y
    del X, y


def wrapper(clf, train, test):
    global X, y
    clf.fit(X[train,], y[train])
    return confusion_matrix(y[test], clf.predict(X[test]))


def parallel_cv_confusion_matrix(clf, X_file, y_file, folds=50):
    cv_iter = split(X, folds)
    cms = []
    # if local:
    #     for train, test in cv_iter:
    #         clf.fit(X[train,], y[train])
    #         cm = confusion_matrix(y[test], clf.predict(X[test]))
    #         cms.append(cm)
    #     return cms

    cluster = dispy.SharedJobCluster(wrapper, depends=[X_file, y_file],
                                     setup=functools.partial(setup, X_file, y_file), cleanup=cleanup,
                                     scheduler_node='dhcp015.aic.uniovi.es')
    try:
        jobs = []
        for train, test in cv_iter:
            job = cluster.submit(clf, train, test)
            jobs.append(job)
        cluster.wait()
        for job in jobs:
            cms.append(job.result)
    except KeyboardInterrupt:
        pass
    finally:
        cluster.print_status()
        cluster.close()
    return cms


def cross_validation_score(estimator, X, y, cv=50, score=None, local=False, **kwargs):
    # Standar number of folds is 50 (See George Forman17. 2008. Quantifying counts and costs via classification)
    cv_iter = split(X, cv)
    # TODO: Split data before give it to the cluster. Computational issues.
    kw_args = {'X': X, 'y': y, 'estimator': estimator, 'score': score}
    kw_args.update(**kwargs)
    parallel = ClusterParallel(_fit_and_score, cv_iter, kw_args, dependencies=[_score], local=local)
    return parallel.retrieve()


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


# TODO: Refactor this in order to not to pass the whole X and y
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
        return confusion_matrix(y_true, y_pred)
