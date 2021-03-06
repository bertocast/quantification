import functools
import logging
from os.path import basename

import dispy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from quantification.utils.errors import ClusterException


def setup(data_file):
    global X, y
    import numpy as np
    with open(data_file, 'rb') as fh:
        data = np.load(fh)
        X = data['X']
        y = data['y']
    return 0


def wrapper(clf, train, test, pos_class=None):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    if pos_class is None:
        clf.fit(X[train], y[train])
        return confusion_matrix(y[test], clf.predict(X[test]))

    mask = (y[train] == pos_class)
    y_bin_train = np.ones(y[train].shape, dtype=np.int)
    y_bin_train[~mask] = 0
    clf.fit(X[train,], y_bin_train)

    mask = (y[test] == pos_class)
    y_bin_test = np.ones(y[test].shape, dtype=np.int)
    y_bin_test[~mask] = 0

    return confusion_matrix(y_bin_test, clf.predict(X[test]), labels=clf.classes_)


def cleanup():
    global X, y
    del X, y


def cv_confusion_matrix(clf, X, y, data_file, pos_class=None, folds=50, verbose=False):
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    cms = []
    cluster = dispy.SharedJobCluster(wrapper,
                                     depends=[data_file],
                                     reentrant=True,
                                     setup=functools.partial(setup, basename(data_file)),
                                     cleanup=cleanup,
                                     scheduler_node='pomar.aic.uniovi.es',
                                     loglevel=logging.ERROR)
    try:
        jobs = []
        for train, test in cv_iter:
            job = cluster.submit(clone(clf), train, test, pos_class)
            jobs.append(job)
        for job in jobs:
            job()
            if job.exception:
                raise ClusterException(job.exception + job.ip_addr)
            cms.append(job.result)
    except KeyboardInterrupt:
        cluster.close()
    if verbose:
        cluster.print_status()
    cluster.close()
    return np.array(cms)
