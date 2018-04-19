from __future__ import print_function
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def cv_confusion_matrix(clf, X, y, folds=50, verbose=False):
    clf_c = deepcopy(clf)
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    n_classes = len(clf_c.classes_)
    cms = []

    if verbose:
        print("Computing cross-validation confusion matrix")

    for n, (train, test) in enumerate(cv_iter):
        if verbose:
            print("\t{}/{}".format(n+1, folds))
        clf_c.fit(X[train,], y[train])
        cm = confusion_matrix(y[test], clf_c.predict(X[test]), labels=clf_c.classes_)
        cms.append(cm)
    return np.array(cms)
