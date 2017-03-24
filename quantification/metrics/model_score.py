import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def cv_confusion_matrix(clf, X, y, folds=50, verbose=False):
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    cms = []

    for n, (train, test) in enumerate(cv_iter):
        if verbose:
            print "Computing CM {}/{}".format(n, folds)
        clf.fit(X[train,], y[train])
        cm = confusion_matrix(y[test], clf.predict(X[test]), labels=clf.classes_)
        cms.append(cm)
    return np.array(cms)
