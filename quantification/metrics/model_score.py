import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def cv_confusion_matrix(clf, X, y, folds=50, verbose=False):
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    n_classes = len(clf.classes_)
    cms = []

    if verbose:
        print "Computing cross-validation confusion matrix"

    for n, (train, test) in enumerate(cv_iter):
        if verbose:
            print "\t{}/{}".format(n+1, folds)
        clf.fit(X[train,], y[train])
        cm = confusion_matrix(y[test], clf.predict(X[test]), labels=clf.classes_)
        cms.append(cm)
    return np.array(cms)
