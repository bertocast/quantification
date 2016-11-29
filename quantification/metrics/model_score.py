from sklearn.metrics import confusion_matrix
import numpy as np

from quantification.utils.validation import split


def cv_confusion_matrix(clf, X, y, folds=50):
    cv_iter = split(X, folds)
    cms = []

    for train, test in cv_iter:
        clf.fit(X[train,], y[train])
        cm = confusion_matrix(y[test], clf.predict(X[test]))
        cms.append(cm)
    return np.array(cms)
