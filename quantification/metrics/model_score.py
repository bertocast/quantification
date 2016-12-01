from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold

from quantification.utils.validation import split


def cv_confusion_matrix(clf, X, y, folds=50):
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    cms = []

    for train, test in cv_iter:
        clf.fit(X[train,], y[train])
        cm = confusion_matrix(y[test], clf.predict(X[test]), labels=clf.classes_)
        cms.append(cm)
    return np.array(cms)
