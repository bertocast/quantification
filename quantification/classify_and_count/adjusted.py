from quantification.classify_and_count.base import BaseClassifyAndCountModel, ClassifyAndCount
from quantification.utils.validation import cross_validation_score

import numpy as np


class AdjustedCount(BaseClassifyAndCountModel):

    def __init__(self, estimator_class=None, copy_X=True, estimator_params=tuple()):
        self.estimator_class = estimator_class
        self.copy_X = copy_X
        self.estimator_params = estimator_params

    def fit(self, X, y):

        self.cc_ = ClassifyAndCount(estimator_class=self.estimator_class, copy_X=self.copy_X,
                              estimator_params=self.estimator_params)
        self.cc_ = self.cc_.fit(X, y)

        self.tpr_, self.fpr_ = self._performance(self.cc_.estimator, X, y)

        return self


    def predict(self, X):
        prevalence, prob = self.cc_.predict(X)
        return prevalence, self._adjust(prob)

    def _performance(self, estimator, X, y, cv=3):
        confusion_matrix = cross_validation_score(estimator, X, y, cv, score="confusion_matrix")
        tpr = []
        fpr = []
        for cm in confusion_matrix:
            tpr.append(cm[0, 0] / float(cm[0, 0] + cm[1, 0]))
            fpr.append(cm[0, 1] / float(cm[0, 1] + cm[1, 1]))
        return np.mean(tpr), np.mean(fpr)

    def _adjust(self, prob):
        return (prob - self.fpr_) / float(self.tpr_ - self.fpr_)


if __name__ == '__main__':


    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)

    cc = ClassifyAndCount()
    cc.fit(X, y)
    print cc.predict(X)

    ac = AdjustedCount()
    ac.fit(X, y)
    print ac.predict(X)