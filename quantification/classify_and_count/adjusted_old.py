from quantification.classify_and_count.base import BaseClassifyAndCountModel, ClassifyAndCount
from quantification.utils.validation import cross_validation_score, split

import numpy as np


class AdjustedCount(BaseClassifyAndCountModel):

    def __init__(self, estimator_class=None, copy_X=True, estimator_params=tuple()):
        self.estimator_class = estimator_class
        self.copy_X = copy_X
        self.estimator_params = estimator_params

    def _fit(self, X, y):

        self.cc_ = ClassifyAndCount(estimator_class=self.estimator_class,
                                    estimator_params=self.estimator_params)
        self.cc_ = self.cc_.fit(X, y)
        return self

    def _predict(self, X):
        if len(self.cc_.labels_) == 2:
            return self.predict_binary(X)
        elif len(self.cc_.labels_) >= 2:
            return self.predict_multiclass(X)
        else:
            raise ValueError("Number of class cannot be less than 2")

    def _performance(self, clf_idx, X, y):
        confusion_matrix = np.mean(
            cross_validation_score(self.cc_.estimator, X, y, 3, score="confusion_matrix"),
            0)
        if len(self.cc_.labels_) == 2:
            self.tpr_ = confusion_matrix[0, 0] / float(confusion_matrix[0, 0] + confusion_matrix[1, 0])
            self.fpr_ = confusion_matrix[0, 1] / float(confusion_matrix[0, 1] + confusion_matrix[1, 1])
        elif len(self.cc_.labels_) >= 2:
            self.conditional_prob_ = np.empty((len(self.cc_.labels_), len(self.cc_.labels_)))
            for i in range(len(self.cc_.labels_)):
                self.conditional_prob_[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])
        else:
            raise ValueError("Number of class cannot be less than 2")
        return self

    def _adjust(self, prob):
        return (prob - self.fpr_) / float(self.tpr_ - self.fpr_)

    def predict_binary(self, X):
        probabilities = self.cc_.predict(X)
        prevalences = self._adjust(probabilities)
        return prevalences

    def predict_multiclass(self, X):
        probabilities = self.cc_.predict(X)
        prevalences = np.linalg.solve(np.matrix.transpose(self.conditional_prob_), probabilities)
        return prevalences

    def fit(self, X, y):
        self._fit(X, y)
        if len(X) == 1:
            raise NotImplementedError

        X, y = np.asarray(X), np.asarray(y)
        cv_iter = split(X, X.shape[0])
        for perf, train in cv_iter:
            self._performance(train[0], X[perf], y[perf])

