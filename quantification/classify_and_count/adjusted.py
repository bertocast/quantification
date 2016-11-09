from quantification.classify_and_count.base import BaseClassifyAndCountModel, ClassifyAndCount
from quantification.utils.validation import cross_validation_score

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
        self.confusion_matrix = np.mean(
            cross_validation_score(self.cc_.estimator, X, y, 3, score="confusion_matrix"),
            0)
        if len(self.cc_.labels_) == 2:
            self.tpr_, self.fpr_ = self._performance()
        elif len(self.cc_.labels_) >= 2:
            self.conditional_prob = np.empty((len(self.cc_.labels_), len(self.cc_.labels_)))
            for i in range(len(self.cc_.labels_)):
                self.conditional_prob[i] = self.confusion_matrix[i] / np.sum(self.confusion_matrix[i])
        else:
            raise ValueError("Number of class cannot be less than 2")
        return self

    def _predict(self, X):
        if len(self.cc_.labels_) == 2:
            return self.predict_binary(X)
        elif len(self.cc_.labels_) >= 2:
            return self.predict_multiclass(X)
        else:
            raise ValueError("Number of class cannot be less than 2")

    def _performance(self):
        tpr = self.confusion_matrix[0, 0] / float(self.confusion_matrix[0, 0] + self.confusion_matrix[1, 0])
        fpr = self.confusion_matrix[0, 1] / float(self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return tpr, fpr

    def _adjust(self, prob):
        return (prob - self.fpr_) / float(self.tpr_ - self.fpr_)

    def predict_binary(self, X):
        probabilities = self.cc_.predict(X)
        prevalences = self._adjust(probabilities)
        return prevalences

    def predict_multiclass(self, X):
        probabilities = self.cc_.predict(X)
        prevalences = np.linalg.solve(np.matrix.transpose(self.conditional_prob), probabilities)
        return prevalences

