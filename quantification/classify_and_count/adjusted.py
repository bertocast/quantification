import numpy as np

from quantification.classify_and_count.base import BaseClassifyAndCountModel, predict_wrapper_per_clf
from quantification.utils.parallelism import ClusterParallel
from quantification.utils.validation import split, cross_validation_score


class BinaryAdjustedCount(BaseClassifyAndCountModel):

    def __init__(self):

        super(BinaryAdjustedCount, self).__init__()
        self.fpr_, self.tpr_ = None, None

    def _predict(self, X):
        parallel = ClusterParallel(predict_wrapper_per_clf, self.estimators_, {'X': X}, local=True)  # TODO: Fix this
        predictions = parallel.retrieve()
        maj = np.argmax(np.average(predictions, axis=0, weights=None), axis=1)
        freq = np.bincount(maj)
        relative_freq = freq / float(np.sum(freq))
        adjusted = self._adjust(relative_freq)
        return adjusted

    def fit(self, X, y, local=False):
        X, y = np.array(X), np.array(y)
        for sample in y:
            if len(np.unique(sample)) != 2:
                raise ValueError('Number of classes must be 2 for a binary quantification problem')
        split_iter = split(X, len(X))
        parallel = ClusterParallel(fit_and_performance_wrapper, split_iter, {'X': X, 'y': y, 'quantifier': self},
                                   local=True)  # TODO: Fix this
        clfs, tpr, fpr = zip(*parallel.retrieve())
        self.estimators_.extend(clfs)
        self.tpr_ = np.mean(tpr)
        self.fpr_ = np.mean(fpr)
        return self

    def fit_and_performance(self, perf, train, X, y):
        clf = self._fit(X[train[0]], y[train[0]])
        confusion_matrix = np.mean(
            cross_validation_score(clf, np.concatenate(X[perf]), np.concatenate(y[perf]), 3, score="confusion_matrix"),
            0)
        tpr = confusion_matrix[0, 0] / float(confusion_matrix[0, 0] + confusion_matrix[1, 0])
        fpr = confusion_matrix[0, 1] / float(confusion_matrix[0, 1] + confusion_matrix[1, 1])
        return clf, tpr, fpr

    def _adjust(self, prob):
        return (prob - self.fpr_) / float(self.tpr_ - self.fpr_)


def fit_and_performance_wrapper(train, perf, X, y, quantifier):
    return quantifier.fit_and_performance(train, perf, X, y)


