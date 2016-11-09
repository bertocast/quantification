import numpy as np
from sklearn.utils import indexable

from quantification.classify_and_count import BaseClassifyAndCountModel
from quantification.utils.parallelism import ClusterParallel
from quantification.utils.validation import split, cross_validation_score


class BinaryAdjustedCount(BaseClassifyAndCountModel):

    def __init__(self):

        super(BinaryAdjustedCount, self).__init__()
        self.fpr_, self.tpr_ = [], []

    def _predict(self, X):
        pass

    def fit(self, X, y, local=False):
        X, y = np.array(X), np.array(y)
        for sample in y:
            if len(np.unique(sample)) != 2:
                raise ValueError('Number of classes must be 2 for a binary quantification problem')
        split_iter = split(X, len(X))
        parallel = ClusterParallel(fit_and_performance_wrapper, split_iter, {'X': X, 'y': y, 'quantifier': self},
                                   local=local)
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


def fit_and_performance_wrapper(train, perf, X, y, quantifier):
    return quantifier.fit_and_performance(train, perf, X, y)

if __name__ == '__main__':
    from quantification.datasets.base import load_folder
    data = load_folder("../datasets/data/cancer")
    for i in range(len(data.target)):
        p = np.random.permutation(len(data.target[i]))
        data.data[i] = data.data[i][p]
        data.target[i] = data.target[i][p]
    X = data.data
    y = data.target
    ac = BinaryAdjustedCount()
    ac.fit(X, y, local=True)
