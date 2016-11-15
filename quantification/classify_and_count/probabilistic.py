import numpy as np

from quantification.classify_and_count import BaseClassifyAndCountModel
from quantification.classify_and_count.base import fit_wrapper
from quantification.utils.parallelism import ClusterParallel
from quantification.utils.validation import split, cross_validation_score


class ProbabilisticClassifyAndCount(BaseClassifyAndCountModel):
    def _predict(self, X):
        parallel = ClusterParallel(predict_proba_wrapper_per_clf, self.estimators_, {'X': X},
                                   local=True)  # TODO: Fix this
        predictions = parallel.retrieve()
        maj = np.average(predictions, axis=0, weights=None)
        prevalence = np.average(maj, axis=0)
        return prevalence

    def fit(self, X, y, local=False):
        if not isinstance(X, list):
            clf = self._fit(X, y)
            self.estimators_.append(clf)
            return self
        parallel = ClusterParallel(fit_wrapper, zip(X, y), {'quantifier': self}, local=local)
        clfs = parallel.retrieve()
        self.estimators_.extend(clfs)
        return self


class ProbabilisticBinaryAdjustedCount(BaseClassifyAndCountModel):
    def __init__(self):
        super(ProbabilisticBinaryAdjustedCount, self).__init__()
        self.tp_pa_, self.fp_pa_ = None, None
        self.tn_pa_, self.fn_pa_ = None, None

    def _predict(self, X):
        parallel = ClusterParallel(predict_proba_wrapper_per_clf, self.estimators_, {'X': X},
                                   local=True)  # TODO: Fix this
        predictions = parallel.retrieve()
        maj = np.average(predictions, axis=0, weights=None)
        prevalence = np.average(maj, axis=0)
        adjusted = self._adjust(prevalence)
        return adjusted

    def fit(self, X, y, local=False):
        """if not isinstance(X, list):
            clf = self._fit(X, y)
            self.tp_pa_, self.fp_pa_, self.tn_pa_, self.fn_pa_ = self._performance(X, y, clf, local)
            self.estimators_.append(clf)
            return self

        X, y = np.array(X), np.array(y)
        for sample in y:
            if len(np.unique(sample)) != 2:
                raise ValueError('Number of classes must be 2 for a binary quantification problem')
        split_iter = split(X, len(X))
        parallel = ClusterParallel(fit_and_performance_wrapper, split_iter, {'X': X, 'y': y,
                                                                             'quantifier': self, 'local': local},
                                   local=True)  # TODO: Fix this
        clfs, tp_pa, fp_pa, tn_pa, fn_pa = zip(*parallel.retrieve())
        self.estimators_.extend(clfs)
        self.tp_pa_ = np.mean(tp_pa)
        self.fp_pa_ = np.mean(fp_pa)
        self.tn_pa_ = np.mean(tn_pa)
        self.fn_pa_ = np.mean(fn_pa)
        return self"""

    def _performance(self, X, y, clf, local, cv=3):
        predictions = clf.predict_proba(X)
        tp_pa_ = np.sum(predictions[y == clf.classes_[0], 0]) / np.sum(y == clf.classes_[0])
        fp_pa_ = np.sum(predictions[y == clf.classes_[1], 0]) / np.sum(y == clf.classes_[1])
        tn_pa_ = np.sum(predictions[y == clf.classes_[1], 1]) / np.sum(y == clf.classes_[1])
        fn_pa_ = np.sum(predictions[y == clf.classes_[0], 1]) / np.sum(y == clf.classes_[0])
        return tp_pa_, fp_pa_, tn_pa_, fn_pa_

    def fit_and_performance(self, perf, train, X, y, local):
        clf = self._fit(X[train[0]], y[train[0]])
        tp_pa, fp_pa, tn_pa, fn_pa = self._performance(np.concatenate(X[perf]), np.concatenate(y[perf]), clf, local)
        return clf, tp_pa, fp_pa, tn_pa, fn_pa

    def _adjust(self, prob):
        pos = (prob[0] - self.fp_pa_) / float(self.tp_pa_ - self.fp_pa_)
        neg = (prob[1] - self.fn_pa_) / float(self.tn_pa_ - self.fn_pa_)
        return pos, neg


def predict_proba_wrapper_per_clf(clf, X):
    return clf.predict_proba(X)
