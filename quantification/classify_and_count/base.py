from abc import ABCMeta, abstractmethod

import numpy as np
import six
from sklearn.linear_model import LogisticRegression

from quantification import BasicModel
from quantification.metrics import distributed, model_score


class BaseClassifyAndCountModel(six.with_metaclass(ABCMeta, BasicModel)):
    """Base class for C&C Models"""

    def __init__(self, estimator_class, estimator_params):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params

    @abstractmethod
    def fit(self, X, y):
        """Fit a sample or a set of samples and combine them"""

    @abstractmethod
    def predict(self, X, method):
        """Predict the prevalence"""

    def _validate_estimator(self, default):
        """Check the estimator."""
        if self.estimator_class is not None:
            estimator = self.estimator_class
        else:
            estimator = default

        if estimator is None:
            raise ValueError('estimator cannot be None')

        return estimator

    def _make_estimator(self):
        estimator = self._validate_estimator(default=LogisticRegression())

        estimator.set_params(**self.estimator_params)

        return estimator


class BaseBinaryClassifyAndCount(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=tuple()):
        super(BaseBinaryClassifyAndCount, self).__init__(estimator_class, estimator_params)
        self.estimator_ = self._make_estimator()
        self.confusion_matrix_ = None
        self.tp_pa_ = np.nan
        self.fp_pa_ = np.nan
        self.tn_pa_ = np.nan
        self.fn_pa_ = np.nan

    def fit(self, X, y, local=True):
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This solver is meant for binary samples "
                             "thus number of classes must be 2, but the "
                             "data contains %s", n_classes)

        if not local:
            self._persist_data(X, y)

        self.estimator_.fit(X, y)
        self.compute_performance_(X, y, local=local)
        return self

    def predict(self, X, method='cc'):
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def compute_performance_(self, X, y, local):
        if local:
            self.confusion_matrix_ = np.mean(
                model_score.cv_confusion_matrix(self.estimator_, X, y, 50), axis=0)
        else:
            self.confusion_matrix_ = np.mean(
                distributed.cv_confusion_matrix(self.estimator_, X, self.X_y_path_, folds=50), axis=0)
        try:
            predictions = self.estimator_.predict_proba(X)
        except AttributeError:
            return
        self.tp_pa_ = np.sum(predictions[y == self.estimator_.classes_[0], 0]) / \
                      np.sum(y == self.estimator_.classes_[0])
        self.fp_pa_ = np.sum(predictions[y == self.estimator_.classes_[1], 0]) / \
                      np.sum(y == self.estimator_.classes_[1])
        self.tn_pa_ = np.sum(predictions[y == self.estimator_.classes_[1], 1]) / \
                      np.sum(y == self.estimator_.classes_[1])
        self.fn_pa_ = np.sum(predictions[y == self.estimator_.classes_[0], 1]) / \
                      np.sum(y == self.estimator_.classes_[0])

    def _predict_cc(self, X):
        predictions = self.estimator_.predict(X)
        freq = np.bincount(predictions, minlength=2)
        relative_freq = freq / float(np.sum(freq))
        return relative_freq

    def _predict_ac(self, X):
        probabilities = self._predict_cc(X)
        tpr = self.confusion_matrix_[1, 1] / float(self.confusion_matrix_[1, 1] + self.confusion_matrix_[0, 1])
        fpr = self.confusion_matrix_[1, 0] / float(self.confusion_matrix_[1, 0] + self.confusion_matrix_[0, 0])
        if np.isnan(tpr):
            tpr = 0
        if np.isnan(fpr):
            fpr = 0
        adjusted = np.clip((probabilities[1] - fpr) / float(tpr - fpr), 0, 1)
        return np.array([1 - adjusted, adjusted])

    def _predict_pcc(self, X):
        try:
            predictions = self.estimator_.predict_proba(X)
        except AttributeError:
            raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                             "with hard (crisp) classifiers like %s", self.estimator_.__class__.__name__)

        p = np.mean(predictions, axis=0)
        return np.array(p)

    def _predict_pac(self, X):
        predictions = self._predict_pcc(X)
        neg = np.clip((predictions[0] - self.fp_pa_) / float(self.tp_pa_ - self.fp_pa_), 0, 1)
        pos = np.clip((predictions[1] - self.fn_pa_) / float(self.tn_pa_ - self.fn_pa_), 0, 1)
        return np.array([neg, pos])


class BaseMulticlassClassifyAndCount(BaseClassifyAndCountModel):
    def __init__(self, estimator_class=None, estimator_params=tuple()):
        super(BaseMulticlassClassifyAndCount, self).__init__(estimator_class, estimator_params)
        self.classes_ = None
        self.estimators_ = None
        self.confusion_matrix_ = None
        self.fpr_ = None
        self.tpr_ = None
        self.tp_pa_ = None
        self.fp_pa_ = None

    def fit(self, X, y, cv=50, verbose=False, local=True):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.estimators_ = dict.fromkeys(self.classes_)
        self.confusion_matrix_ = dict.fromkeys(self.classes_)
        self.tp_pa_ = dict.fromkeys(self.classes_)
        self.fp_pa_ = dict.fromkeys(self.classes_)

        if not local:
            self._persist_data(X, y)

        for pos_class in self.classes_:
            if verbose:
                print "Fiting classifier for class {}/{}".format(pos_class + 1, n_classes)
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.int)
            y_bin[~mask] = 0
            clf = self._make_estimator()
            clf = clf.fit(X, y_bin)
            self.estimators_[pos_class] = clf
            if verbose:
                print "Computing performance for classifier of class {}/{}".format(pos_class + 1, n_classes)
            self.compute_performance_(X, y_bin, pos_class, folds=cv, local=local, verbose=verbose)

        return self

    def compute_performance_(self, X, y, pos_class, folds, local, verbose):
        if local:
            self.confusion_matrix_[pos_class] = np.mean(
                model_score.cv_confusion_matrix(self.estimators_[pos_class], X, y, folds=folds), axis=0)
        else:

            self.confusion_matrix_[pos_class] = np.mean(
                distributed.cv_confusion_matrix(self.estimators_[pos_class], X, pos_class, self.X_y_path_, folds=folds,
                                                verbose=verbose),
                axis=0)

        try:
            predictions = self.estimators_[pos_class].predict_proba(X)
        except AttributeError:
            return

        self.tp_pa_[pos_class] = np.sum(predictions[y == self.estimators_[pos_class].classes_[1], 1]) / \
                                 np.sum(y == self.estimators_[pos_class].classes_[1])
        self.fp_pa_[pos_class] = np.sum(predictions[y == self.estimators_[pos_class].classes_[0], 1]) / \
                                 np.sum(y == self.estimators_[pos_class].classes_[0])

    def predict(self, X, method='cc'):
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def _predict_cc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            predictions = clf.predict(X)
            freq = np.bincount(predictions, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            probabilities[n] = relative_freq[1]
        return probabilities / np.sum(probabilities)

    def _predict_ac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            predictions = clf.predict(X)
            freq = np.bincount(predictions, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            tpr = self.confusion_matrix_[cls][1, 1] / float(self.confusion_matrix_[cls][1, 1]
                                                            + self.confusion_matrix_[cls][0, 1])
            fpr = self.confusion_matrix_[cls][1, 0] / float(self.confusion_matrix_[cls][1, 0]
                                                            + self.confusion_matrix_[cls][0, 0])
            if np.isnan(tpr):
                tpr = 0
            if np.isnan(fpr):
                fpr = 0

            adjusted = (relative_freq - fpr) / float(tpr - fpr)
            probabilities[n] = np.clip(adjusted[1], 0, 1)
        return probabilities / np.sum(probabilities)

    def _predict_pcc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            try:
                predictions = clf.predict_proba(X)
            except AttributeError:
                raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                 "with hard (crisp) classifiers like %s", clf.__class__.__name__)

            p = np.mean(predictions, axis=0)
            probabilities[n] = p[1]
        return probabilities / np.sum(probabilities)

    def _predict_pac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            try:
                predictions = clf.predict_proba(X)
            except AttributeError:
                raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                 "with hard (crisp) classifiers like %s", clf.__class__.__name__)

            p = np.mean(predictions, axis=0)
            probabilities[n] = np.clip((p[1] - self.fp_pa_[cls]) / float(self.tp_pa_[cls] - self.fp_pa_[cls]), 0, 1)

        return probabilities / np.sum(probabilities)


class BinaryClassifyAndCount(BaseBinaryClassifyAndCount):
    def predict(self, X, method='cc'):
        assert method == 'cc'
        return self._predict_cc(X)


class BinaryAdjustedCount(BaseBinaryClassifyAndCount):
    def predict(self, X, method='ac'):
        assert method == 'ac'
        return self._predict_ac(X)


class BinaryProbabilisticCC(BaseBinaryClassifyAndCount):
    def predict(self, X, method='pcc'):
        assert method == 'pcc'
        return self._predict_pcc(X)


class BinaryProbabilisticAC(BaseBinaryClassifyAndCount):
    def predict(self, X, method='pac'):
        assert method == 'pac'
        return self._predict_pac(X)


class MulticlassClassifyAndCount(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='cc'):
        assert method == 'cc'
        return self._predict_cc(X)


class MulticlassAdjustedCount(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='ac'):
        assert method == 'ac'
        return self._predict_ac(X)


class MulticlassProbabilisticCC(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='pcc'):
        assert method == 'pcc'
        return self._predict_pcc(X)


class MulticlassProbabilisticAC(BaseMulticlassClassifyAndCount):
    def predict(self, X, method='pac'):
        assert method == 'pac'
        return self._predict_pac(X)
