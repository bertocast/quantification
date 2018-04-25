from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
import six
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
from quantification.base import BasicModel
from quantification.metrics import distributed, model_score


class BaseClassifyAndCountModel(six.with_metaclass(ABCMeta, BasicModel)):
    def __init__(self, estimator_class, estimator_params, estimator_grid, grid_params, b):
        if estimator_params is None:
            estimator_params = dict()
        if estimator_grid is None:
            estimator_grid = dict()
        if grid_params is None:
            grid_params = dict()
        self.b = b
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.estimator_grid = estimator_grid
        self.grid_params = grid_params

    @abstractmethod
    def fit(self, X, y):
        """Fit a sample or a set of samples and combine them"""

    @abstractmethod
    def predict(self, X):
        """Predict the prevalence"""

    def _validate_estimator(self, default, default_params, default_grid):
        """Check the estimator."""
        if self.estimator_class is not None:
            clf = self.estimator_class
        else:
            clf = default
        if self.estimator_params is not None:
            clf.set_params(**self.estimator_params)
            if not self.estimator_grid:
                estimator = clf
            else:
                estimator = GridSearchCV(estimator=self.estimator_class, param_grid=self.estimator_grid,
                                         **self.grid_params)
        else:
            clf.set_params(**default_params)
            if not self.estimator_grid:
                estimator = clf
            else:
                estimator = GridSearchCV(estimator=clf, param_grid=default_grid, verbose=11)

        if estimator is None:
            raise ValueError('estimator cannot be None')

        return estimator

    def _make_estimator(self):
        """Build the estimator"""
        estimator = self._validate_estimator(default=LogisticRegression(), default_grid={'C': [0.1, 1, 10]},
                                             default_params=dict())
        return estimator


class BaseCC(BaseClassifyAndCountModel):
    """
        Multiclass Classify And Count method.

        It is meant to be trained once and be able to predict using the following methods:
            - Classify & Count
            - Adjusted Count
            - Probabilistic Classify & Count
            - Probabilistic Adjusted Count
            - HDy

        The idea is not to trained the classifiers more than once, due to the computational cost. In the training phase
        every single performance metric that is needed in predictions are computed, that is, FPR, TPR and so on.

        If you are only going to use one of the methods above, you can use the wrapper classes in this package.

        Parameters
        ----------
        b : integer, optional
            Number of bins to compute the distributions in the HDy method. If you are not going to use that method in the
            prediction phase, leave it as None. The training phase will be probably faster.

        estimator_class : object, optional
            An instance of a classifier class. It has to have fit and predict methods. It is highly advised to use one of
            the implementations in sklearn library. If it is leave as None, Logistic Regression will be used.

        estimator_params : dictionary, optional
            Additional params to initialize the classifier.

        estimator_grid : dictionary, optional
            During training phase, grid search is performed. This parameter should provided the parameters of the classifier
            that will be tested (e.g. estimator_grid={C: [0.1, 1, 10]} for Logistic Regression).

        strategy : string, optional
            Strategy to follow when aggregating.

        multiclass : string, optional
            One versus all or one vs one

        Attributes
        ----------
        estimator_class : object
            The underlying classifier+

        confusion_matrix_ : dictionary
            The confusion matrix estimated by cross-validation of the underlying classifier for each class

        tpr_ : dictionary
            True Positive Rate of the underlying classifier from the confusion matrix for each class

        fpr_ : dictionary
            False Positive Rate of the underlying classifier from the confusion matrix for each class

        tp_pa_ : dictionary
            True Positive Probability Average of the underlying classifier if it is probabilistic for each class

        fp_pa_ : dictionary
            False Positive Probability Average of the underlying classifier if it is probabilistic for each class

        train_dist_ : dictionary
            Distribution of the positive and negative samples for each bin in the training data for each class

        """

    def __init__(self, estimator_class=None, estimator_params=None, estimator_grid=None, grid_params=None, b=None,
                 strategy='macro', multiclass='ova'):
        super(BaseCC, self).__init__(estimator_class, estimator_params, estimator_grid,
                                     grid_params, b)
        self.strategy = strategy
        self.multiclass = multiclass

    def fit(self, X, y, cv=50, verbose=False, local=True):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)

        self.fpr_ = dict.fromkeys(self.classes_)
        self.tpr_ = dict.fromkeys(self.classes_)

        if not local:
            self._persist_data(X, y)

        self.estimators_ = dict.fromkeys(self.classes_)
        self.confusion_matrix_ = dict.fromkeys(self.classes_)

        self.tp_pa_ = dict.fromkeys(self.classes_)
        self.fp_pa_ = dict.fromkeys(self.classes_)

        for pos_class in self.classes_:
            if verbose:
                print("Class {}/{}".format(pos_class + 1, n_classes))
                print("\tFitting  classifier...")
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.int)
            y_bin[~mask] = 0
            clf = self._make_estimator()
            clf = clf.fit(X, y_bin)
            if isinstance(clf, GridSearchCV):
                clf = clf.best_estimator_
            self.estimators_[pos_class] = deepcopy(clf)
            if verbose:
                print("\tComputing performance...")
            self._compute_performance(X, y_bin, pos_class, folds=cv, local=local, verbose=verbose)
        if self.b:
            if verbose:
                print("\tComputing distribution...")
        self._compute_distribution(X, y)

        return self

    def _compute_performance(self, X, y, pos_class, folds, local, verbose):

        if local:
            cm = model_score.cv_confusion_matrix(self.estimators_[pos_class], X, y, folds, verbose)
        else:
            cm = distributed.cv_confusion_matrix(self.estimators_[pos_class], X, y, self.X_y_path_, pos_class=pos_class,
                                                 folds=folds,
                                                 verbose=verbose)

        if self.strategy == 'micro':
            self.confusion_matrix_[pos_class] = np.mean(cm, axis=0)
            self.tpr_[pos_class] = self.confusion_matrix_[pos_class][1, 1] / float(
                self.confusion_matrix_[pos_class][1, 1] + self.confusion_matrix_[pos_class][1, 0])
            self.fpr_[pos_class] = self.confusion_matrix_[pos_class][0, 1] / float(
                self.confusion_matrix_[pos_class][0, 1] + self.confusion_matrix_[pos_class][0, 0])
        elif self.strategy == 'macro':
            self.confusion_matrix_[pos_class] = cm
            self.tpr_[pos_class] = np.mean([cm_[1, 1] / float(cm_[1, 1] + cm_[1, 0]) for cm_ in cm])
            self.fpr_[pos_class] = np.mean([cm_[0, 1] / float(cm_[0, 1] + cm_[0, 0]) for cm_ in cm])

        try:
            predictions = self.estimators_[pos_class].predict_proba(X)
        except AttributeError:
            return

        self.tp_pa_[pos_class] = np.sum(predictions[y == self.estimators_[pos_class].classes_[1], 1]) / \
                                 np.sum(y == self.estimators_[pos_class].classes_[1])
        self.fp_pa_[pos_class] = np.sum(predictions[y == self.estimators_[pos_class].classes_[0], 1]) / \
                                 np.sum(y == self.estimators_[pos_class].classes_[0])

    def _compute_distribution(self, X, y):

        if not self.b:
            return

        n_classes = len(self.classes_)
        n_clfs = n_classes  # OvA
        self.train_dist_ = np.zeros((n_classes, self.b, n_clfs))

        if len(self.classes_) == 1:
            # If it is a binary problem, add the representation of the negative samples
            pos_preds = self.estimators_[1].predict_proba(X[y == 1])[:, 1]
            neg_preds = self.estimators_[1].predict_proba(X[y == 0])[:, 1]
            pos_pdf, _ = np.histogram(pos_preds, bins=self.b)
            neg_pdf, _ = np.histogram(neg_preds, bins=self.b)
            self.train_dist_= np.vstack([(pos_pdf / float(sum(y == 1)))[None, :, None], (pos_pdf / float(sum(y == 0)))[None, :, None]])
        else:
            for n_cls, cls in enumerate(self.classes_):
                mask = (y == cls)
                y_bin = np.ones(y.shape, dtype=np.int)
                y_bin[~mask] = 0

                for n_clf, (clf_cls, clf) in enumerate(self.estimators_.items()):
                    preds = clf.predict_proba(X[y == cls])[:, 1]
                    pdf, _ = np.histogram(preds, bins=self.b)
                    self.train_dist_[n_cls, :, n_clf] = pdf / float(sum(y_bin))







    def predict(self, X, method='cc'):
        if method == 'cc':
            return self._predict_cc(X)
        elif method == 'ac':
            return self._predict_ac(X)
        elif method == 'pcc':
            return self._predict_pcc(X)
        elif method == 'pac':
            return self._predict_pac(X)
        elif method == "hdy":
            return self._predict_hdy(X)
        else:
            raise ValueError("Invalid method %s. Choices are `cc`, `ac`, `pcc`, `pac`.", method)

    def _predict_cc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.zeros(n_classes)

        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            predictions = clf.predict(X)
            freq = np.bincount(predictions, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            probabilities[n] = relative_freq[1]

        if len(probabilities) < 2:
            probabilities = np.array([1 - probabilities[0], probabilities[0]])

        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_ac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.zeros(n_classes)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            predictions = clf.predict(X)
            freq = np.bincount(predictions, minlength=2)
            relative_freq = freq / float(np.sum(freq))
            adjusted = (relative_freq - self.fpr_[cls]) / float(self.tpr_[cls] - self.fpr_[cls])
            adjusted = np.nan_to_num(adjusted)
            probabilities[n] = np.clip(adjusted[1], 0, 1)

        if len(probabilities) < 2:
            probabilities = np.array([1 - probabilities[0], probabilities[0]])

        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_pcc(self, X):
        n_classes = len(self.classes_)
        probabilities = np.zeros(n_classes)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            try:
                predictions = clf.predict_proba(X)
            except AttributeError:
                raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                 "with hard (crisp) classifiers like %s", clf.__class__.__name__)

            p = np.mean(predictions, axis=0)
            probabilities[n] = p[1]

        if len(probabilities) < 2:
            probabilities = np.array([1 - probabilities[0], probabilities[0]])

        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_pac(self, X):
        n_classes = len(self.classes_)
        probabilities = np.zeros(n_classes)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            try:
                predictions = clf.predict_proba(X)
            except AttributeError:
                raise ValueError("Probabilistic methods like PCC or PAC cannot be used "
                                 "with hard (crisp) classifiers like %s", clf.__class__.__name__)

            p = np.mean(predictions, axis=0)
            probabilities[n] = np.clip((p[1] - self.fp_pa_[cls]) / float(self.tp_pa_[cls] - self.fp_pa_[cls]), 0, 1)

        if len(probabilities) < 2:
            probabilities = np.array([1 - probabilities[0], probabilities[0]])

        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)

    def _predict_hdy(self, X):
        if not self.b:
            raise ValueError("If HDy predictions are in order, the quantifier must be trained with the parameter `b`")
        n_classes = len(self.classes_)

        test_dist = np.zeros((self.b, len(self.estimators_)))
        for n_clf, (clf_cls, clf) in enumerate(self.estimators_.items()):
            pdf, _ = np.histogram(clf.predict_proba(X)[:, 1], self.b)
            test_dist[:, n_clf] = pdf / float(X.shape[0])

        if n_classes == 1:
            p_combs = np.linspace(0, 1, 101)[:, None]

        else:
            num_combs = 101
            p = np.linspace(0, 1, num_combs)
            p_combs = np.array(np.meshgrid(*([p] * (n_classes - 1)))).T.reshape(-1, (n_classes - 1))
            p_combs = p_combs[p_combs.sum(1) <= 1.]

        p_combs = np.hstack([p_combs, 1 - p_combs.sum(axis=1, keepdims=True)])

        di = (p_combs[:, :, None, None] * self.train_dist_[None, :]).sum(axis=1)
        di = np.sqrt(di)
        ti = np.sqrt(test_dist)
        diff = np.power(ti - di, 2)
        hds = np.sqrt(diff.sum(1).sum(1))
        p = p_combs[hds.argmin()]

        return p




class CC(BaseCC):
    """
        Multiclass Classify And Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use CC.
        """

    def predict(self, X, method='cc'):
        assert method == 'cc'
        return self._predict_cc(X)


class AC(BaseCC):
    """
        Multiclass Adjusted Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use AC.

        """

    def predict(self, X, method='ac'):
        assert method == 'ac'
        return self._predict_ac(X)


class PCC(BaseCC):
    """
        Multiclass Probabilistic Classify And Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use PCC."""

    def predict(self, X, method='pcc'):
        assert method == 'pcc'
        return self._predict_pcc(X)


class PAC(BaseCC):
    """
        Multiclass Probabilistic Adjusted Count method.

        Just a wrapper to perform adjusted count without the need of every other single methods.
        The main difference with the general class is the `predict` method that enforces to use PAC.
        """

    def predict(self, X, method='pac'):
        assert method == 'pac'
        return self._predict_pac(X)
