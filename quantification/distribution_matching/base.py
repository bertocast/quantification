from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import six
from sklearn.linear_model import LogisticRegression

from quantification import BasicModel


class BaseDistributionMatchingModel(six.with_metaclass(ABCMeta, BasicModel)):
    """Basic Distribution Matching Model"""

    def __init__(self, estimator_class=None, estimator_params=tuple()):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params

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

        estimator.set_params(**dict((p, getattr(self, p))

                                    for p in self.estimator_params))
        return estimator

    @abstractmethod
    def fit(self, X, y, plot=False):
        """Fit a sample or a set of samples and combine them"""

    @abstractmethod
    def predict(self, X, plot=False):
        """Predict the prevalence"""


class BinaryHDy(BaseDistributionMatchingModel):
    def __init__(self, b, estimator_class=None, estimator_params=tuple()):
        super(BinaryHDy, self).__init__(estimator_class, estimator_params)
        self.estimator_ = self._make_estimator()
        self.b = b
        self.train_dist_ = None

    def fit(self, X, y, plot=False):
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This solver is meant for binary samples "
                             "thus number of classes must be 2, but the "
                             "data contains %s", n_classes)
        self.estimator_.fit(X, y)
        pos_class = self.estimator_.classes_[1]
        neg_class = self.estimator_.classes_[0]
        pos_preds = self.estimator_.predict_proba(X[y == pos_class,])[:, 1]
        neg_preds = self.estimator_.predict_proba(X[y == neg_class,])[:, +1]

        train_pos_pdf, _ = np.histogram(pos_preds, self.b)
        train_neg_pdf, _ = np.histogram(neg_preds, self.b)
        self.train_dist_ = np.full((self.b, 2), np.nan)
        for i in range(self.b):
            self.train_dist_[i] = [train_pos_pdf[i] / float(sum(y == pos_class)),
                                  train_neg_pdf[i] / float(sum(y == neg_class))]

        if plot:
            plt.subplot(121)
            plt.hist(neg_preds, self.b)
            plt.title('Negative PDF')

            plt.figure(1)
            plt.subplot(122)
            plt.hist(pos_preds, self.b)
            plt.title('Positive PDF')

            plt.show()

        return self

    def predict(self, X, plot=False):

        preds = self.estimator_.predict_proba(X)[:, 1]
        test_pdf, _ = np.histogram(preds, self.b)

        if plot:
            plt.figure(2)
            plt.hist(preds, self.b)
            plt.title('Test PDF')

            plt.show()

        probas = [p / 100.0 for p in range(0, 100)]
        hd = np.full(len(probas), np.nan)
        for p in range(len(probas)):
            diff = np.full(self.b, np.nan)
            for i in range(self.b):
                di = np.sqrt(self.train_dist_[i, 0] * probas[p] + self.train_dist_[i, 1] * (1 - probas[p]))
                ti = np.sqrt(test_pdf[i] / float(X.shape[0]))
                diff[i] = np.power(di - ti, 2)
            hd[p] = np.sqrt(np.sum(diff))

        p_min = np.argmin(hd)
        prevalence = probas[p_min]
        return np.array([1 - prevalence, prevalence])


class MulticlassHDy(BaseDistributionMatchingModel):
    def __init__(self, b, estimator_class=None, estimator_params=tuple()):
        super(MulticlassHDy, self).__init__(estimator_class, estimator_params)
        self.b = b
        self.train_dist_ = None

    def fit(self, X, y, verbose=False, plot=False):
        self.classes_ = np.unique(y).tolist()
        n_classes = len(self.classes_)
        self.estimators_ = dict.fromkeys(self.classes_)
        self.train_dist_ = dict.fromkeys(self.classes_)

        for cls in self.classes_:
            mask = (y == cls)
            y_bin = np.ones(y.shape, dtype=np.int)
            y_bin[~mask] = 0
            clf = self._make_estimator()
            if verbose:
                print "Fitting classifier for class {}/{}".format(cls + 1, n_classes)
            clf.fit(X, y_bin)
            self.estimators_[cls] = clf
            pos_class = clf.classes_[1]
            neg_class = clf.classes_[0]
            pos_preds = clf.predict_proba(X[y_bin == pos_class,])[:, 1]
            neg_preds = clf.predict_proba(X[y_bin == neg_class,])[:, +1]

            train_pos_pdf, _ = np.histogram(pos_preds, self.b)
            train_neg_pdf, _ = np.histogram(neg_preds, self.b)
            self.train_dist_[cls] = np.full((self.b, 2), np.nan)
            for i in range(self.b):
                self.train_dist_[cls][i] = [train_pos_pdf[i] / float(sum(y_bin == pos_class)),
                                      train_neg_pdf[i] / float(sum(y_bin == neg_class))]

        return self

    def predict(self, X, plot=False):
        n_classes = len(self.classes_)
        probabilities = np.full(n_classes, np.nan)
        for n, (cls, clf) in enumerate(self.estimators_.iteritems()):
            preds = clf.predict_proba(X)[:, 1]
            test_pdf, _ = np.histogram(preds, self.b)
            probas = [p / 100.0 for p in range(0, 100)]
            hd = np.full(len(probas), np.nan)
            for p in range(len(probas)):
                diff = np.full(self.b, np.nan)
                for i in range(self.b):
                    di = np.sqrt(self.train_dist_[cls][i, 0] * probas[p] + self.train_dist_[cls][i, 1] * (1 - probas[p]))
                    ti = np.sqrt(test_pdf[i] / float(X.shape[0]))
                    diff[i] = np.power(di - ti, 2)
                hd[p] = np.sqrt(np.sum(diff))

            p_min = np.argmin(hd)
            probabilities[n] = probas[p_min]

        return probabilities / np.sum(probabilities)
