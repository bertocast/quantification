from abc import ABCMeta, abstractmethod

import numpy as np
import six
from sklearn.linear_model import LogisticRegression

from quantification.base import BasicModel
from quantification.utils.parallelism import ClusterParallel, predict_wrapper_per_sample, fit_wrapper, predict_wrapper_per_clf


class BaseClassifyAndCountModel(six.with_metaclass(ABCMeta, BasicModel)):
    """Base class for C&C Models"""

    @abstractmethod
    def _fit(self, X, y):
        """Fit a single model"""

    @abstractmethod
    def _predict(self, X):
        """Predict using the classifier model"""

    @abstractmethod
    def fit(self, X, y):
        """Fit a set of models and combine them"""

    def predict(self, X, local=False):
        if not isinstance(X, list):
            return self._predict(X)

        parallel = ClusterParallel(predict_wrapper_per_sample, X, {'clf': self}, local=local)
        return parallel.retrieve().tolist()


class ClassifyAndCount(BaseClassifyAndCountModel):
    """
    Ordinary classify and count method.

    Parameters
    ----------

    """

    def __init__(self, estimator_class=None, estimator_params=tuple()):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.estimators_ = []
        self._make_estimator()

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

    def _fit(self, X, y):
        clf = self._make_estimator()
        clf.fit(X, y)
        return clf

    def _predict(self, X):
        parallel = ClusterParallel(predict_wrapper_per_clf, self.estimators_, {'X': X})
        predictions = parallel.retrieve()
        freq = np.bincount(predictions)
        relative_freq = freq / float(np.sum(freq))
        return relative_freq

    def fit(self, X, y, local=False):
        if not isinstance(X, list):
            clf = self._fit(X, y)
            self.estimators_.append(clf)
        parallel = ClusterParallel(fit_wrapper, zip(X, y), {'clf': self}, local=local)
        clfs = parallel.retrieve()
        self.estimators_.extend(clfs)
        return self
