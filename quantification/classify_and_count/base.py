import logging

import dispy
import numpy as np
import six
from abc import ABCMeta, abstractmethod

from sklearn.linear_model import LogisticRegression

from quantification.base import BasicModel


class BaseClassifyAndCountModel(six.with_metaclass(ABCMeta, BasicModel)):
    """Base class for C&C Models"""

    @abstractmethod
    def _fit(self, X, y):
        """Fit model."""

    @abstractmethod
    def _predict(self, X):
        """Predict using the classifier model"""

    def fit(self, X, y):
        if not isinstance(X, list):
            return self._fit(X, y)
        raise NotImplementedError


    def predict(self, X):
        if not isinstance(X, list):
            return self._predict(X)

        def predict_wrapper(estimator, X):
            return estimator.predict(X)

        cluster = dispy.JobCluster(predict_wrapper, loglevel=logging.ERROR, pulse_interval=10,
                                   reentrant=True)
        jobs = []
        for X_ in X:
            job = cluster.submit(estimator=self, X=X_)
            jobs.append(job)
        cluster.wait()
        predictions = []
        for job in jobs:
            job()
            predictions.append(job.result)
        cluster.close()
        return predictions


class ClassifyAndCount(BaseClassifyAndCountModel):
    """
    Ordinary classify and count method.

    Parameters
    ----------

    """

    def __init__(self, estimator_class=None, estimator_params=tuple()):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params

    def _validate_estimator(self, default):
        """Check the estimator."""
        if self.estimator_class is not None:
            self.estimator = self.estimator_class
        else:
            self.estimator = default

        if self.estimator is None:
            raise ValueError('estimator cannot be None')

    def _make_estimator(self):
        self._validate_estimator(default=LogisticRegression())

        self.estimator.set_params(**dict((p, getattr(self, p))
                                         for p in self.estimator_params))

    def _fit(self, X, y):

        self._make_estimator()
        self.estimator.fit(X, y)
        self.labels_ = np.unique(y)
        return self

    def _predict(self, X):
        predictions = self.estimator.predict(X)
        freq = np.bincount(predictions)
        relative_freq = freq / float(np.sum(freq))
        return relative_freq


if __name__ == '__main__':
    cc = ClassifyAndCount()
    def func(X):
        pass

    print type(cc.predict)
