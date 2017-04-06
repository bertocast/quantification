import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_true

from quantification.distribution_matching.ensemble import BinaryEnsembleHDy, MulticlassEnsembleHDy
from tests.base import ModelTestCase


class TestBinaryEnsembleHDy(ModelTestCase):
    def test_predict_returns_feasible_probabilities(self):
        hdy = BinaryEnsembleHDy(b=100)
        X = self.binary_X
        y = self.binary_y
        hdy.fit(X, y)

        probabilities = hdy.predict(X[0])
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        freq = np.bincount(y[0], minlength=2)
        freq = (freq / float(np.sum(freq)))
        pass


class TestMulticlassEnsembleHDy(ModelTestCase):
    def test_predict_returns_feasible_probabilities(self):
        hdy = MulticlassEnsembleHDy(b=100)
        X = self.mc_X
        y = self.mc_y
        hdy.fit(X, y)

        probabilities = hdy.predict(X[0])
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)
