import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_false
from nose.tools import assert_true

from quantification.distribution_matching.base import BinaryHDy, MulticlassHDy, BinaryEM
from tests.base import ModelTestCase


class TestBinnaryHdy(ModelTestCase):
    def test_predict_returns_feasible_probabilities(self):
        hdy = BinaryHDy(b=8)
        X = np.concatenate(self.binary_X)
        y = np.concatenate(self.binary_y)
        hdy.fit(X, y)

        probabilities = hdy.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)


class TestMulticlassHdy(ModelTestCase):
    def test_train_dist_has_not_nan_after_train(self):
        hdy = MulticlassHDy(b=100)
        X = np.concatenate(self.mc_X)
        y = np.concatenate(self.mc_y)
        hdy.fit(X, y)
        for val in hdy.train_dist_.values():
            assert_false(np.any(np.isnan(val)))

    def test_predict_returns_feasible_probabilities(self):
        hdy = MulticlassHDy(b=100)
        X = np.concatenate(self.mc_X)
        y = np.concatenate(self.mc_y)
        hdy.fit(X, y)

        probabilities = hdy.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)


class TestBinaryEM(ModelTestCase):

    def test_base(self):
        X = np.concatenate(self.binary_X)
        y = np.concatenate(self.binary_y)
        em = BinaryEM()

        em.fit(X, y)
        em.predict(X)

