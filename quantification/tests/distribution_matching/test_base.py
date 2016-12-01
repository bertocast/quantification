from nose.tools import assert_almost_equal
from nose.tools import assert_false
from nose.tools import assert_true

from quantification.distribution_matching.base import BinaryHDy, MulticlassHDy
from quantification.tests.base import ModelTestCase

import numpy as np


class TestBinnaryHdy(ModelTestCase):
    def test_predict_returns_feasible_probabilities(self):
        hdy = BinaryHDy(b=100)
        X = np.concatenate(self.binary_data.data)
        y = np.concatenate(self.binary_data.target)
        hdy.fit(X, y)

        probabilities = hdy.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)


class TestMulticlassHdy(ModelTestCase):
    def test_train_dist_has_not_nan_after_train(self):
        hdy = MulticlassHDy(b=100)
        X = np.concatenate(self.multiclass_data.data)
        y = np.concatenate(self.multiclass_data.target)
        hdy.fit(X, y)
        for val in hdy.train_dist_.values():
            assert_false(np.any(np.isnan(val)))

    def test_predict_returns_feasible_probabilities(self):
        hdy = MulticlassHDy(b=100)
        X = np.concatenate(self.multiclass_data.data)
        y = np.concatenate(self.multiclass_data.target)
        hdy.fit(X, y)

        probabilities = hdy.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)
