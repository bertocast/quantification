import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_false
from nose.tools import assert_true

from quantification.dm.base import HDy
from tests.base import ModelTestCase


class TestBinnaryHdy(ModelTestCase):
    def test_predict_returns_feasible_probabilities(self):
        hdy = HDy(b=8)
        hdy.fit(self.Xb, self.yb)

        probabilities = hdy.predict(self.Xb)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)
