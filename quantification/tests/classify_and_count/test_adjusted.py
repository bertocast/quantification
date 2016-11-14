import numpy as np
from nose.tools import assert_almost_equal

from quantification.classify_and_count import BinaryAdjustedCount, MulticlassAdjustedCount
from quantification.tests.base import ModelTestCase


class TestAdjustedCount(ModelTestCase):
    def test_fit_single_sample_of_binary_data(self):
        ac = BinaryAdjustedCount()
        X = self.binary_data.data[0]
        y = self.binary_data.target[0]
        ac.fit(X, y, local=True)
        predictions = ac.predict(X, local=True)
        assert_almost_equal(np.sum(predictions), 1, places=1)

    def test_fit_ensemble_of_binary_data(self):
        ac = BinaryAdjustedCount()
        X = self.binary_data.data
        y = self.binary_data.target
        ac.fit(X, y, local=True)
        predictions = ac.predict(X, local=True)
        assert_almost_equal(np.sum(predictions), len(y), places=1)

    def test_fit_single_sample_of_multiclass_data(self):
        ac = MulticlassAdjustedCount()
        X = self.multiclass_data.data[0]
        y = self.multiclass_data.target[0]
        ac.fit(X, y, local=True)
        predictions = ac.predict(X, local=True)
        assert_almost_equal(np.sum(predictions), 1, places=1)

    def test_fit_ensemble_of_multiclass_data(self):
        ac = MulticlassAdjustedCount()
        X = self.multiclass_data.data
        y = self.multiclass_data.target
        ac.fit(X, y, local=True)
        predictions = ac.predict(X, local=True)
        assert_almost_equal(np.sum(predictions), len(y), places=1)
