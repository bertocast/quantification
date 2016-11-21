from nose.tools import assert_almost_equal
from nose.tools import assert_false
from nose.tools import assert_raises
from nose.tools import assert_true

from quantification.classify_and_count.ensemble import EnsembleBinaryCC, EnsembleMulticlassCC
from quantification.tests.base import ModelTestCase

import numpy as np


class TestEnsembleBinaryCC(ModelTestCase):

    def test_X_y_different_length_raise_an_error(self):
        X = self.binary_data.data
        y = self.binary_data.target[:-2]
        cc = EnsembleBinaryCC()
        assert_raises(ValueError, cc.fit, X, y)

    def test_performance_not_empty_after_fit(self):
        X = self.binary_data.data
        y = self.binary_data.target
        cc = EnsembleBinaryCC()
        cc.fit(X, y)

        assert_true(np.all([qnf.confusion_matrix_ for qnf in cc.qnfs_]))

    def test_predict_returns_feasible_probabilities(self):
        cc = EnsembleBinaryCC()
        X = self.binary_data.data
        y = self.binary_data.target
        cc.fit(X, y)

        probabilities = cc.predict(X[0])
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X[0], method='ac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X[0], method='pcc')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X[0], method='pac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)



class TestEnsembleMulticlassCC(ModelTestCase):
    def test_predict_returns_feasible_probabilities(self):
        cc = EnsembleMulticlassCC()
        X = self.multiclass_data.data
        y = self.multiclass_data.target
        cc.fit(X, y)

        probabilities = cc.predict(X[0])
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X[0], method='ac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X[0], method='pcc')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X[0], method='pac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)