import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_false
from nose.tools import assert_raises
from nose.tools import assert_true

from quantification.classify_and_count.ensemble import EnsembleBinaryCC, EnsembleMulticlassCC
from tests.base import ModelTestCase


class TestEnsembleBinaryCC(ModelTestCase):
    def test_X_y_different_length_raise_an_error(self):
        X = self.binary_X
        y = self.binary_y[2:]
        cc = EnsembleBinaryCC()
        assert_raises(ValueError, cc.fit, X, y)

    def test_performance_not_empty_after_fit(self):
        X = self.binary_X
        y = self.binary_y
        cc = EnsembleBinaryCC()
        cc.fit(X, y)

        assert_true(np.all([isinstance(qnf.confusion_matrix_, list) for qnf in cc.qnfs_]))

    def test_predict_returns_feasible_probabilities(self):
        cc = EnsembleBinaryCC()
        X = self.binary_X
        y = self.binary_y
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
    def test_performance_not_null_after_fit(self):
        cc = EnsembleMulticlassCC(b=100)
        X = self.mc_X
        y = self.mc_y
        cc.fit(X, y)

        for qnf in cc.qnfs_:
            for cm, tp, fp in zip(qnf.confusion_matrix_, qnf.tp_pa_, qnf.fp_pa_):
                assert_false(np.any(np.isnan(cm)))
                assert_false(np.isnan(tp))
                assert_false(np.isnan(fp))

    def test_predict_returns_feasible_probabilities(self):
        cc = EnsembleMulticlassCC(b=100)
        X = self.mc_X
        y = self.mc_y
        cc.fit(X, y)

        probabilities = cc.predict(np.concatenate(X))
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(np.concatenate(X), method='ac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(np.concatenate(X), method='pcc')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(np.concatenate(X), method='pac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)
