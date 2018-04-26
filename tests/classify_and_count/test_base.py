import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_false
from nose.tools import assert_not_equal
from nose.tools import assert_raises, assert_is_instance
from nose.tools import assert_true
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from quantification.cc.base import BaseCC
from tests.base import ModelTestCase



class TestClassifyAndCount(ModelTestCase):
    def test_fit_raise_error_if_parameters_are_not_arrays(self):
        X = 1
        y = None
        cc = BaseCC()
        assert_raises(ValueError, cc.fit, X, y)

    def test_fit_raise_error_if_sample_is_not_binary(self):
        X = None
        y = np.array([1, 2, 3])
        cc = BaseCC()
        assert_raises(ValueError, cc.fit, X, y)

    def test_default_classifier(self):
        cc = BaseCC()
        cc.fit(self.Xb, self.yb)
        assert_is_instance(cc.estimators_[1], LogisticRegression)

    def test_non_default_classifier(self):
        cc = BaseCC(estimator_class=DecisionTreeClassifier(), estimator_params=dict(max_depth=3))
        cc.fit(self.Xb, self.yb)
        assert_is_instance(cc.estimators_[1], DecisionTreeClassifier)
        assert_equal(cc.estimators_[1].max_depth, 3)

    def test_tpr_and_fpr_are_not_nan_after_fit(self):
        cc = BaseCC()
        cc.fit(self.Xb, self.yb)
        assert_false(np.isnan(cc.tp_pa_[1]))
        assert_false(np.isnan(cc.fp_pa_[1]))

    def test_predict_raise_error_if_method_is_invalid(self):
        cc = BaseCC()
        assert_raises(ValueError, cc.predict, None, method='bla')

    def test_predict_returns_feasible_probabilities(self):
        cc = BaseCC()
        cc.fit(self.Xb, self.yb)

        probabilities = cc.predict(self.Xb)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(self.Xb, method="ac")
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0, 1)

        probabilities = cc.predict(self.Xb, method="pcc")
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(self.Xb, method="pac")
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)


