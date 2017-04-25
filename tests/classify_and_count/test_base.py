import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_false
from nose.tools import assert_not_equal
from nose.tools import assert_raises, assert_is_instance
from nose.tools import assert_true
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from quantification.classify_and_count.base import BaseBinaryClassifyAndCount, BaseMulticlassClassifyAndCount
from tests.base import ModelTestCase


class TestBinaryClassifyAndCount(ModelTestCase):
    def test_fit_raise_error_if_parameters_are_not_arrays(self):
        X = 1
        y = None
        cc = BaseBinaryClassifyAndCount()
        assert_raises(ValueError, cc.fit, X, y)

    def test_fit_raise_error_if_sample_is_not_binary(self):
        X = None
        y = np.array([1, 2, 3])
        cc = BaseBinaryClassifyAndCount()
        assert_raises(ValueError, cc.fit, X, y)

    def test_default_classifier(self):
        cc = BaseBinaryClassifyAndCount()
        assert_is_instance(cc.estimator_, LogisticRegression)

    def test_non_default_classifier(self):
        cc = BaseBinaryClassifyAndCount(estimator_class=DecisionTreeRegressor(), estimator_params=dict(max_depth=3),
                                        estimator_grid=dict(min_samples_split=[2, 3, 4]))
        assert_is_instance(cc.estimator_.estimator, DecisionTreeRegressor)
        assert_equal(cc.estimator_.param_grid, dict(min_samples_split=[2, 3, 4]))
        assert_equal(cc.estimator_.estimator.max_depth, 3)

    def test_tpr_and_fpr_are_not_nan_after_fit(self):
        cc = BaseBinaryClassifyAndCount()
        X = np.concatenate(self.binary_X)
        y = np.concatenate(self.binary_y)
        cc.fit(X, y)
        assert_false(np.isnan(cc.tp_pa_))
        assert_false(np.isnan(cc.fp_pa_))
        assert_false(np.isnan(cc.tn_pa_))
        assert_false(np.isnan(cc.fn_pa_))

    def test_predict_raise_error_if_method_is_invalid(self):
        cc = BaseBinaryClassifyAndCount()
        assert_raises(ValueError, cc.predict, None, method='bla')

    def test_predict_returns_feasible_probabilities(self):
        cc = BaseBinaryClassifyAndCount()
        X = np.concatenate(self.binary_X)
        y = np.concatenate(self.binary_y)
        cc.fit(X, y)

        probabilities = cc.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method="ac")
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0, 1)

        probabilities = cc.predict(X, method="pcc")
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method="pac")
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)


class TestMulticlassClassifyAndCount(ModelTestCase):
    def test_fit_raise_error_if_parameters_are_not_arrays(self):
        X = 1
        y = None
        cc = BaseMulticlassClassifyAndCount()
        assert_raises(AttributeError, cc.fit, X, y)

    def test_one_clf_for_each_class_after_fit(self):
        cc = BaseMulticlassClassifyAndCount()
        X = np.concatenate(self.mc_X)
        y = np.concatenate(self.mc_y)
        cc.fit(X, y)
        for label in cc.classes_:
            assert_true(cc.estimators_.get(label))

    def test_performance_not_nan_nor_equal_after_fit(self):
        cc = BaseMulticlassClassifyAndCount()
        X = np.concatenate(self.mc_X)
        y = np.concatenate(self.mc_y)
        cc.fit(X, y)

        assert_false(np.isnan(cc.tp_pa_.values()).any())
        assert_false(np.isnan(cc.fp_pa_.values()).any())

    def test_predict_returns_feasible_probabilities(self):
        cc = BaseMulticlassClassifyAndCount()
        X = np.concatenate(self.mc_X)
        y = np.concatenate(self.mc_y)
        cc.fit(X, y)

        probabilities = cc.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method='ac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method='pcc')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method='pac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

    def test_ovo_classifier(self):
        cc = BaseMulticlassClassifyAndCount(multiclass='ovo')
        X = np.concatenate(self.mc_X)
        y = np.concatenate(self.mc_y)
        cc.fit(X, y, cv=3)

        probabilities = cc.predict(X)
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method='ac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method='pcc')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

        probabilities = cc.predict(X, method='pac')
        assert_true(np.all(probabilities <= 1.))
        assert_true(np.all(probabilities >= 0.))
        assert_almost_equal(np.sum(probabilities), 1.0)

