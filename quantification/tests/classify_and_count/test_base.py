import numpy as np
from nose.tools import assert_almost_equal
from nose.tools import assert_false
from nose.tools import assert_not_equal
from nose.tools import assert_raises, assert_is_instance
from nose.tools import assert_true
from sklearn.linear_model import LogisticRegression
from quantification.classify_and_count.base import BinaryClassifyAndCount, MulticlassClassifyAndCount
from quantification.tests.base import ModelTestCase


class TestBinaryClassifyAndCount(ModelTestCase):
    def test_fit_raise_error_if_parameters_are_not_arrays(self):
        X = 1
        y = None
        cc = BinaryClassifyAndCount()
        assert_raises(ValueError, cc.fit, X, y)

    def test_fit_raise_error_if_sample_is_not_binary(self):
        X = None
        y = np.array([1, 2, 3])
        cc = BinaryClassifyAndCount()
        assert_raises(ValueError, cc.fit, X, y)

    def test_default_classifier(self):
        cc = BinaryClassifyAndCount()
        assert_is_instance(cc.estimator_, LogisticRegression)

    def test_tpr_and_fpr_are_not_nan_after_fit(self):
        cc = BinaryClassifyAndCount()
        X = np.concatenate(self.binary_data.data)
        y = np.concatenate(self.binary_data.target)
        cc.fit(X, y)
        assert_not_equal(cc.tp_pa_, np.nan)
        assert_not_equal(cc.fp_pa_, np.nan)
        assert_not_equal(cc.tn_pa_, np.nan)
        assert_not_equal(cc.fn_pa_, np.nan)

    def test_predict_raise_error_if_method_is_invalid(self):
        cc = BinaryClassifyAndCount()
        assert_raises(ValueError, cc.predict, None, method='bla')

    def test_predict_returns_feasible_probabilities(self):
        cc = BinaryClassifyAndCount()
        X = np.concatenate(self.binary_data.data)
        y = np.concatenate(self.binary_data.target)
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
        cc = MulticlassClassifyAndCount()
        assert_raises(AttributeError, cc.fit, X, y)

    def test_one_clf_for_each_class_after_fit(self):
        cc = MulticlassClassifyAndCount()
        X = np.concatenate(self.multiclass_data.data)
        y = np.concatenate(self.multiclass_data.target)
        cc.fit(X, y)
        for label in cc.classes_:
            assert_true(cc.estimators_.get(label))

    def test_performance_not_nan_nor_equal_after_fit(self):
        cc = MulticlassClassifyAndCount()
        X = np.concatenate(self.multiclass_data.data)
        y = np.concatenate(self.multiclass_data.target)
        cc.fit(X, y)

        assert_false(np.all(cc.tp_pa_ == np.nan))
        assert_false(np.all(cc.tp_pa_ == cc.tp_pa_[0]))
        assert_false(np.all(cc.fp_pa_ == np.nan))
        assert_false(np.all(cc.fp_pa_ == cc.fp_pa_[0]))

    def test_predict_returns_feasible_probabilities(self):
        cc = MulticlassClassifyAndCount()
        X = np.concatenate(self.multiclass_data.data)
        y = np.concatenate(self.multiclass_data.target)
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


