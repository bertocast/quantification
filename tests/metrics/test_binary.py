from nose.tools import assert_almost_equals

from quantification.metrics.base import absolute_error
from quantification.metrics.base import bias, square_error


class TestBinaryMetrics:
    def test_bias(self):
        assert_almost_equals(bias(0.7, 0.7), 0)
        assert_almost_equals(bias(0., 0.), 0)
        assert_almost_equals(bias(1., 1.), 0)
        assert_almost_equals(bias(0.7, 0.), -0.7)
        assert_almost_equals(bias(0.7, 1.), 0.3)
        assert_almost_equals(bias(0.7, 0.4), -0.3)

    def test_absolute_error(self):
        assert_almost_equals(absolute_error(0.7, 0.7), 0)
        assert_almost_equals(absolute_error(0., 0.), 0)
        assert_almost_equals(absolute_error(1., 1.), 0)
        assert_almost_equals(absolute_error(0.7, 0.), 0.7)
        assert_almost_equals(absolute_error(0.7, 1.), 0.3)
        assert_almost_equals(absolute_error(0.7, 0.4), 0.3)

    def test_square_error(self):
        assert_almost_equals(square_error(0.7, 0.7), 0)
        assert_almost_equals(square_error(0., 0.), 0)
        assert_almost_equals(square_error(1., 1.), 0)
        assert_almost_equals(square_error(0.7, 0.), 0.49)
        assert_almost_equals(square_error(0.7, 1.), 0.09)
        assert_almost_equals(square_error(0.7, 0.4), 0.09)


