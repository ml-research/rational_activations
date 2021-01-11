"""
this file tests that cpu calculations produce correct results
"""

from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid

# initialization of input data
from rational.keras.tests import _test_template

# test cpu execution
cuda = False


def test_a_on_cpu_lrelu():
    _test_template(version='A', approx_func=leaky_relu, cuda=cuda)


def test_b_on_cpu_lrelu():
    _test_template(version='B', approx_func=leaky_relu, cuda=cuda)


def test_c_on_cpu_lrelu():
    _test_template(version='C', approx_func=leaky_relu, cuda=cuda)


def test_d_on_cpu_lrelu():
    _test_template(version='D', approx_func=leaky_relu, cuda=cuda)


def test_a__on_cpu_tanh():
    _test_template(version='A', approx_func=tanh, cuda=cuda)


def test_b__on_cpu_tanh():
    _test_template(version='B', approx_func=tanh, cuda=cuda)


def test_c__on_cpu_tanh():
    _test_template(version='C', approx_func=tanh, cuda=cuda)


def test_d__on_cpu_tanh():
    _test_template(version='D', approx_func=tanh, cuda=cuda)


def test_a__on_cpu_sigmoid():
    _test_template(version='A', approx_func=sigmoid, cuda=cuda)


def test_b__on_cpu_sigmoid():
    _test_template(version='B', approx_func=sigmoid, cuda=cuda)


def test_c__on_cpu_sigmoid():
    _test_template(version='C', approx_func=sigmoid, cuda=cuda)


def test_d__on_cpu_sigmoid():
    _test_template(version='D', approx_func=sigmoid, cuda=cuda)
