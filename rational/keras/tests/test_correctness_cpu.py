import numpy as np
import tensorflow as tf
from rational.keras import Rational
from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid
from rational.keras.tests import activation

# initialization of input data
data = [-2., -1, 0., 1., 2.]


def _test_on_cpu(version, data, func):
    """
    compare the result of Rational activation function with expected result

    :param version: which version of Rational activation function to test
    :param data: test tensor for testing Rational function
    """

    # instantiate tensor for testing purpose
    test_tensor = tf.convert_to_tensor(np.array(data, np.float32), np.float32)

    # instantiate expected results of activation function
    expected_res = activation(func, data)

    # instantiate Rational activation function with specific version
    rational = Rational(approx_func=func.__name__, version=version,
                        cuda=False)(test_tensor).numpy()

    assert np.all(np.isclose(rational, expected_res, atol=5e-02))


def test_a_on_cpu_lrelu():
    _test_on_cpu(version='A', data=data, func=leaky_relu)


def test_b_on_cpu_lrelu():
    _test_on_cpu(version='B', data=data, func=leaky_relu)


def test_c_on_cpu_lrelu():
    _test_on_cpu(version='C', data=data, func=leaky_relu)


# def test_d_on_cpu_lrelu():
    #_test_on_cpu(version='D', data=data, func=leaky_relu)


def test_a__on_cpu_tanh():
    _test_on_cpu(version='A', data=data, func=tanh)


def test_b__on_cpu_tanh():
    _test_on_cpu(version='B', data=data, func=tanh)


def test_c__on_cpu_tanh():
    _test_on_cpu(version='C', data=data, func=tanh)


# def test_d__on_cpu_tanh():
    #_test_on_cpu(version='D', data=data, func=tanh)


def test_a__on_cpu_sigmoid():
    _test_on_cpu(version='A', data=data, func=sigmoid)


def test_b__on_cpu_sigmoid():
    _test_on_cpu(version='B', data=data, func=sigmoid)


def test_c__on_cpu_sigmoid():
    _test_on_cpu(version='C', data=data, func=sigmoid)


# def test_d__on_cpu_sigmoid():
    #_test_on_cpu(version='D', data=data, func=sigmoid)
