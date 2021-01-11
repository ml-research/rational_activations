import numpy as np
import tensorflow as tf
from rational.keras import Rational
from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid

# initialization of input data
data = [-2., -1, 0., 1., 2.]


def activation(func, data):
    """
    apply activation function to data

    :param func: activaion function
    :param data: data to be applied
    """
    if func == leaky_relu:
        return np.array(func(data, alpha=0.01))
    return np.array(func(data))


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
    trainable = False if version != 'D' else True
    rational = Rational(approx_func=func.__name__, version=version,
                        cuda=False, trainable=trainable)(test_tensor).numpy()

    assert np.all(np.isclose(rational, expected_res, atol=5e-02))


def test_A_on_cpu_lrelu():
    _test_on_cpu(version='A', data=data, func=leaky_relu)


def test_B_on_cpu_lrelu():
    _test_on_cpu(version='B', data=data, func=leaky_relu)


def test_C_on_cpu_lrelu():
    _test_on_cpu(version='C', data=data, func=leaky_relu)


def test_D_on_cpu_lrelu():
    _test_on_cpu(version='D', data=data, func=leaky_relu)


def test_A__on_cpu_tanh():
    _test_on_cpu(version='A', data=data, func=tanh)


def test_B__on_cpu_tanh():
    _test_on_cpu(version='B', data=data, func=tanh)


def test_C__on_cpu_tanh():
    _test_on_cpu(version='C', data=data, func=tanh)


def test_D__on_cpu_tanh():
    _test_on_cpu(version='D', data=data, func=tanh)


def test_A__on_cpu_sigmoid():
    _test_on_cpu(version='A', data=data, func=sigmoid)


def test_B__on_cpu_sigmoid():
    _test_on_cpu(version='B', data=data, func=sigmoid)


def test_C__on_cpu_sigmoid():
    _test_on_cpu(version='C', data=data, func=sigmoid)


def test_D__on_cpu_sigmoid():
    _test_on_cpu(version='D', data=data, func=sigmoid)
