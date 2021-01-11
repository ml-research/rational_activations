"""
This test file tests the keras rational activation functions on cuda devices.
The individual test methods are repetitive for the sake of comprehension. This way, they can be
executed and analyzed independently from each other.
"""
import tensorflow as tf
import numpy as np

from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid
from rational.keras.tests import _test_template

# instantiate a tensor for testing (from numpy array)
test_tensor = tf.convert_to_tensor(
    np.array([-2., -1, 0., 1., 2.], np.float32), np.float32)

"""
def _test_template(version: str, approx_func):
    """"""
    test rational activation function from keras package on test_tensor
    - device: cuda
    - approximated to: default

    :param approx_func: which function to use as initial shape
    :param version: which version of the function to test
    """"""
    # instantiate expected result, to be used as ground truth
    expected_result = _activation(approx_func, np.array([-0.02, -0.01, 0, 1, 2]))

    # instantiate a rational activation function under test
    trainable = False
    fut = Rational(version=version, cuda=True, approx_func=approx_func.__name__, trainable=trainable) \
        if version != 'D' else Rational(version=version, cuda=True, trainable=trainable, approx_func=approx_func)

    # run the function under test on our test tensor
    result = fut(test_tensor).numpy()

    # check that the result is correct (enough)
    assert np.all(np.isclose(result, expected_result, atol=5e-02))
"""
# test cuda execution
cuda = True


def test_a_on_cuda_lrelu():
    _test_template(version='A', approx_func=leaky_relu, cuda=cuda)


def test_a_on_cuda_tanh():
    _test_template(version='A', approx_func=tanh, cuda=cuda)


def test_a_on_cuda_sigmoid():
    _test_template(version='A', approx_func=sigmoid, cuda=cuda)


def test_b_on_cuda_lrelu():
    _test_template(version='B', approx_func=leaky_relu, cuda=cuda)


def test_b_on_cuda_tanh():
    _test_template(version='B', approx_func=tanh, cuda=cuda)


def test_b_on_cuda_sigmoid():
    _test_template(version='B', approx_func=sigmoid, cuda=cuda)


def test_c_on_cuda_lrelu():
    _test_template(version='C', approx_func=leaky_relu, cuda=cuda)


def test_c_on_cuda_tanh():
    _test_template(version='C', approx_func=tanh, cuda=cuda)


def test_c_on_cuda_sigmoid():
    _test_template(version='C', approx_func=sigmoid, cuda=cuda)


def test_d_on_cuda_lrelu():
    _test_template(version='D', approx_func=leaky_relu, cuda=cuda)


def test_d_on_cuda_tanh():
    _test_template(version='D', approx_func=tanh, cuda=cuda)


def test_d_on_cuda_sigmoid():
    _test_template(version='D', approx_func=sigmoid, cuda=cuda)
