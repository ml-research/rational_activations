import tensorflow as tf

import numpy as np

from rational.keras import Rational

"""
This test file tests the keras rational activation functions on cuda devices.
The individual test methods are repetitive for the sake of comprehension. This way, they can be executed and analyzed
independently from each other.
"""

# instantiate a tensor for testing (from numpy array)
test_tensor = tf.convert_to_tensor(np.array([-2., -1, 0., 1., 2.], np.float32), np.float32)
# instantiate expected result, to be used as ground truth
expected_result = np.array([-0.02, -0.01, 0, 1, 2])


def test_a_on_cuda():
    """
    test rational activation function from keras package on test_tensor
    - version: a
    - device: cuda
    - approximated to: default
    """
    result = Rational(version='A', cuda=True)(test_tensor).clone().detach().cpu().numpy()
    assert np.all(np.isclose(result, expected_result, atol=5e-02))


def test_b_on_cuda():
    """
       test rational activation function from keras package on test_tensor
       - version: b
       - device: cuda
       - approximated to: default
       """
    result = Rational(version='B', cuda=True)(test_tensor).clone().detach().cpu().numpy()
    assert np.all(np.isclose(result, expected_result, atol=5e-02))


def test_c_on_cuda():
    """
       test rational activation function from keras package on test_tensor
       - version: c
       - device: cuda
       - approximated to: default
       """
    result = Rational(version='C', cuda=True)(test_tensor).clone().detach().cpu().numpy()
    assert np.all(np.isclose(result, expected_result, atol=5e-02))


def test_d_on_cuda():
    """
       test rational activation function from keras package on test_tensor
       - version: d
       - device: cuda
       - approximated to: default
       """
    result = Rational(version='D', cuda=True, trainable=False)(test_tensor).clone().detach().cpu().numpy()
    assert np.all(np.isclose(result, expected_result, atol=5e-02))
