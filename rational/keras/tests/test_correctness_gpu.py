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


def _test_on_cuda(version: str):
    """
    test rational activation function from keras package on test_tensor
    - device: cuda
    - approximated to: default

    :param version: which version of the function to test
    """
    # instantiate a rational activation function under test
    fut = Rational(version=version, cuda=True) if version != 'D' else Rational(version=version, cuda=True,
                                                                               trainable=False)

    # run the function under test on our test tensor
    result = fut(test_tensor).clone().detach().cpu().numpy()

    # check that the result is correct (enough)
    assert np.all(np.isclose(result, expected_result, atol=5e-02))


def test_a_on_cuda():
    _test_on_cuda('A')


def test_b_on_cuda():
    _test_on_cuda('B')


def test_c_on_cuda():
    _test_on_cuda('C')


def test_d_on_cuda():
    _test_on_cuda('D')
