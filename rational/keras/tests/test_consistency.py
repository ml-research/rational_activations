import tensorflow as tf

from rational.keras import Rational
import numpy as np

# instantiate a tensor for testing (from numpy array)
test_tensor = tf.convert_to_tensor(np.array([-2., -1, 0., 1., 2.], np.float32), np.float32)
# instantiate expected result, to be used as ground truth
expected_result = np.array([-0.02, -0.01, 0, 1, 2])


def _test_consistency(version: str):
    """
    test rational activation function from keras package on test_tensor,
    validating that cuda and cpu results are consistent, i.e. that there is no significant difference
    between cuda and cpu results

    :param version: which version of the function to test
    """
    # instantiate rational activation functions under test on cpu and cuda
    cpu_fut = Rational(version=version, cuda=False) if version != 'D' else Rational(version=version, cuda=False,
                                                                                    trainable=False)
    cuda_fut = Rational(version=version, cuda=True) if version != 'D' else Rational(version=version, cuda=True,
                                                                                    trainable=False)
    # run the functions under test on our test tensor
    cpu_result = cpu_fut(test_tensor).numpy()
    cuda_result = cuda_fut(test_tensor).clone().detach().cpu().numpy()

    # check that there is no significant difference between the results
    assert np.all(np.isclose(cpu_result, cuda_result, atol=1e-06))


def test_a_for_consistency():
    _test_consistency('A')


def test_b_for_consistency():
    _test_consistency('B')


def test_c_for_consistency():
    _test_consistency('C')


def test_d_for_consistency():
    _test_consistency('D')
