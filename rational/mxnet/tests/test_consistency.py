"""
this file tests that cuda and cpu results are consistent
"""

from mxnet.ndarray import LeakyReLU, tanh, sigmoid
from numpy import all, isclose

from rational.mxnet import Rational

# instantiate a tensor for testing (from numpy array)
# test_tensor = tf.convert_to_tensor(
#    np.array([-2., -1, 0., 1., 2.], np.float32), np.float32)
# TODO


def _test_consistency(version: str, approx_func, sym: bool):
    """
    test rational activation function from keras package on test_tensor,
    validating that cuda and cpu results are consistent, i.e. that there is no significant
    difference between cuda and cpu results

    :param sym: use symbolic execution if True, else imperative execution
    :param approx_func: which function to use as initial shape
    :param version: which version of the function to test
    """

    # instantiate rational activation functions under test on cpu and cuda
    # TODO
    '''
    trainable = False
    cpu_fut = Rational(version=version, cuda=False, approx_func=approx_func.__name__) \
        if version != 'D' else Rational(version=version, cuda=False,
                                        approx_func=approx_func.__name__, trainable=trainable)

    cuda_fut = Rational(version=version, cuda=True, approx_func=approx_func.__name__) \
        if version != 'D' else Rational(version=version, cuda=True,
                                        approx_func=approx_func.__name__, trainable=trainable)
    # run the functions under test on our test tensor
    cpu_result = cpu_fut(test_tensor).numpy()
    cuda_result = cuda_fut(test_tensor).numpy()
    '''
    # check that there is no significant difference between the results
    assert all(isclose(cpu_result, cuda_result, atol=1e-06))


def test_a_for_consistency_lrelu_nd():
    _test_consistency(version='A', approx_func=LeakyReLU, sym=False)


def test_a_for_consistency_tanh_nd():
    _test_consistency(version='A', approx_func=tanh, sym=False)


def test_a_for_consistency_sigmoid_nd():
    _test_consistency(version='A', approx_func=sigmoid, sym=False)


def test_b_for_consistency_lrelu_nd():
    _test_consistency(version='B', approx_func=LeakyReLU, sym=False)


def test_b_for_consistency_tanh_nd():
    _test_consistency(version='B', approx_func=tanh, sym=False)


def test_b_for_consistency_sigmoid_nd():
    _test_consistency(version='B', approx_func=sigmoid, sym=False)


def test_c_for_consistency_lrelu_nd():
    _test_consistency(version='C', approx_func=LeakyReLU, sym=False)


def test_c_for_consistency_tanh_nd():
    _test_consistency(version='C', approx_func=tanh, sym=False)


def test_c_for_consistency_sigmoid_nd():
    _test_consistency(version='C', approx_func=sigmoid, sym=False)


def test_d_for_consistency_lrelu_nd():
    _test_consistency(version='D', approx_func=LeakyReLU, sym=False)


def test_d_for_consistency_tanh_nd():
    _test_consistency(version='D', approx_func=tanh, sym=False)


def test_d_for_consistency_sigmoid_nd():
    _test_consistency(version='D', approx_func=sigmoid, sym=False)


def test_a_for_consistency_lrelu_sym():
    _test_consistency(version='A', approx_func=LeakyReLU, sym=True)


def test_a_for_consistency_tanh_sym():
    _test_consistency(version='A', approx_func=tanh, sym=True)


def test_a_for_consistency_sigmoid_sym():
    _test_consistency(version='A', approx_func=sigmoid, sym=True)


def test_b_for_consistency_lrelu_sym():
    _test_consistency(version='B', approx_func=LeakyReLU, sym=True)


def test_b_for_consistency_tanh_sym():
    _test_consistency(version='B', approx_func=tanh, sym=True)


def test_b_for_consistency_sigmoid_sym():
    _test_consistency(version='B', approx_func=sigmoid, sym=True)


def test_c_for_consistency_lrelu_sym():
    _test_consistency(version='C', approx_func=LeakyReLU, sym=True)


def test_c_for_consistency_tanh_sym():
    _test_consistency(version='C', approx_func=tanh, sym=True)


def test_c_for_consistency_sigmoid_sym():
    _test_consistency(version='C', approx_func=sigmoid, sym=True)


def test_d_for_consistency_lrelu_sym():
    _test_consistency(version='D', approx_func=LeakyReLU, sym=True)


def test_d_for_consistency_tanh_sym():
    _test_consistency(version='D', approx_func=tanh, sym=True)


def test_d_for_consistency_sigmoid_sym():
    _test_consistency(version='D', approx_func=sigmoid, sym=True)
