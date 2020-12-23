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
    validating that cuda and cpu results are consistent

    :param version: which version of the function to test
    """
    cpu_fut = Rational(version=version, cuda=False) if version != 'D' else Rational(version=version, cuda=False,
                                                                                    trainable=False)
    cuda_fut = Rational(version=version, cuda=True) if version != 'D' else Rational(version=version, cuda=True,
                                                                                    trainable=False)

    cpu_result = cpu_fut(test_tensor).numpy()
    cuda_result = cuda_fut(test_tensor).clone().detach().cpu().numpy()
    assert np.all(np.isclose(cpu_result, cuda_result, atol=1e-06))


def test_a_for_consistency():
    _test_consistency('A')


def test_b_for_consistency():
    _test_consistency('B')


def test_c_for_consistency():
    _test_consistency('C')


def test_d_for_consistency():
    _test_consistency('D')


def _test_conversion_to_cpu(version: str):
    """
    this method instantiates a rational activation function on a cuda device, moves it to a cpu and validates that is
    executes in the same manner

    :param version: which version of the function to test
    """
    # TODO rewrite for keras
    # instantiate a function under test on cuda
    fut = Rational(version=version, cuda=True) if version != 'D' else Rational(version=version, cuda=True,
                                                                               trainable=False)
    # move the function to cpu
    fut.cpu()
    # check that all parameters of the function are on the cpu as well
    assert (
        np.all(
            [str(parameter.device) == 'cpu'
             for parameter in fut.parameters()]))

    # check that the activation function version is correct by calling its qualified name
    assert ("KERAS_" + version in fut.activation_function.__qualname__)

    # call function under test on the test tensor
    result = fut(test_tensor).detach().numpy()
    # check that the results are as expected
    assert (np.all(np.isclose(result, expected_result, atol=5e-02)))


def test_a_conversion_to_cpu():
    _test_conversion_to_cpu('A')


def test_b_conversion_to_cpu():
    _test_conversion_to_cpu('B')


def test_c_conversion_to_cpu():
    _test_conversion_to_cpu('C')


def test_d_conversion_to_cpu():
    _test_conversion_to_cpu('D')


def _test_conversion_to_cuda(version: str):
    """
    this method instantiates a rational activation function on a cpu, moves it to a cuda device and validates that is
    executes in the same manner

    :param version: which version of the function to test
    """
    # TODO rewrite for keras
    # instantiate function under test on cpu
    fut = Rational(version=version, cuda=False) if version != 'D' else Rational(version=version, cuda=False,
                                                                                trainable=False)
    # move the function to cuda device
    fut.cuda()
    # check that all parameters of the function are on the cpu as well
    assert (np.all(
        ['cuda' in str(parameter.device)
         for parameter in fut.parameters()]))

    # check that the activation function version is correct by calling its qualified name
    assert ("CUDA_" + version in fut.activation_function.__qualname__)

    # call function under test on the test tensor
    new_res = fut(test_tensor).clone().detach().cpu().numpy()
    # check that the results are as expected
    assert (np.all(np.isclose(new_res, expected_result, atol=5e-02)))


def test_a_conversion_to_cuda():
    _test_conversion_to_cuda('A')


def test_b_conversion_to_cuda():
    _test_conversion_to_cuda('B')


def test_c_conversion_to_cuda():
    _test_conversion_to_cuda('C')


def test_d_conversion_to_cuda():
    _test_conversion_to_cuda('D')
