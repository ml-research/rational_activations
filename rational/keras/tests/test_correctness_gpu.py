"""
This file tests that cuda calculations produce correct results.
"""
from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid

from rational.keras.tests import test_template

# test cuda execution
CUDA = True


def test_a_on_cuda_lrelu():
    test_template(version='A', approx_func=leaky_relu, cuda=CUDA)


def test_a_on_cuda_tanh():
    test_template(version='A', approx_func=tanh, cuda=CUDA)


def test_a_on_cuda_sigmoid():
    test_template(version='A', approx_func=sigmoid, cuda=CUDA)


def test_b_on_cuda_lrelu():
    test_template(version='B', approx_func=leaky_relu, cuda=CUDA)


def test_b_on_cuda_tanh():
    test_template(version='B', approx_func=tanh, cuda=CUDA)


def test_b_on_cuda_sigmoid():
    test_template(version='B', approx_func=sigmoid, cuda=CUDA)


def test_c_on_cuda_lrelu():
    test_template(version='C', approx_func=leaky_relu, cuda=CUDA)


def test_c_on_cuda_tanh():
    test_template(version='C', approx_func=tanh, cuda=CUDA)


def test_c_on_cuda_sigmoid():
    test_template(version='C', approx_func=sigmoid, cuda=CUDA)


def test_d_on_cuda_lrelu():
    test_template(version='D', approx_func=leaky_relu, cuda=CUDA)


def test_d_on_cuda_tanh():
    test_template(version='D', approx_func=tanh, cuda=CUDA)


def test_d_on_cuda_sigmoid():
    test_template(version='D', approx_func=sigmoid, cuda=CUDA)
