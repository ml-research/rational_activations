"""
This file tests that cpu calculations produce correct results.
"""
import mxnet as mx
from ..rationals import Rational
from mxnet.ndarray import LeakyReLU
from mxnet import gluon
import numpy as np



# build a small neural net containing one Rational layer
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(Rational(version='D'))
net.initialize()


def test():
    input_data = mx.nd.array([-2., -1, 0., 1., 2.])
    net(input_data)  # need to feed data for polynomial calculations before hyberdize!
    net.hybridize()  # running time: all ndarrays converted to symbols

def test_a_on_cpu_lrelu():
    _test_template(version='A', approx_func=leaky_relu, cuda=CUDA)


def test_b_on_cpu_lrelu():
    _test_template(version='B', approx_func=leaky_relu, cuda=CUDA)


def test_c_on_cpu_lrelu():
    _test_template(version='C', approx_func=leaky_relu, cuda=CUDA)


def test_d_on_cpu_lrelu():
    _test_template(version='D', approx_func=leaky_relu, cuda=CUDA)


def test_a_on_cpu_tanh():
    _test_template(version='A', approx_func=tanh, cuda=CUDA)


def test_b_on_cpu_tanh():
    _test_template(version='B', approx_func=tanh, cuda=CUDA)


def test_c_on_cpu_tanh():
    _test_template(version='C', approx_func=tanh, cuda=CUDA)


def test_d_on_cpu_tanh():
    _test_template(version='D', approx_func=tanh, cuda=CUDA)


def test_a_on_cpu_sigmoid():
    _test_template(version='A', approx_func=sigmoid, cuda=CUDA)


def test_b_on_cpu_sigmoid():
    _test_template(version='B', approx_func=sigmoid, cuda=CUDA)


def test_c_on_cpu_sigmoid():
    _test_template(version='C', approx_func=sigmoid, cuda=CUDA)


def test_d_on_cpu_sigmoid():
    _test_template(version='D', approx_func=sigmoid, cuda=CUDA)
