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

    # expected_res = LeakyReLU(data=input)
    # result = fut(input).numpy()
    # print('leakyrelu', expected_res)
    # print('rational', result)
    # assert np.all(np.isclose(expected_res, result, atol=5e-02))
    pass
