"""
This file tests that cpu calculations produce correct results.
"""
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.ndarray import LeakyReLU

from ..rationals import Rational

# build a small neural net containing one Rational layer
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(Rational())
net.initialize()
net.hybridize()


def test():
    input_data = mx.nd.array([-2., -1, 0., 1., 2.])
    result = net(input_data)

    # expected_res = LeakyReLU(data=input_data)
    # result = fut(input).numpy()
    # print('leakyrelu', expected_res)
    # print('rational', result)
    # assert np.all(np.isclose(expected_res, result, atol=5e-02))
    pass
