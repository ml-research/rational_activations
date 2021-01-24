"""
This file tests that cpu calculations produce correct results.
"""
import mxnet as mx
from ..rationals import Rational
from mxnet.ndarray import LeakyReLU
import numpy as np

# quick test on versions.py
input = mx.nd.array([-2., -1, 0., 1., 2.])
expected_res = LeakyReLU(data=input)
rational = Rational()(input).numpy()


def test():
    print('leakyrelu', expected_res)
    print('rational', rational)
    assert np.all(np.isclose(expected_res, rational, atol=5e-02))
