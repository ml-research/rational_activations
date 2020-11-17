from mxnet import gpu, cpu
from rational_mxnet import Rational

rational_function = Rational() # Initialized closed to Leaky ReLU
print(rational_function)
#    Pade Activation Unit (version A) of degrees (5, 4) running on cuda:0
# or Pade Activation Unit (version A) of degrees (5, 4) running on cpu

rational_function.initialize(ctx=cpu())
rational_function.initialize(ctx=gpu())

print(rational_function.degrees)
# (5, 4)
print(rational_function.version)
# A
print(rational_function.training)
# True

import mxnet.gluon.nn as nn
from mxnet import initializer


class RationalNetwork(nn.Block):
    n_features = 512

    def __init__(self, input_shape, output_shape, recurrent=False, cuda=False, **kwargs):
        super(RationalNetwork, self).__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        init = initializer.Xavier()

        self._h1 = nn.Conv2D(in_channels=n_input, channels=32, kernel_size=8, strides=4, weight_initializer=init)
        self._h2 = nn.Conv2D(in_channels=32, channels=64, kernel_size=4, strides=2, weight_initializer=init)
        self._h3 = nn.Conv2D(in_channels=64, channels=64, kernel_size=3, strides=1, weight_initializer=init)
        self._h4 = nn.Dense(in_units=3136, units=self.n_features, weight_initializer=init)
        self._h5 = nn.Dense(in_units=self.n_features, units=n_output, weight_initializer=init)

        if recurrent:
            self.act_func1 = Rational(cuda=cuda)
            self.act_func2 = self.act_func1
            self.act_func3 = self.act_func1
            self.act_func4 = self.act_func1
        else:
            self.act_func1 = Rational(cuda=cuda)
            self.act_func2 = Rational(cuda=cuda)
            self.act_func3 = Rational(cuda=cuda)
            self.act_func4 = Rational(cuda=cuda)

    def forward(self, input):
        x1 = self._h1(input)
        h = self.act_func1(x1)
        x2 = self._h2(h)
        h = self.act_func2(x2)
        x3 = self._h3(h)
        h = self.act_func3(x3)
        x4 = self._h4(h.reshape(-1, 3136))
        h = self.act_func4(x4)
        out = self._h5(h)
        return out

from mxnet import random, nd

use_cuda = False
ctx = gpu() if use_cuda else cpu()
RN = RationalNetwork((1, 84, 84), (3,), cuda=use_cuda)
RRN = RationalNetwork((1, 84, 84), (3,), recurrent=True, cuda=use_cuda)

RN.initialize(ctx=ctx)
RRN.initialize(ctx=ctx)

input_tensor = random.uniform(low=0, high=1, shape=(2, 1, 84, 84), ctx=ctx)  # Batch of 2 84x84 images (Black&White)

output_rn = RN(input_tensor)
output_rrn = RRN(input_tensor)

print(output_rn)
# [[ 0.01991099  0.04645921 -0.04806275]
#  [-0.01167973  0.05652686 -0.072688  ]]
# <NDArray 2x3 @cpu(0)>
print(output_rrn)
# [[ 0.08455121 -0.00432447  0.0079731 ]
#  [ 0.08299865 -0.01842497  0.02664253]]
# <NDArray 2x3 @cpu(0)>

import matplotlib.pyplot as plt
import numpy as np

input_tensor = nd.array(np.arange(-2, 2, 0.1))
rational_function.initialize(ctx=cpu())
lrelu = nn.LeakyReLU(0.01)

plt.plot(input_tensor.asnumpy(), rational_function(input_tensor).asnumpy(), label="rational")
plt.plot(input_tensor.asnumpy(), lrelu(input_tensor).asnumpy(), label="leaky_relu")
plt.legend()
plt.grid()
plt.show()

