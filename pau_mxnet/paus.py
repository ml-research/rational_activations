from pau.get_weights import get_parameters
from mxnet.gluon.parameter import Parameter
from mxnet.gluon.nn as nn


class PAU(nn.Activation):
    def __init__(self, approx_func="leaky_relu", degrees=(5, 4),
                 version="A", trainable=True, train_center=True,
                 train_numerator=True, train_denominator=True):

        center, w_numerator, w_denominator = get_parameters(version, degrees,
                                                            approx_func)
        self.center = center
        self.numerator = Parameter(w_numerator)
        self.denominator = Paramete(rw_denominator)
        self.degrees = degrees
        self.version = version
        self.training = trainable

        self.init_approximation = approx_func
