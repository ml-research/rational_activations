"""
This file contains the Rational class, the anchor class of the mxnet package
"""
from mxnet import initializer
from mxnet.gluon import HybridBlock

from rational.utils.get_weights import get_parameters
from rational.mxnet.versions import _version_a, _version_b, _version_c, _version_d


class Rational(HybridBlock):
    """
    This class implements rational activation functions for MxNet, inheriting from
    mxnet.gluon.HybridBlock.
    """

    def __init__(self, approx_func='leaky_relu', degrees=(5, 4), cuda=False,
                 version='A', trainable=True, train_numerator=True,
                 train_denominator=True):
        """
        Initializes this custom HybridBlock, which implements a Rational Activation Function.
        Sets the initial configuration of weights, according to the specified version and
        approximated function, makes the weights trainable or not, specifies on which device
        to execute (cpu or gpu) etc.
        :param approx_func: The name of the approximated function for initialisation.
        The different functions are available in `rational.rationals_config.json`.
        Default ``leaky_relu``.
        :param degrees: The degrees of the numerator (P) and denominator (Q).
        Default ``(5, 4)``
        :param cuda: whether to execute on cuda device. NOTE: THIS PARAMETER IS CURRENTLY NOT
        CONSIDERED
        :param version: Version of Rational to use. Rational(x) = P(x)/Q(x)
        `A`: Q(x) = 1 + |b_1.x| + |b_2.x| + ... + |b_n.x|
        `B`: Q(x) = 1 + |b_1.x + b_2.x + ... + b_n.x|
        `C`: Q(x) = 0.1 + |b_1.x + b_2.x + ... + b_n.x|
        `D`: like `B` with noise
        Default ``A``
        :param trainable: If the weights are trainable, i.e, if they are updated during
        backward pass.
        Default ``True``
        :param train_numerator: whether numerator coefficients are trainable
        :param train_denominator: whether denominator coefficients are trainable
        """
        super(Rational, self).__init__()

        # read initial parameter configuration from external files
        w_numerator, w_denominator = get_parameters(
            version, degrees, approx_func)

        # set specified context (currently not happening, since unclear, how and why helpful)
        # self.device = gpu() if cuda else cpu()

        # register and configure weights (numerator and denominator coefficients)
        with self.name_scope():
            self.numerator = self.params.get(name='w_numerator', shape=(len(w_numerator),),
                                             init=initializer.Constant(
                                                 w_numerator),
                                             grad_req='write' if train_numerator and trainable
                                             else 'null',
                                             differentiable=train_numerator and trainable)
            self.denominator = self.params.get(name='w_denominator', shape=(len(w_denominator),),
                                               init=initializer.Constant(
                                                   w_denominator),
                                               grad_req='write' if train_denominator and trainable
                                               else 'null',
                                               differentiable=train_denominator and trainable)

        # register whether function is trainable, since this information needs to be passed to
        # version D
        self.training = trainable

        self.init_approximation = approx_func

        # set rational activation function version
        self.rational_func = {'A': _version_a, 'B': _version_b, 'C': _version_c, 'D': _version_d} \
            .get(version)
        if self.rational_func is None:
            raise ValueError(
                "rational activation function version %s not implemented" % version)

    def hybrid_forward(self, F, x, numerator, denominator):
        return self.rational_func(F, x, numerator, denominator, self.training)
