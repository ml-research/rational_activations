"""
TODO explain what this file is doing
"""
from rational.utils.get_weights import get_parameters
from mxnet.gluon.block import HybridBlock
from mxnet import initializer, cpu, gpu
from rational.mxnet.versions import _version_a, _version_b, _version_c, _version_d


class Rational(HybridBlock):
    """
    Rational activation function inherited from mxnet.gluon ``HybridBlock``

    Arguments:
            approx_func (str):
                The name of the approximated function for initialisation. \
                The different initialable functions are available in \
                `rational.rationals_config.json`. \n
                Default ``leaky_relu``.
            degrees (tuple of int):
                The degrees of the numerator (P) and denominator (Q).\n
                Default ``(5, 4)``
            cuda (bool):
                Use GPU version. \n
                If ``None``, use cuda if available on the machine\n
                Default ``None``
            version (str):
                Version of Rational to use. Rational(x) = P(x)/Q(x)\n
                `A`: Q(x) = 1 + \|b_1.x\| + \|b_2.x\| + ... + \|b_n.x\|\n
                `B`: Q(x) = 1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `C`: Q(x) = 0.1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `D`: like `B` with noise\n
                Default ``A``
            trainable (bool):
                If the weights are trainable, i.e, if they are updated during \
                backward pass\n
                Default ``True``
    Returns:
        Module: Rational module
    """

    def __init__(self, approx_func='leaky_relu', degrees=(5, 4), cuda=False,
                 version='A', trainable=True, train_numerator=True,
                 train_denominator=True):
        super(Rational, self).__init__()
        w_numerator, w_denominator = get_parameters(version, degrees, approx_func)
        self.device = gpu() if cuda else cpu()

        with self.name_scope():
            self.numerator = self.params.get(name='w_numerator', shape=(len(w_numerator),),
                                             init=initializer.Constant(w_numerator),
                                             grad_req='write' if train_numerator and trainable else 'null')
            self.denominator = self.params.get(name='w_denominator', shape=(len(w_denominator),),
                                               init=initializer.Constant(w_denominator),
                                               grad_req='write' if train_denominator and trainable else 'null')

        self.degrees = degrees
        self.training = trainable

        self.init_approximation = approx_func

        # set rational activation function version
        self.rational_func = {'A': _version_a, 'B': _version_b, 'C': _version_c, 'D': _version_d} \
            .get(version)
        if self.rational_func is None:
            raise ValueError("rational activation function version %s not implemented" % version)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.rational_func(x, self.numerator.data(self.device),
                                  self.denominator.data(self.device), self.training)
