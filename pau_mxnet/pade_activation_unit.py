from pau.get_weights import get_parameters
from mxnet.gluon.block import HybridBlock
from mxnet import initializer, cpu, gpu
from pau_mxnet.pade_mxnet_functions import PAU_MXNET_A_F, PAU_MXNET_B_F, PAU_MXNET_C_F, PAU_MXNET_D_F


class PAU(HybridBlock):
    def __init__(self, approx_func='leaky_relu', degrees=(5, 4), cuda=False,
                 version="A", trainable=True, train_center=True,
                 train_numerator=True, train_denominator=True):
        super(PAU, self).__init__()
        center, w_numerator, w_denominator = get_parameters(version, degrees, approx_func)
        self.device = gpu() if cuda else cpu()

        with self.name_scope():
            self.center = self.params.get(name='w_center', shape=(1,),
                                          init=initializer.Constant(center),
                                          grad_req='write' if train_center and trainable else 'null')
            self.numerator = self.params.get(name='w_numerator', shape=(len(w_numerator),),
                                             init=initializer.Constant(w_numerator),
                                             grad_req='write' if train_numerator and trainable else 'null')
            self.denominator = self.params.get(name='w_denominator', shape=(len(w_denominator),),
                                               init=initializer.Constant(w_denominator),
                                               grad_req='write' if train_denominator and trainable else 'null')

        self.degrees = degrees
        self.version = version
        self.training = trainable

        self.init_approximation = approx_func

        if version == "A":
            pau_func = PAU_MXNET_A_F
        elif version == "B":
            pau_func = PAU_MXNET_B_F
        elif version == "C":
            pau_func = PAU_MXNET_C_F
        elif version == "D":
            pau_func = PAU_MXNET_D_F
        else:
            raise ValueError("version %s not implemented" % version)

        self.activation_function = pau_func

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.activation_function(x + self.center.data(self.device), self.numerator.data(self.device),
                                       self.denominator.data(self.device), self.training)
        return out
