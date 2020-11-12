"""
Padé Activation Units - Rational Activation Functions for pytorch
=================================================================

This module allows you to create Rational Neural Networks using Padé Activation
Units - Learnabe Rational activation functions.
"""

import torch.nn as nn
from torch.cuda import is_available as torch_cuda_available
from rational.get_weights import get_parameters
from rational_torch.pade_cuda_functions import PAU_CUDA_A_F, PAU_CUDA_B_F, PAU_CUDA_C_F, \
                                 PAU_CUDA_D_F
from rational_torch.pade_pytorch_functions import PAU_PYTORCH_A_F, PAU_PYTORCH_B_F, \
                                    PAU_PYTORCH_C_F, PAU_PYTORCH_D_F


if torch_cuda_available():
    try:
        from rational_torch.pade_cuda_functions import *
    except:
        print('error importing pade_cuda, is cuda not avialable?')

from rational_torch.pade_pytorch_functions import *


class Rational(nn.Module):
    """
    PAU activation function inherited from ``torch.nn.Module``

    Arguments:
            approx_func (str):
                The name of the approximated function for initialisation. \
                The different initialable functions are available in
                `rational.rationals_config.json`. \n
                Default ``leaky_relu``.
            degrees (tuple of int):
                The degrees of the numerator (P) and denominator (Q).\n
                Default ``(5, 4)``
            cuda (bool):
                Use GPU CUDA version. If None, use cuda if available on the
                machine\n
                Default ``None``
            version (str):
                Version of PAU to use. PAU(x) = P(x)/Q(x)\n
                `A`: Q(x) = 1 + \|b_1.x\| + \|b_2.x\| + ... + \|b_n.x\|\n
                `B`: Q(x) = 1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `C`: Q(x) = 0.1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `D`: like `B` with noise\n
                Default ``A``
            trainable (bool):
                If the weights are trainable, i.e, if they are updated during
                backward pass\n
                Default ``True``
    Returns:
        Module: PAU module
    """

    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=None,
                 version="A", trainable=True, train_center=True,
                 train_numerator=True, train_denominator=True):
        super(PAU, self).__init__()

        if cuda is None:
            cuda = torch_cuda_available()
        device = "cuda" if cuda else "cpu"

        center, w_numerator, w_denominator = get_parameters(version, degrees,
                                                            approx_func)

        self.center = nn.Parameter(torch.FloatTensor([center]).to(device),
                                   requires_grad=trainable and train_center)
        self.numerator = nn.Parameter(torch.FloatTensor(w_numerator).to(device),
                                      requires_grad=trainable and train_numerator)
        self.denominator = nn.Parameter(torch.FloatTensor(w_denominator).to(device),
                                        requires_grad=trainable and train_denominator)
        self.degrees = degrees
        self.version = version
        self.training = trainable
        self.device = device

        self.init_approximation = approx_func

        if cuda:
            if version == "A":
                rational_func = PAU_CUDA_A_F
            elif version == "B":
                rational_func = PAU_CUDA_B_F
            elif version == "C":
                rational_func = PAU_CUDA_C_F
            elif version == "D":
                rational_func = PAU_CUDA_D_F
            else:
                raise ValueError("version %s not implemented" % version)

            self.activation_function = rational_func.apply
        else:
            if version == "A":
                rational_func = PAU_PYTORCH_A_F
            elif version == "B":
                rational_func = PAU_PYTORCH_B_F
            elif version == "C":
                rational_func = PAU_PYTORCH_C_F
            elif version == "D":
                rational_func = PAU_PYTORCH_D_F
            else:
                raise ValueError("version %s not implemented" % version)

            self.activation_function = rational_func

    def forward(self, x):
        out = self.activation_function(x + self.center, self.numerator,
                                       self.denominator, self.training)
        return out

    def __repr__(self):
        return (f"Rational Activation Function (PYTORCH version "
               f"{self.version}) of degrees {self.degrees} running on "
               f"{self.center.device}")

    def cpu(self):
        if self.version == "A":
            rational_func = PAU_PYTORCH_A_F
        elif self.version == "B":
            rational_func = PAU_PYTORCH_B_F
        elif self.version == "C":
            rational_func = PAU_PYTORCH_C_F
        elif self.version == "D":
            rational_func = PAU_PYTORCH_D_F
        else:
            raise ValueError("version %s not implemented" % self.version)
        self.activation_function = rational_func
        self.device = "cpu"
        return super().cpu()

    def cuda(self):
        if self.version == "A":
            rational_func = PAU_CUDA_A_F
        elif self.version == "B":
            rational_func = PAU_CUDA_B_F
        elif self.version == "C":
            rational_func = PAU_CUDA_C_F
        elif self.version == "D":
            rational_func = PAU_CUDA_D_F
        else:
            raise ValueError("version %s not implemented" % self.version)

        self.activation_function = rational_func.apply
        self.device = "cuda"
        return super().cuda()

    def numpy(self):
        """
        Returns a numpy version of this activation function
        """
        from rational import PAU as PAU_numpy
        rational_n = PAU_numpy(self.init_approximation, self.degrees, self.version)
        rational_n.center = self.center.tolist()[0]
        rational_n.numerator = self.numerator.tolist()
        rational_n.denominator = self.denominator.tolist()
        return rational_n

    def fit(self, function, x=None, show=False):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent
        function of the provided one as close as possible to this rational function.
        Arguments:
                function (callable):
                    The function you want to fit to rational\n
                x (array):
                    The range on which the curves of the functions are fitted
                    together
                    Default ``True``
                show (bool):
                    If  ``True``, plots the final fitted function and rational.
                    (using matplotlib)\n
                    Default ``False``
        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function
                (vertical and horizontal scales) \n
            dist: The final distance between the rational function and the
            fitted one
        """
        rational_numpy = self.numpy()
        if x is not None:
            rational_numpy.fit(function, x, show)
        else:
            rational_numpy.fit(function, show=show)

    def _from_old(self, old_rational_func):
        self.version = old_rational_func.version
        self.degrees = old_rational_func.degrees
        self.center = old_rational_func.center
        self.numerator = old_rational_func.numerator
        self.denominator = old_rational_func.denominator
        self.training = old_rational_func.training
        if "init_approximation" not in dir("init_approximation"):
            self.init_approximation = "leaky_relu"
        else:
            self.init_approximation = old_rational_func.init_approximation
        self.activation_function = old_rational_func.activation_function


    def retrieve_input(self):
        self.histogram = []
        self.buffer = torch.Tensor().cuda()
        print("retrieving input from now on.")
        self.register_forward_hook(_save_input)

def _save_input(self, input, output):
    import ipdb; ipdb.set_trace()
    self.buffer = torch.cat((self.buffer, input[0]), 0)
