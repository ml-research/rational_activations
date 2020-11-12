"""
Padé Activation Units - Rational Activation Functions for pytorch
=================================================================

This module allows you to create Rational Neural Networks using Padé Activation
Units - Learnabe Rational activation functions.
"""
import torch
import torch.nn as nn
from torch.cuda import is_available as torch_cuda_available
from rational.get_weights import get_parameters
from rational_torch.rational_cuda_functions import Rational_CUDA_A_F, Rational_CUDA_B_F, Rational_CUDA_C_F, \
                                 Rational_CUDA_D_F
from rational_torch.rational_pytorch_functions import Rational_PYTORCH_A_F, Rational_PYTORCH_B_F, \
                                    Rational_PYTORCH_C_F, Rational_PYTORCH_D_F


if torch_cuda_available():
    try:
        from rational_torch.rational_cuda_functions import *
    except:
        print('error importing rational_cuda, is cuda not avialable?')

from rational_torch.rational_pytorch_functions import *


class Rational(nn.Module):
    """
    Rational activation function inherited from ``torch.nn.Module``

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
                Version of Rational to use. Rational(x) = P(x)/Q(x)\n
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
        Module: Rational module
    """

    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=None,
                 version="A", trainable=True, train_numerator=True,
                 train_denominator=True):
        super(Rational, self).__init__()

        if cuda is None:
            cuda = torch_cuda_available()
        device = "cuda" if cuda else "cpu"

        w_numerator, w_denominator = get_parameters(version, degrees,
                                                    approx_func)

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
                rational_func = Rational_CUDA_A_F
            elif version == "B":
                rational_func = Rational_CUDA_B_F
            elif version == "C":
                rational_func = Rational_CUDA_C_F
            elif version == "D":
                rational_func = Rational_CUDA_D_F
            else:
                raise ValueError("version %s not implemented" % version)

            self.activation_function = rational_func.apply
        else:
            if version == "A":
                rational_func = Rational_PYTORCH_A_F
            elif version == "B":
                rational_func = Rational_PYTORCH_B_F
            elif version == "C":
                rational_func = Rational_PYTORCH_C_F
            elif version == "D":
                rational_func = Rational_PYTORCH_D_F
            else:
                raise ValueError("version %s not implemented" % version)

            self.activation_function = rational_func

    def forward(self, x):
        out = self.activation_function(x, self.numerator, self.denominator,
                                       self.training)
        return out

    def __repr__(self):
        return (f"Rational Activation Function (PYTORCH version "
                f"{self.version}) of degrees {self.degrees} running on "
                f"{self.device}")

    def cpu(self):
        if self.version == "A":
            rational_func = Rational_PYTORCH_A_F
        elif self.version == "B":
            rational_func = Rational_PYTORCH_B_F
        elif self.version == "C":
            rational_func = Rational_PYTORCH_C_F
        elif self.version == "D":
            rational_func = Rational_PYTORCH_D_F
        else:
            raise ValueError("version %s not implemented" % self.version)
        self.activation_function = rational_func
        self.device = "cpu"
        return super().cpu()

    def cuda(self):
        if self.version == "A":
            rational_func = Rational_CUDA_A_F
        elif self.version == "B":
            rational_func = Rational_CUDA_B_F
        elif self.version == "C":
            rational_func = Rational_CUDA_C_F
        elif self.version == "D":
            rational_func = Rational_CUDA_D_F
        else:
            raise ValueError("version %s not implemented" % self.version)

        self.activation_function = rational_func.apply
        self.device = "cuda"
        return super().cuda()

    def numpy(self):
        """
        Returns a numpy version of this activation function
        """
        from rational import Rational as Rational_numpy
        rational_n = Rational_numpy(self.init_approximation, self.degrees,
                     self.version)
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
                (vertical and horizontal scales and bias) \n
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


class AugmentedRational(nn.Module):
    """
    Augmented Rational activation function inherited from ``Rational``

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
                Version of Rational to use. Rational(x) = P(x)/Q(x)\n
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
        Module: Augmented Rational module
    """

    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=None,
                 version="A", trainable=True, train_numerator=True,
                 train_denominator=True):
        super(AugmentedRational, self).__init__()
        self.in_bias = nn.Parameter(torch.FloatTensor([0.0]))
        self.out_bias = nn.Parameter(torch.FloatTensor([0.0]))
        self.vertical_scale = nn.Parameter(torch.FloatTensor([1.0]))
        self.horizontal_scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, x):
        x = self.horizontal_scale * x + self.in_bias
        out = self.activation_function(x, self.numerator, self.denominator,
                                       self.training)
        return self.vertical_scale * out + self.out_bias
