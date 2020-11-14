import numpy as np
from rational.get_weights import get_parameters


class Rational():
    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), version="A"):
        w_numerator, w_denominator = get_parameters(version, degrees,
                                                    approx_func)
        self.numerator = w_numerator
        self.denominator = w_denominator
        self.init_approximation = approx_func
        self.degrees = degrees
        self.version = version

        if version == "A":
            rational_func = Rational_version_A
        elif version == "B":
            rational_func = Rational_version_B
        elif version == "C":
            rational_func = Rational_version_C
        else:
            raise ValueError("version %s not implemented" % version)
        self.activation_function = rational_func

    def __call__(self, x):
        if type(x) is int:
            x = float(x)
        return self.activation_function(x, self.numerator, self.denominator)

    def torch(self, cuda=None, trainable=True, train_numerator=True,
              train_denominator=True):
        """
        Returns a torch version of this activation function
        Arguments:
                cuda (bool):
                    Use GPU CUDA version. If None, use cuda if available on the
                    machine\n
                    Default ``None``
                trainable (bool):
                    If the weights are trainable, i.e, if they are updated during
                    backward pass\n
                    Default ``True``
        Returns:
            function: Rational torch function
        """
        from rational_torch import Rational as Rational_torch
        import torch.nn as nn
        import torch
        rtorch = Rational_torch(self.init_approximation, self.degrees,
                                cuda, self.version, trainable,
                                train_numerator, train_denominator)
        rtorch.numerator = nn.Parameter(torch.FloatTensor(self.numerator)
                                        .to(rtorch.device),
                                        requires_grad=trainable and train_numerator)
        rtorch.denominator = nn.Parameter(torch.FloatTensor(self.denominator)
                                          .to(rtorch.device),
                                          requires_grad=trainable and train_denominator)
        return rtorch

    def fit(self, function, x_range=np.arange(-3., 3., 0.1), show=False):
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
        from rational.utils import find_closest_equivalent
        (a, b, c, d), distance = find_closest_equivalent(self, function,
                                                         x_range)
        if show:
            import matplotlib.pyplot as plt
            import torch
            plt.plot(x_range, self(x_range), label="Rational (self)")
            if '__name__' in dir(function):
                func_label = function.__name__
            else:
                func_label = str(function)
            result = a * function(c * torch.tensor(x_range) + d) + b
            plt.plot(x_range, result, label=f"Fitted {func_label}")
            plt.legend()
            plt.show()
        return (a, b, c, d), distance

    def __repr__(self):
        return (f"Rational Activation Function (PYTORCH version "
                f"{self.version}) of degrees {self.degrees} running on "
                f"{self.device}")


def Rational_version_A(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.ones_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += np.abs(d_array[i] * xi)
    return P/Q


def Rational_version_B(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += d_array[i] * xi
    Q = np.abs(Q) + np.ones_like(Q)
    return P/Q


def Rational_version_C(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        Q += d_array[i] * xi  # Here b0 is considered
        xi *= x
    Q = np.abs(Q) + np.full_like(Q, 0.1)
    return P/Q
