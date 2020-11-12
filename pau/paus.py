import numpy as np
from pau.get_weights import get_parameters


class PAU():
    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), version="A"):
        center, w_numerator, w_denominator = get_parameters(version, degrees,
                                                            approx_func)
        self.center = center
        self.numerator = w_numerator
        self.denominator = w_denominator
        self.init_approximation = approx_func
        self.degrees = degrees
        self.version = version

        if version == "A":
            pau_func = PAU_version_A
        elif version == "B":
            pau_func = PAU_version_B
        elif version == "C":
            pau_func = PAU_version_C
        else:
            raise ValueError("version %s not implemented" % version)
        self.activation_function = pau_func

    def __call__(self, x):
        if type(x) is int:
            x = float(x)
        return self.activation_function(x, self.numerator, self.denominator)

    def torch(self, cuda=None, trainable=True, train_center=True,
                 train_numerator=True, train_denominator=True):
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
            function: PAU torch function
        """
        from pau_torch import PAU as PAU_torch
        import torch.nn as nn
        import torch
        pau_torch = PAU_torch(self.init_approximation, self.degrees, cuda,
                              self.version, trainable, train_center,
                              train_numerator, train_denominator)
        pau_torch.center = nn.Parameter(torch.FloatTensor([self.center])
                                        .to(pau_torch.device),
                                        requires_grad=trainable and train_center)
        pau_torch.numerator = nn.Parameter(torch.FloatTensor(self.numerator)
                                           .to(pau_torch.device),
                                           requires_grad=trainable and train_numerator)
        pau_torch.denominator = nn.Parameter(torch.FloatTensor(self.denominator)
                                             .to(pau_torch.device),
                                             requires_grad=trainable and train_denominator)
        return pau_torch

    def fit(self, function, x_range=np.arange(-3., 3., 0.1), show=False):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent
        function of the provided one as close as possible to this pau function.
        Arguments:
                function (callable):
                    The function you want to fit to pau\n
                x (array):
                    The range on which the curves of the functions are fitted
                    together
                    Default ``True``
                show (bool):
                    If  ``True``, plots the final fitted function and pau.
                    (using matplotlib)\n
                    Default ``False``
        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function
                (vertical and horizontal scales) \n
            dist: The final distance between the rational function and the
            fitted one
        """
        from pau.utils import find_closest_equivalent
        (a, b, c, d), distance = find_closest_equivalent(self, function,
                                                         x_range)
        if show:
            import matplotlib.pyplot as plt
            import torch
            plt.plot(x_range, self(x_range), label="PAU (self)")
            if '__name__' in dir(function):
                func_label = function.__name__
            else:
                func_label = str(function)
            result = a * function(c * torch.tensor(x_range) + d) + b
            plt.plot(x_range, result, label=f"Fitted {func_label}")
            plt.legend()
            plt.show()
        return (a, b, c, d), distance


def PAU_version_A(x, w_array, d_array):
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


def PAU_version_B(x, w_array, d_array):
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


def PAU_version_C(x, w_array, d_array):
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
