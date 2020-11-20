"""
Padé Activation Units - Rational Activation Functions for pytorch
=================================================================

This module allows you to create Rational Neural Networks using Padé Activation
Units - Learnabe Rational activation functions.
"""
import torch.nn as nn
from torch.cuda import is_available as torch_cuda_available
from rational.utils.get_weights import get_parameters

if torch_cuda_available():
    try:
        from rational.torch.rational_cuda_functions import *
    except:
        print('error importing rational_cuda, is cuda not avialable?')

from rational.torch.rational_pytorch_functions import *


class RecurrentRational():
    """
        Recurrent rational activation function - wrapper for Rational

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
                    Use GPU CUDA version. \n
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

    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), cuda=None,
                 version="A", trainable=True, train_numerator=True,
                 train_denominator=True):
        self.rational = Rational(approx_func=approx_func,
                                 degrees=degrees,
                                 cuda=cuda,
                                 version=version,
                                 trainable=trainable,
                                 train_numerator=train_numerator,
                                 train_denominator=train_denominator)

    def __call__(self, *args, **kwargs):
        return _RecurrentRational(self.rational)


class _RecurrentRational(nn.Module):
    def __init__(self, rational):
        super(_RecurrentRational, self).__init__()
        self.rational = rational
        self._handle_retrieve_mode = None
        self.distribution = None

    def forward(self, x):
        return self.rational(x)

    def __repr__(self):
        return (f"Recurrent Rational Activation Function (PYTORCH version "
                f"{self.rational.version}) of degrees {self.rational.degrees} running on "
                f"{self.rational.device}")

    def cpu(self):
        return self.rational.cpu()

    def cuda(self):
        return self.rational.cuda()

    def numpy(self):
        return self.rational.numpy()

    def fit(self, function, x=None, show=False):
        return self.rational.fit(function=function, x=x, show=show)

    def input_retrieve_mode(self, auto_stop=True, max_saves=1000):
        """
                Will retrieve the distribution of the input in self.distribution. \n
                This will slow down the function, as it has to retrieve the input \
                dist.\n

                Arguments:
                        auto_stop (bool):
                            If True, the retrieving will stop after `max_saves` \
                            calls to forward.\n
                            Else, use :meth:`torch.Rational.training_mode`.\n
                            Default ``True``
                        max_saves (int):
                            The range on which the curves of the functions are fitted \
                            together.\n
                            Default ``1000``
                """
        from physt import h1 as hist1
        self.distribution = hist1(None, "fixed_width", bin_width=0.1,
                                  adaptive=True)
        print("Retrieving input from now on.")
        if auto_stop:
            self.inputs_saved = 0
            self._handle_retrieve_mode = self.register_forward_hook(_save_input_auto_stop)
            self._max_saves = max_saves
        else:
            self._handle_retrieve_mode = self.register_forward_hook(_save_input)

    def training_mode(self):
        """
                Stops retrieving the distribution of the input in `self.distribution`.
                """
        print("Training mode, no longer retrieving the input.")
        self._handle_retrieve_mode.remove()

    def show(self, input_range=None, display=True):
        self.rational.show(input_range=input_range, display=display, distribution=self.distribution)


class Rational(nn.Module):
    """
    Rational activation function inherited from ``torch.nn.Module``

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
                Use GPU CUDA version. \n
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
        self._handle_retrieve_mode = None
        self.distribution = None

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
        Returns a numpy version of this activation function.
        """
        from rational import Rational as Rational_numpy
        rational_n = Rational_numpy(self.init_approximation, self.degrees,
                                    self.version)
        rational_n.numerator = self.numerator.tolist()
        rational_n.denominator = self.denominator.tolist()
        return rational_n

    def fit(self, function, x=None, show=False):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent \
        function of the provided one as close as possible to this rational \
        function.

        Arguments:
                function (callable):
                    The function you want to fit to rational.\n
                x (array):
                    The range on which the curves of the functions are fitted
                    together.\n
                    Default ``True``
                show (bool):
                    If  ``True``, plots the final fitted function and \
                    rational (using matplotlib).\n
                    Default ``False``
        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function \
                (vertical and horizontal scales and bias) \n
            dist: The final distance between the rational function and the \
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
        if "center" in dir(old_rational_func) and old_rational_func.center != 0:
            print("Found a non zero center, please adapt the bias of the",
                  "previous layer to have an equivalent neural network")
        self.training = old_rational_func.training
        self.device = self.numerator.device
        if "init_approximation" not in dir("init_approximation"):
            self.init_approximation = "leaky_relu"
        else:
            self.init_approximation = old_rational_func.init_approximation
        if "cuda" in str(self.device):
            if self.version == "A":
                rational_func = Rational_CUDA_A_F
            elif self.version == "B":
                self.rational_func = Rational_CUDA_B_F
            elif self.version == "C":
                rational_func = Rational_CUDA_C_F
            elif self.version == "D":
                rational_func = Rational_CUDA_D_F
            else:
                raise ValueError("version %s not implemented" % self.version)

            self.activation_function = rational_func.apply
        else:
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

        self._handle_retrieve_mode = None
        self.distribution = None

    def input_retrieve_mode(self, auto_stop=True, max_saves=1000):
        """
        Will retrieve the distribution of the input in self.distribution. \n
        This will slow down the function, as it has to retrieve the input \
        dist.\n

        Arguments:
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``True``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
        """
        from physt import h1 as hist1
        self.distribution = hist1(None, "fixed_width", bin_width=0.1,
                                  adaptive=True)
        print("Retrieving input from now on.")
        if auto_stop:
            self.inputs_saved = 0
            self._handle_retrieve_mode = self.register_forward_hook(_save_input_auto_stop)
            self._max_saves = max_saves
        else:
            self._handle_retrieve_mode = self.register_forward_hook(_save_input)

    def training_mode(self):
        """
        Stops retrieving the distribution of the input in `self.distribution`.
        """
        print("Training mode, no longer retrieving the input.")
        self._handle_retrieve_mode.remove()

    def show(self, input_range=None, display=True, distribution=None):
        """
        Show the function using `matplotlib`.

        Arguments:
                input_range (range):
                    The range to print the function on.\n
                    Default ``None``
                display (bool):
                    If ``True``, displays the graph.
                    Otherwise, returns it. \n
                    Default ``True``
        """
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError as e:
            print("seaborn not found on computer, install it for better",
                  "visualisation")
        ax = plt.gca()
        if input_range is None:
            if distribution is None:
                distribution = self.distribution
            if distribution is None:
                input_range = torch.arange(-3, 3, 0.01, device=self.device)
            else:
                freq, bins = _cleared_arrays(distribution)
                ax2 = ax.twinx()
                ax2.set_yticks([])
                grey_color = (0.5, 0.5, 0.5, 0.6)
                ax2.bar(bins, freq, width=bins[1] - bins[0],
                        color=grey_color, edgecolor=grey_color)
                input_range = torch.tensor(bins, device=self.device).float()
        else:
            input_range = torch.tensor(input_range, device=self.device).float()
        outputs = self.activation_function(input_range, self.numerator,
                                           self.denominator, False)
        outputs_np = outputs.detach().cpu().numpy()
        ax.plot(input_range.detach().cpu().numpy(),
                outputs_np)
        if display:
            plt.show()
        else:
            return plt.gcf()


def _save_input(self, input, output):
    self.distribution.fill_n(input[0].detach().cpu().numpy())


def _save_input_auto_stop(self, input, output):
    self.inputs_saved += 1
    self.distribution.fill_n(input[0].detach().cpu().numpy())
    if self.inputs_saved > self._max_saves:
        self.training_mode()


def _cleared_arrays(hist, tolerance=0.001):
    hist = hist.normalize()
    freq, bins = hist.numpy_like
    first = (freq > 0.001).argmax()
    last = (freq > 0.001)[::-1].argmax()
    return freq[first:-last], bins[first:-last - 1]


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
