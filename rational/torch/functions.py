import torch
import torch.nn.functional as F
from rational.utils.find_init_weights import find_weights
from rational.utils.utils import _cleared_arrays
from rational.utils.warnings import RationalImportScipyWarning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

_2pi_sqrt = 2.5066
tanh = torch.tanh
relu = F.relu
leaky_relu = F.leaky_relu
gaussian = lambda x: torch.exp(-0.5*x**2) / _2pi_sqrt
gaussian.__name__ = "gaussian"

def _save_input(self, input, output):
    self.distribution.fill_n(input[0])


def _save_input_auto_stop(self, input, output):
    self.inputs_saved += 1
    self.distribution.fill_n(input[0])
    if self.inputs_saved > self._max_saves:
        self.training_mode()


class ActivationModule(torch.nn.Module):
    def __init__(self, function, device=None):
        print("In ActMod")
        if isinstance(function, str):
            function = None
        super().__init__()
        if function is not None:
            self.function = function
            if "__forward__" in dir(function):
                self.forward = self.function.forward
            else:
                self.forward = lambda *args, **kwargs: self.function(*args, **kwargs)
        self._handle_retrieve_mode = None
        self._saving_input = False
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.use_kde = True

    def input_retrieve_mode(self, auto_stop=False, max_saves=1000,
                            bin_width=0.1, mode="all", category_name=0):
        """
        Will retrieve the distribution of the input in self.distribution. \n
        This will slow down the function, as it has to retrieve the input \
        dist.\n

        Arguments:
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``all``
                category_name (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``0``
        """
        if self._handle_retrieve_mode is not None:
            # print("Already in retrieve mode")
            return
        if "cuda" in self.device:
            from rational.utils.histograms_cupy import Histogram
        else:
            from rational.utils.histograms_numpy import Histogram
        if mode == "all":
            self.distribution = Histogram(bin_width)
        elif mode == "categories":
            self.distribution = Histogram(bin_width)
            self.distributions = [self.distribution]
            self._current_category = category_name
            self.categories = [category_name]
        else:
            print("Unknow mode to retrieve the input")
        # print("Retrieving input from now on.")
        self._irm = mode
        self._inp_bin_width = bin_width
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
        self._handle_retrieve_mode = None

    def __repr__(self):
        if "__name__" in dir(self.function):
            return f"{self.function.__name__} ActivationModule"
        return f"{self.function} ActivationModule"

    # def show(self, x=None, figsize=None, axis=None):
    def show(self, x=None, fitted_function=True, other_func=None, display=True,
             tolerance=0.001, title=None, axis=None, writer=None, step=None,
             hist_color="#1f77b4"):
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        y = self.function(x)
        with sns.axes_style("whitegrid"):
            if axis is None:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(x, y)
        if self.distributions is not None:
            ax2 = ax.twinx()
            cmap = plt.get_cmap('Pastel1')
            for dist, category, color in zip(self.distributions, self.categories, cmap.colors):
                plot_distribution(ax2, dist, self.use_kde, category, color)
        elif self.distribution is not None:
            ax2 = ax.twinx()
            plot_distribution(ax2, self.distribution, self.use_kde, "", color="b")
        plt.show()

    @property
    def current_inp_category(self):
        return self._current_inp_category

    @current_inp_category.setter
    def current_inp_category(self, value):
        if value == self.current_inp_category:
            return
        if "cuda" in self.device:
            from rational.utils.histograms_cupy import Histogram
        else:
            from rational.utils.histograms_numpy import Histogram
        self.distribution = Histogram(self._inp_bin_width)
        self.distributions.append(self.distribution)
        self.categories.append(value)
        self._current_inp_category = value


def plot_distribution(ax, distribution, use_kde, inp_label, color):
    weights, x = _cleared_arrays(distribution, 0.001)
    ax.set_yticks([])
    try:
        import scipy.stats as sts
        scipy_imported = True
    except ImportError:
        RationalImportScipyWarning.warn()
    if use_kde and scipy_imported:
        if len(x) > 5:
            refined_bins = np.linspace(x[0], x[-1], 200)
            kde_curv = distribution.kde()(refined_bins)
            # ax.plot(refined_bins, kde_curv, lw=0.1)
            ax.fill_between(refined_bins, kde_curv, alpha=0.45,
                             color=color, label=inp_label)
        else:
            print("The bin size is too big, bins contain too few "
                  "elements.\nbins:", x)
            ax.bar([], []) # in case of remove needed
    else:
        ax.bar(x, weights/weights.max(), width=x[1] - x[0],
                linewidth=0, alpha=0.3)
    ax.set_zorder(ax.get_zorder()+1) # put a x in front of ax
    ax.patch.set_visible(False)

if __name__ == '__main__':
    gau = ActivationModule(gaussian)
    print(gau)
    gau.input_retrieve_mode(mode="categories", category_name="neg")
    inp = (torch.rand(10000)-1)*2
    gau(inp.cuda())
    gau.current_inp_category = "pos"
    inp = (torch.rand(10000)+1)*2
    gau(inp.cuda())
    gau.show(figsize=(20, 15))
