import torch
import torch.nn.functional as F
from rational.utils.find_init_weights import find_weights
from rational.utils.utils import _cleared_arrays
from rational.utils.warnings import RationalImportScipyWarning
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored

_2pi_sqrt = 2.5066
tanh = torch.tanh
relu = F.relu
leaky_relu = F.leaky_relu
gaussian = lambda x: torch.exp(-0.5*x**2) / _2pi_sqrt
gaussian.__name__ = "gaussian"
lined = dict()
loc = 1
CONNECTED = False

def _save_input(self, input, output):
    self._selected_distribution.fill_n(input[0])


def _save_input_auto_stop(self, input, output):
    self.inputs_saved += 1
    self._selected_distribution.fill_n(input[0])
    if self.inputs_saved > self._max_saves:
        self.training_mode()

class Metaclass(type):
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            key_str = colored(key, "red")
            self_name_str = colored(self, "red")
            msg = colored(f"Setting new Class attribute {key_str}", "yellow") + \
                  colored(f" of {self_name_str}", "yellow")
            print(msg)
        type.__setattr__(self, key, value)


class ActivationModule(torch.nn.Module, metaclass=Metaclass):
    # histograms_colors = plt.get_cmap('Pastel1').colors
    histograms_colors = ["red", "green", "black"]


    def __init__(self, function, device=None):
        if isinstance(function, str):
            self.type = function
            function = None
        super().__init__()
        if not type(self) in self.instances:
            self.instances[type(self)] = []
        self.instances[type(self)].append(self)
        if function is not None:
            self.activation_function = function
            if "__forward__" in dir(function):
                self.forward = self.activation_function.forward
            else:
                self.forward = lambda *args, **kwargs: self.activation_function(*args, **kwargs)
        self._handle_retrieve_mode = None
        self._saving_input = False
        self.distributions = []
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.use_kde = True

    def input_retrieve_mode(self, auto_stop=False, max_saves=1000,
                            bin_width=0.01, mode="all", category_name=0):
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
            if "layer" in mode.lower():
                from rational.utils.histograms_cupy import LayerHistogram as Histogram
            else:
                from rational.utils.histograms_cupy import Histogram
        else:
            from rational.utils.histograms_numpy import Histogram
        if "categor" in mode.lower():
            self._selected_distribution_name = None
            self.categories = []
            self._selected_distribution = None
            self.distributions = []
        else:
            self._selected_distribution_name = "distribution"
            self.categories = ["distribution"]
            self._selected_distribution = Histogram(bin_width)
            self.distributions = [self._selected_distribution]
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
        if "type" in dir(self):
            return  f"{self.type} ActivationModule at {hex(id(self))}"
        if "__name__" in dir(self.activation_function):
            return f"{self.activation_function.__name__} ActivationModule at {hex(id(self))}"
        return f"{self.activation_function} ActivationModule at {hex(id(self))}"


    def show(self, x=None, fitted_function=True, other_func=None, display=True,
             tolerance=0.001, title=None, axis=None, writer=None, step=None, label=None,
             color=None):
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        elif isinstance(x, tuple) and len(x) in (2, 3):
            x = torch.arange(*x).float()
        elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            x = torch.tensor(x.float())
        with sns.axes_style("whitegrid"):
            if axis is None:
                fig, axis = plt.subplots(1, 1, figsize=(8, 6))
        if self.distributions:
            if self.distribution_display_mode in ["kde", "bar"]:
                ax2 = axis.twinx()
                if "layer" in self._irm:
                    x = self.plot_layer_distributions(ax2)
                else:
                    x = self.plot_distributions(ax2, color)
                axis.set_zorder(ax2.get_zorder()+1)
                axis.patch.set_visible(False)
            elif self.distribution_display_mode == "points":
                x0, x_last, _ = self.get_distributions_range()
                x_edges = torch.tensor([x0, x_last]).float()
                y_edges = self.forward(x_edges.to(self.device)).detach().cpu().numpy()
                axis.scatter(x_edges, y_edges, color=color)
        y = self.forward(x.to(self.device)).detach().cpu().numpy()
        if label:
            axis.plot(x, y, label=label, color=color)
        else:
            axis.plot(x, y, color=color)
        if display:
            plt.show()

    @property
    def current_inp_category(self):
        return self._selected_distribution_name

    @current_inp_category.setter
    def current_inp_category(self, value):
        if value == self._selected_distribution_name:
            return
        if "cuda" in self.device:
            if "layer" in self._irm.lower():
                from rational.utils.histograms_cupy import LayerHistogram as Histogram
            else:
                from rational.utils.histograms_cupy import Histogram
        else:
            from rational.utils.histograms_numpy import Histogram
        self._selected_distribution = Histogram(self._inp_bin_width)
        self.distributions.append(self._selected_distribution)
        self.categories.append(value)
        self._selected_distribution_name = value

    def plot_distributions(self, ax, colors=None):
        """
        Plot the distribution and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False
        dists_fb = []
        x_min, x_max = np.inf, -np.inf
        if colors is None:
            colors = self.histograms_colors
        elif not(isinstance(colors, list) or isinstance(colors, tuple)):
            colors = [colors] * len(self.distributions)
        for distribution, inp_label, color in zip(self.distributions, self.categories, colors):
            weights, x = _cleared_arrays(distribution.weights, distribution.bins, 0.0001)
            # weights, x = distribution.weights, distribution.bins
            if self.distribution_display_mode == "kde" and scipy_imported:
                if len(x) > 5:
                    refined_bins = np.linspace(x[0], x[-1], 200)
                    kde_curv = distribution.kde()(refined_bins)
                    # ax.plot(refined_bins, kde_curv, lw=0.1)
                    fill = ax.fill_between(refined_bins, kde_curv, alpha=0.45,
                                           color=color, label=inp_label)
                    fill = None
                else:
                    print("The bin size is too big, bins contain too few "
                          "elements.\nbins:", x)
                    ax.bar([], []) # in case of remove needed
            else:
                fill = ax.bar(x, weights/weights.max(), width=x[1] - x[0],
                              linewidth=0, alpha=0.3, label=inp_label)
            dists_fb.append(fill)
            x_min, x_max = min(x_min, x[0]), max(x_max, x[-1])
            size = x[1] - x[0]
        if self.distribution_display_mode in ["kde", "bar"]:
            # ax.set_zorder(ax.get_zorder()+1) # put a x in front of ax
            ax.patch.set_visible(False)
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(1)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                lined[legline] = origline
            fig = plt.gcf()
            def toggle_fill_between(event):
                leg = event.artist
                orig = lined[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)

                if vis:
                    leg.set_alpha(0.45)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_fill_between)
        if x_min == np.inf or x_max == np.inf:
            torch.arange(-3, 3, 0.01)
        return torch.arange(x_min, x_max, size)

    def plot_layer_distributions(self, ax):
        """
        Plot the layer distributions and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
        dists_fb = []
        for distribution, inp_label, color in zip(self.distributions, self.categories, self.histograms_colors):
            for n, (weights, x) in enumerate(zip(distribution.weights, distribution.bins)):
                if self.use_kde and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                        kde_curv = distribution.kde(n)(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.4,
                                                color=color, label=f"Class {inp_label} ({n})")
                    else:
                        print("The bin size is too big, bins contain too few "
                              "elements.\nbins:", x)
                        ax.bar([], []) # in case of remove needed
                else:
                    fill = ax.bar(x, weights/weights.max(), width=x[1] - x[0],
                            linewidth=0, alpha=0.4, color=color, label=f"Class {inp_label} ({n})")
                dists_fb.append(fill)
        ax.set_zorder(ax.get_zorder()+1) # put a x in front of ax
        # ax.patch.set_visible(False)
        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(1)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                lined[legline] = origline
            fig = plt.gcf()
            def toggle_fill_between(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = lined[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(1)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_fill_between)
            return torch.arange(*self.get_distributions_range())

    def get_distributions_range(self):
        x_min, x_max = np.inf, -np.inf
        for dist in self.distributions:
            if not dist.is_empty:
                x_min, x_max = min(x_min, dist.range[0]), max(x_max, dist.range[-1])
                size = dist.range[1] - dist.range[0]
        if x_min == np.inf or x_max == np.inf:
            return -3, 3, 0.01
        return x_min, x_max, size

    def __setattr__(self, key, value):
        if not hasattr(self, key):
            key_str = colored(key, "red")
            self_name_str = colored(self.__class__, "red")
            msg = colored(f"Setting new attribute {key_str}", "yellow") + \
                  colored(f" of instance of {self_name_str}", "yellow")
            print(msg)
        object.__setattr__(self, key, value)


    # def load_state_dict(self, state_dict):
    #     if "distributions" in state_dict.keys():
    #         _distributions = state_dict.pop("distributions")
    #         if "cuda" in self.device and _cupy_installed():
    #             msg = f"Loading input distributions on {self.device} using cupy"
    #             RationalLoadWarning.warn(msg)
    #             self.distributions = _distributions
    #     super().load_state_dict(state_dict)
    #
    # def state_dict(self, destination=None, *args, **kwargs):
    #     _state_dict = super().state_dict(destination, *args, **kwargs)
    #     if self.distributions is not None:
    #         _state_dict["distributions"] = self.distributions
    #     return _state_dict




if __name__ == '__main__':
    gau = ActivationModule(gaussian)
    print(gau)
    gau.input_retrieve_mode(mode="categories", category_name="neg")
    inp = (torch.rand(10000)-1)*2
    gau(inp.cuda())
    gau.current_inp_category = "pos"
    inp = (torch.rand(10000)+1)*2
    gau(inp.cuda())
    gau.show()
