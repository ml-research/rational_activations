import torch
import torch.nn.functional as F
from rational.utils.find_init_weights import find_weights
from rational.utils.utils import _cleared_arrays
from rational.torch.rationals import _save_input, _save_input_auto_stop
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


class ActivationModule(torch.nn.Module):
    def __init__(self, function, cuda=None):
        super().__init__()
        self.function = function
        self._handle_retrieve_mode = None
        self._saving_input = False
        if cuda is None:
            self.cuda = torch.cuda.is_available()
            self.device = "cuda"
        else:
            self.cuda = cuda
            device = cuda
        self.use_kde = True

    def forward(self, x):
        return self.function(x)

    def input_retrieve_mode(self, auto_stop=False, max_saves=1000,
                            bin_width=0.1):
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
        """
        if self._handle_retrieve_mode is not None:
            # print("Already in retrieve mode")
            return
        if "cuda" in self.device:
            from rational.utils.histograms_cupy import Histogram
        else:
            from rational.utils.histograms_numpy import Histogram
        self.distribution = Histogram(bin_width)
        # print("Retrieving input from now on.")
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

    def show(self, x=None):
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        y = self.function(x)
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1)
            ax.plot(x, y)
        if self.distribution is not None:
            weights = _cleared_arrays(self.distribution, 0.001)[1]
            ax2 = ax.twinx()
            ax2.set_yticks([])
            try:
                import scipy.stats as sts
                scipy_imported = True
            except ImportError:
                RationalImportScipyWarning.warn()
            if self.use_kde and scipy_imported:
                if len(x) > 5:
                    refined_bins = np.linspace(x[0], x[-1], 200)
                    kde_curv = self.distribution.kde()(refined_bins)
                    # ax2.plot(refined_bins, kde_curv, lw=0.1)
                    ax2.fill_between(refined_bins, kde_curv, alpha=0.15,
                                     color="b")
                else:
                    print("The bin size is too big, bins contain too few "
                          "elements.\nbins:", x)
                    ax2.bar([], []) # in case of remove needed
            else:
                ax2.bar(x, weights/weights.max(), width=x[1] - x[0],
                        linewidth=0, alpha=0.3)
            ax.set_zorder(ax2.get_zorder()+1) # put a x in front of ax2
            ax.patch.set_visible(False)
        plt.show()


if __name__ == '__main__':
    gau = InputRetriever(gaussian)
    print(gau)
    inp = (torch.rand(10000)-0.5)*2
    gau.input_retrieve_mode()
    gau(inp.cuda())
    gau.show()
