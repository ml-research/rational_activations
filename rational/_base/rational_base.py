import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import warnings


class Rational_base():
    count = 0
    list = []
    _FR_WARNED = False
    _HIST_WARNED = False

    def __init__(self, name):
        super().__init__()
        self._handle_retrieve_mode = None
        self.distribution = None
        self.best_fitted_function = None
        self.best_fitted_function_params = None
        self.snapshot_list = list()
        self._verbose = True
        if name in [rat.name for rat in self.list]:
            name = _increment_string(name)
        self.name = name
        Rational_base.count += 1
        Rational_base.list.append(self)

    @classmethod
    def show_all(cls, x=None, fitted_function=True, other_func=None,
                 display=True, tolerance=0.001, title=None, axes=None,
                 layout="auto"):
        if axes is None:
            if layout == "auto":
                total = len(cls.list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout)
            except ImportError:
                print("Try install seaborn")
                fig, axes = plt.subplots(*layout)
            # if display:
            fig.tight_layout()
            for ax in axes.flatten()[len(cls.list):]:
                ax.remove()
            axes = axes[:len(cls.list)]
        for rat, ax in zip(cls.list, axes.flatten()):
            rat.show(x, fitted_function, other_func, False, tolerance,
                     None, axis=ax)
        if title is not None:
            fig.suptitle(title, y=1.02)
        if display:
            plt.legend()
            plt.show()
        else:
            return fig

    def show(self, x=None, fitted_function=True, other_func=None, display=True,
             tolerance=0.001, title=None, axis=None):
        """
        Shows a graph of the function (or returns it if ``returns=True``).

        Arguments:
                name (str):
                    Name of the snapshot.\n
                    Default ``snapshot_0``
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                display (bool):
                    If ``True``, displays the plot.
                    Otherwise, returns the figure. \n
                    Default ``False``
        """
        snap = self.capture(returns=True)
        snap.histogram = self.distribution
        if title is None:
            rats_names = [_erase_suffix(rat.name) for rat in self.list]
            if len(set(rats_names)) != 1:
                title = self.name
        if axis is None:
            fig = snap.show(x, fitted_function, other_func, display, tolerance,
                            title)
            if not display:
                return fig
            else:
                fig.show()
        else:
            snap.show(x, fitted_function, other_func, display, tolerance,
                      title, axis=axis)

    @classmethod
    def capture_all(cls, name="snapshot_0", x=None, fitted_function=True,
                    other_func=None, returns=False):
        """
        Captures snapshot for every instanciated rational
        """
        for rat in cls.list:
            rat.capture(name, x, fitted_function, other_func, returns)

    def capture(self, name="snapshot_0", x=None, fitted_function=True,
                other_func=None, returns=False):
        """
        Captures a snapshot of the rational functions and related in the
        snapshot_list variable (or returns it if ``returns=True``).

        Arguments:
                name (str):
                    Name of the snapshot.\n
                    Default ``snapshot_0``
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                returns (bool):
                    If ``True``, returns the snapshot.
                    Otherwise, saves it in self.snapshot_list \n
                    Default ``False``
        """
        while name in [snst.name for snst in self.snapshot_list] and \
              not returns:
            name = _increment_string(name)
        snapshot = Snapshot(name, self)
        if returns:
            return snapshot
        self.snapshot_list.append(snapshot)
        # np_func = self.numpy()
        # freq = None
        # if x is None and self.distribution is None:
        #     input_range = np.arange(-3, 3, 0.01)
        #     x = input_range
        # elif self.distribution is not None and len(self.distribution.bins) > 0:
        #     freq, bins = _cleared_arrays(self.distribution, tolerance)
        #     if freq is not None:
        #         input_range = np.array(bins, dtype=float)
        #         x = input_range
        # else:
        #     input_range = np.array(x, dtype=float)
        # outputs = np_func(input_range)
        # other_funcs_dict = {}
        # if other_func is not None:
        #     if type(other_func) is dict:
        #         for func_label, func in other_func.items():
        #             other_funcs_dict[func_label] = func(x)
        #     else:
        #         if type(other_func) is not list:
        #             other_func = [other_func]
        #         for func in other_func:
        #             if '__name__' in dir(func):
        #                 func_label = func.__name__
        #             else:
        #                 func_label = str(func)
        #             other_funcs_dict[func_label] = func(x)
        # if freq is None:
        #     hist_dict = None
        # else:
        #     hist_dict = {"bins": bins, "freq": freq,
        #                  "width": bins[1] - bins[0]}
        # if "best_fitted_function" not in dir(self) or self.best_fitted_function is None:
        #     fitted_function = None
        # else:
        #     a, b, c, d = self.best_fitted_function_params
        #     result = a * self.best_fitted_function(c * input_range + d) + b
        #     fitted_function = {"function": self.best_fitted_function,
        #                        "params": (a, b, c, d),
        #                        "y": result}


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
        if "rational.keras" in str(type(function)) or \
           "rational.torch" in str(type(function)):
            function = function.numpy()
        used_dist = False
        rational_numpy = self.numpy()
        if x is not None:
            (a, b, c, d), distance = rational_numpy.fit(function, x)
        else:
            if self.distribution is not None:
                freq, bins = _cleared_arrays(self.distribution)
                x = bins
                used_dist = True
            else:
                import numpy as np
                x = np.arange(-3., 3., 0.1)
            (a, b, c, d), distance = rational_numpy.fit(function, x)
        if show:
            def func(inp):
                return a * function(c * inp + d) + b

            if '__name__' in dir(function):
                func_label = function.__name__
            else:
                func_label = str(function)
            self.show(x, other_func={func_label: func})
        if self.best_fitted_function is None:
            self.best_fitted_function = function
            self.best_fitted_function_params = (a, b, c, d)
        return (a, b, c, d), distance

    def best_fit(self, functions_list, x=None, show=False):
        if self.distribution is not None:
            freq, bins = _cleared_arrays(self.distribution)
            x = bins
        (a, b, c, d), distance = self.fit(functions_list[0], x=x, show=show)
        min_dist = distance
        print(f"{functions_list[0]}: {distance:>3}")
        params = (a, b, c, d)
        final_function = functions_list[0]
        for func in functions_list[1:]:
            (a, b, c, d), distance = self.fit(func, x=x, show=show)
            print(f"{func}: {distance:>3}")
            if min_dist > distance:
                min_dist = distance
                params = (a, b, c, d)
                final_func = func
                print(f"{func} is the new best fitted function")
        self.best_fitted_function = final_func
        self.best_fitted_function_params = params
        return final_func, (a, b, c, d)

    def numpy(self):
        raise NotImplementedError("the numpy method is not implemented for",
                                  " this class, only for the mother class")

    def __repr__(self):
        if "_verbose" in dir(self) and self._verbose:
            return (f"Rational Activation Function "
                    f"{self.version}) of degrees {self.degrees} running on "
                    f"{self.device} {hex(id(self))}\n")
        else:
            return (f"Rational Activation Function "
                    f"({self.version}) of degrees {self.degrees} running on "
                    f"{self.device}")

    def export_graph(self, path="rational_function.svg", snap_number=-1,
                     other_func=None):
        if not len(self.snapshot_list):
            print("Cannot use the last snapshot as the snapshot_list \
                  is empty, making a capture with default params")
            self.capture()
        snap = self.snapshot_list[snap_number]
        snap.save(path=path, other_func=other_func)

    @classmethod
    def export_graphs(cls, path="rational_evolution.svg", together=True,
                      layout="auto", snap_number=-1, other_func=None):
        """

        """
        if together:
            for i, rat in enumerate(cls.list):
                if not len(rat.snapshot_list) > 0:
                    print(f"Cannot use the last snapshots as snapshot n {i} \
                          is empty, capturing...")
                    cls.capture_all()
                    break
            if layout == "auto":
                total = len(cls.list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout)
            except ImportError:
                warnings.warn("Seaborn not found on computer, install it for ",
                              "better visualisation")
            for rat, ax in zip(cls.list, axes.flatten()):
                snap = rat.snapshot_list[snap_number]
                snap.show(display=False, axis=ax, other_func=other_func)
            for ax in axes.flatten()[len(cls.list):]:
                ax.remove()
            fig.savefig(_repair_path(path))
            fig.clf()
        else:
            path = _path_for_multiple(path, "graphs")
            for i, rat in enumerate(cls.list):
                pos = path.rfind(".")
                new_path = f"{path[pos:]}_{i}{path[:pos]}"
                rat.export_graph(new_path)


    @classmethod
    def export_evolution_graphs(cls, path="rational_evolution.gif",
                                together=True, layout="auto", animated=True,
                                other_func=None):
        if animated:
            if together:
                nb_sn = len(cls.list[0].snapshot_list)
                if any([len(rat.snapshot_list) != nb_sn for rat in cls.list]):
                    msg = "Seems that not all rationals have the same " \
                          "number of snapshots."
                    warnings.warn(msg)
                import io
                from PIL import Image
                limits = []
                for i, rat in enumerate(cls.list):
                    if len(rat.snapshot_list) < 2:
                        msg = "Cannot save a gif as you have taken less " \
                              f"than 1 snapshot for rational n {i}"
                        print(msg)
                        return
                    limits.append(_get_frontiers(rat.snapshot_list,
                                                 other_func))
                if layout == "auto":
                    total = len(cls.list)
                    layout = _get_auto_axis_layout(total)
                if len(layout) != 2:
                    msg = 'layout should be either "auto" or a tuple of size 2'
                    raise TypeError(msg)
                fig = plt.gcf()
                fig.set_tight_layout(True)
                gif_images = []
                seaborn_installed = False
                try:
                    import seaborn as sns
                except ImportError:
                    warnings.warn("Seaborn not found on computer, install",
                                  " it for better visualisation")
                for i in range(nb_sn):
                    if seaborn_installed:
                        with sns.axes_style("whitegrid"):
                            fig, axes = plt.subplots(*layout)
                    else:
                        fig, axes = plt.subplots(*layout)
                    for rat, ax, lim in zip(cls.list, axes.flatten(), limits):
                        x_min, x_max, y_min, y_max = lim
                        input = np.arange(x_min, x_max, (x_max - x_min)/10000)
                        snap = rat.snapshot_list[i]
                        snap.show(x=input, other_func=other_func,
                                  display=False, axis=ax)
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                    for ax in axes.flatten()[len(cls.list):]:
                        ax.remove()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    gif_images.append(Image.open(buf))
                    fig.clf()
                if path[-4:] != ".gif":
                    path += ".gif"
                path = _repair_path(path)
                gif_images[0].save(path, save_all=True, duration=800, loop=0,
                                   append_images=gif_images[1:], optimize=False)
            else:
                path = _path_for_multiple(path, "graphs")
                for i, rat in enumerate(cls.list):
                    pos = path.rfind(".")
                    if pos > 0:
                        new_path = f"{path[pos:]}_{i}{path[:pos]}"
                    else:
                        new_path = f"{path}_{i}"
                    rat.export_evolution_graph(new_path, True, other_func)
        else:  # not animated
            if path[-4:] == ".gif":
                path = path[-4:] + ".svg"
            path = _path_for_multiple(path, "evolution")
            if together:
                nb_sn = len(cls.list[0].snapshot_list)
                if any([len(rat.snapshot_list) != nb_sn for rat in cls.list]):
                    msg = "Seems that not all rationals have the " \
                          "same number of snapshots."
                    warnings.warn(msg)
                for snap_number in range(nb_sn):
                    if "." in path:
                        ext = path.split(".")[-1]
                        main = ".".join(path.split(".")[:-1])
                        new_path = f"{main}_{snap_number}.{ext}"
                    else:
                        new_path = f"{path}_{snap_number}"
                    cls.export_graphs(new_path, together, layout, snap_number,
                                      other_func)
            else:
                for i, rat in enumerate(cls.list):
                    pos = path.rfind(".")
                    new_path = f"{path[pos:]}_{i}{path[:pos]}"
                    rat.export_evolution_graph(path, False, other_func)


    def export_evolution_graph(self, path="rational_evolution.gif",
                               animated=True, other_func=None):
        """
        Creates and saves an animated graph of the animated graph of the \
        function evolution based on the successive snapshots saved in \
        `snapshot_list`.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
        """
        if animated:
            import io
            from PIL import Image
            if len(self.snapshot_list) < 2:
                print("Cannot save a gif as you have taken less than 1 snapshot")
                return
            fig = plt.gcf()
            fig.set_tight_layout(True)
            x_min, x_max, y_min, y_max = _get_frontiers(self.snapshot_list,
                                                        other_func)
            input = np.arange(x_min, x_max, (x_max - x_min)/10000)
            gif_images = []
            for i, snap in enumerate(self.snapshot_list):
                fig = snap.show(x=input, other_func=other_func, display=False)
                ax0 = fig.axes[0]
                ax0.set_xlim([x_min, x_max])
                ax0.set_ylim([y_min, y_max])
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                gif_images.append(Image.open(buf))
                fig.clf()
            if path[-4:] != ".gif":
                path += ".gif"
            path = _repair_path(path)
            gif_images[0].save(path, save_all=True, duration=800, loop=0,
                               append_images=gif_images[1:], optimize=False)
        else:
            if path[-4:] == ".gif":
                path = path[-4:] + ".svg"
            new_path = _path_for_multiple(path, "evolution")
            for i, snap in enumerate(self.snapshot_list):
                snap.save(path=new_path, other_func=other_func)


class Snapshot():
    """
    Snapshot to display rational functions

    Arguments:
            name (str):
                The name of Snapshot.
            rational (Rational):
                A rational function to save
    Returns:
        Module: Rational module
    """
    def __init__(self, name, rational, other_func=None):
        self.name = name
        self.rational = rational.numpy()
        self.range = None
        self.histogram = None
        if rational.distribution is not None and \
           not rational.distribution.is_empty:
            from copy import deepcopy
            self.histogram = deepcopy(rational.distribution)
            if not Rational_base._HIST_WARNED:
                msg = "Automatically clearing the distribution after snapshot"
                warnings.warn(msg)
                Rational_base._HIST_WARNED = True
            rational.clear_hist()
        self.best_fitted_function = None
        self.other_func = other_func

    def show(self, x=None, fitted_function=True, other_func=None,
             display=True, tolerance=0.001, title=None, axis=None):
        """
        Show the function using `matplotlib`.

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                display (bool):
                    If ``True``, displays the graph.
                    Otherwise, returns a dictionary with functions informations. \n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other callable \
                    functions or a dictionary with the function name as key \
                    and the callable as value.
                tolerance (float):
                    Tolerance the bins frequency.
                    If tolerance is 0.001, every frequency smaller than 0.001 \
                    will be cutted out of the histogram.\n
                    Default ``True``
        """
        if x is not None:
            if x.dtype != float:
                x = x.astype(float)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
        elif x is None and self.range is not None:
            print("Snapshot: Using range from initialisation")
            x = self.range
        elif self.histogram is not None:
            x = np.array(self.histogram.bins, dtype=float)
        elif x is None:
            x = np.arange(-3, 3, 0.01)
        y_rat = self.rational(x)
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            warnings.warn("Seaborn not found on computer, install it for ",
                          "better visualisation")
        #  Rational
        if axis is None:
            ax = plt.gca()
        else:
            ax = axis
        ax.plot(x, y_rat, label="Rational", zorder=2)
        #  Histogram
        if self.histogram is not None:
            freq, bins = _cleared_arrays(self.histogram, tolerance)
            ax2 = ax.twinx()
            ax2.set_yticks([])
            grey_color = (0.5, 0.5, 0.5, 0.6)
            ax2.bar(bins, freq, width=bins[1] - bins[0],
                    color=grey_color, edgecolor=grey_color, alpha=0.4)
            ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
            ax.patch.set_visible(False)
        # Other funcs
        if other_func is None and self.other_func is not None:
            other_func = self.other_func
        other_funcs_dict = {}
        if other_func is not None:
            if type(other_func) is dict:
                for func_label, func in other_func.items():
                    ax.plot(x, func(x), label=func)
            else:
                if type(other_func) is not list:
                    other_func = [other_func]
                for func in other_func:
                    if '__name__' in dir(func):
                        func_label = func.__name__
                    else:
                        func_label = str(func)
                    ax.plot(x, numpify(func, x), label=func_label)
            ax.legend(loc='upper right')
        if title is None:
            if not "snapshot" in self.name:
                ax.set_title(self.name)
        else:
            ax.set_title(f"{title}")
        if axis is None:
            if display:
                plt.show()
            else:
                return plt.gcf()

    def save(self, x=None, fitted_function=True, other_func=None,
             path=None, tolerance=0.001, title=None, format="svg"):
        fig = self.show(x, fitted_function, other_func, False, tolerance,
                        title)
        if path is None:
            path = self.name + f".{format}"
        elif "." not in path:
            path += f".{format}"
        path = _repair_path(path)
        fig.savefig(path)
        fig.clf()

    def __repr__(self):
        return f"Snapshot ({self.name})"


def _cleared_arrays(hist, tolerance=0.001):
    freq, bins = hist.normalize()
    first = (freq > tolerance).argmax()
    last = - (freq > tolerance)[::-1].argmax()
    if last == 0:
        return freq[first:], bins[first:]
    return freq[first:last], bins[first:last]


def _repair_path(path):
    import os
    changed = False
    if os.path.exists(path):
        print(f'Path "{path}" exists')
        changed = True
    while os.path.exists(path):
        if "." in path:
            path_list = path.split(".")
            path_list[-2] = _increment_string(path_list[-2])
            path = '.'.join(path_list)
        else:
            path = _increment_string(path)
    if changed:
        print(f'Incremented, new path : "{path}"')
    if '/' in path:
        directory = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory):
            print(f'Path "{directory}" does not exist, creating')
            os.makedirs(directory)
    return path


def _increment_string(string):
    if string[-1] in [str(i) for i in range(10)]:
        import re
        last_number = re.findall(r'\d+', string)[-1]
        return string[:-len(last_number)] + str(int(last_number) + 1)
    else:
        return string + "_2"


def _erase_suffix(string):
    if string[-1] in [str(i) for i in range(10)]:
        return "_".join(string.split("_")[:-1])
    else:
        return string


def _get_frontiers(snapshot_list, other_func=None):
    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    for snap in snapshot_list:
        fig = snap.show(display=False, other_func=other_func)
        x_mi, x_ma = fig.axes[0].get_xlim()
        y_mi, y_ma = fig.axes[0].get_ylim()
        if x_mi < x_min:
            x_min = x_mi
        if y_mi < y_min:
            y_min = y_mi
        if x_ma > x_max:
            x_max = x_ma
        if y_ma > y_max:
            y_max = y_ma
    fig.clf()
    return x_min, x_max, y_min, y_max


def numpify(func, x):
    """
    assert that the function is called and returns a numpy array
    """
    try:
        return np.array(func(x))
    except TypeError as tper:
        if "Tensor" in str(tper):
            import torch
            return func(torch.tensor(x)).detach().numpy()
        else:
            print("Doesn't know how to handle this type of data")
            raise tper


def _get_auto_axis_layout(nb_plots):
    if nb_plots == 1:
        return 1, 1
    mid = int(np.sqrt(nb_plots))
    for i in range(mid, 1, -1):
        mod = nb_plots % i
        if mod == 0:
            return i, nb_plots // i
    if mid * (mid + 1) >= nb_plots:
        return mid, mid + 1
    return mid + 1, mid + 1


def _path_for_multiple(path, suffix):
    from os import makedirs
    if "." in path:
        path_root = ".".join(path.split(".")[:-1])
        path_ext = "." + path.split(".")[-1]
    else:
        path_root = path
        path_ext = ""
    main_part = path_root.split("/")[-1]
    save_folder = _repair_path(f"{path_root}_{suffix}")
    makedirs(save_folder)
    return f"{save_folder}/{main_part}{path_ext}"
